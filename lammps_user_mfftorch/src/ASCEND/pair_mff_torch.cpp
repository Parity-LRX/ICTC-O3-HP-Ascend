#include "pair_mff_torch.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "utils.h"
#include "mff_periodic_table.h"

#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <numeric>
#include <stdexcept>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace LAMMPS_NS;
namespace {
struct Gil {
  PyGILState_STATE state;
  Gil() : state(PyGILState_Ensure()) {}
  ~Gil() { PyGILState_Release(state); }
};
[[noreturn]] void python_error(const char *where) {
  PyObject *type=nullptr,*value=nullptr,*trace=nullptr;
  PyErr_Fetch(&type,&value,&trace);PyErr_NormalizeException(&type,&value,&trace);
  std::string detail=where;
  PyObject *module=PyImport_ImportModule("traceback");
  PyObject *lines=module?PyObject_CallMethod(module,"format_exception","OOO",
      type?type:Py_None,value?value:Py_None,trace?trace:Py_None):nullptr;
  PyObject *separator=PyUnicode_FromString("");
  PyObject *text=lines?PyUnicode_Join(separator,lines):nullptr;
  if(text)detail += std::string(": ")+PyUnicode_AsUTF8(text);
  else if(value) {PyObject *s=PyObject_Str(value);if(s){detail+=std::string(": ")+PyUnicode_AsUTF8(s);Py_DECREF(s);}}
  Py_XDECREF(type);Py_XDECREF(value);Py_XDECREF(trace);Py_XDECREF(module);
  Py_XDECREF(lines);Py_XDECREF(separator);Py_XDECREF(text);PyErr_Clear();
  throw std::runtime_error(detail);
}
struct Ref {
  PyObject *value;
  explicit Ref(PyObject *p) : value(p) {if(!p)python_error("Python call failed");}
  ~Ref(){Py_DECREF(value);}
  operator PyObject*()const{return value;}
};
template<class T> PyObject *view(std::vector<T>& value) {
  static char empty=0;
  return PyMemoryView_FromMemory(value.empty()?&empty:reinterpret_cast<char*>(value.data()),
                                value.size()*sizeof(T),PyBUF_READ);
}
PyObject *cell_view(float *cell) {
  return PyMemoryView_FromMemory(reinterpret_cast<char*>(cell),9*sizeof(float),PyBUF_READ);
}
void buffer_copy(PyObject *object,float *target,size_t count) {
  Py_buffer buffer;
  if(PyObject_GetBuffer(object,&buffer,PyBUF_CONTIG_RO)<0)python_error("Expected contiguous output buffer");
  if(buffer.len!=static_cast<Py_ssize_t>(count*sizeof(float))) {
    PyBuffer_Release(&buffer);throw std::runtime_error("Invalid Ascend output buffer size");
  }
  if(count)std::memcpy(target,buffer.buf,count*sizeof(float));
  PyBuffer_Release(&buffer);
}
int free_port() {
  int fd=socket(AF_INET,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("Cannot create HCCL bootstrap socket");
  sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_ANY);
  if(bind(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))<0){close(fd);throw std::runtime_error("Cannot reserve HCCL bootstrap port");}
  socklen_t length=sizeof(address);getsockname(fd,reinterpret_cast<sockaddr*>(&address),&length);
  const int port=ntohs(address.sin_port);close(fd);return port;
}
void interpreter_exit_gil() {
  // Torch's shared-library static objects can retain Python type references.
  // Acquire the GIL before their process-exit destructors execute. Finalizing
  // Python while Torch/NPU shared libraries are still loaded is unsafe.
  if(Py_IsInitialized() && !PyGILState_Check())(void)PyGILState_Ensure();
}
}

PairMFFTorch::PairMFFTorch(LAMMPS *lmp) : Pair(lmp) {
  restartinfo=0;one_coeff=1;manybody_flag=1;single_enable=0;
  comm_forward=2;comm_reverse=0;no_virial_fdotr_compute=1;
}
PairMFFTorch::~PairMFFTorch() {
  if(engine_ && Py_IsInitialized()) {
    Gil gil;
    PyObject *host_stats=Py_BuildValue("{s:d,s:d,s:d}",
        "topology_seconds",topology_seconds_,"positions_seconds",positions_seconds_,
        "output_seconds",output_seconds_);
    if(host_stats){PyObject_SetAttrString(engine_,"host_statistics",host_stats);Py_DECREF(host_stats);}
    if(PyErr_Occurred())PyErr_Clear();
    PyObject *result=PyObject_CallMethod(engine_,"close",nullptr);
    if(!result)PyErr_Clear();else Py_DECREF(result);
    Py_DECREF(engine_);
    // Registered after runtime imports, hence before their static destructors
    // in reverse atexit order. The interpreter is owned by the host process.
    static bool registered=false;
    if(!registered){std::atexit(interpreter_exit_gil);registered=true;}
  }
  if(allocated){memory->destroy(setflag);memory->destroy(cutsq);}
}
void PairMFFTorch::allocate() {
  allocated=1;const int n=atom->ntypes;
  memory->create(setflag,n+1,n+1,"mff/torch:setflag");
  memory->create(cutsq,n+1,n+1,"mff/torch:cutsq");
  for(int i=0;i<=n;i++)for(int j=0;j<=n;j++)setflag[i][j]=0;
}
void PairMFFTorch::settings(int narg,char **arg) {
  if(narg<1)error->all(FLERR,"pair_style mff/torch expects cutoff [npu] [depth N]");
  cutoff_=utils::numeric(FLERR,arg[0],false,lmp);
  if(cutoff_<=0)error->all(FLERR,"mff/torch cutoff must be positive");
  for(int i=1;i<narg;i++) {
    const std::string option(arg[i]);
    if(option=="npu")continue;
    if(option=="depth" && i+1<narg){requested_depth_=utils::inumeric(FLERR,arg[++i],false,lmp);continue;}
    error->all(FLERR,"Ascend mff/torch accepts only npu and an optional depth");
  }
}
void PairMFFTorch::coeff(int narg,char **arg) {
  if(narg!=atom->ntypes+3 || std::string(arg[0])!="*" || std::string(arg[1])!="*")
    error->all(FLERR,"pair_coeff expects * * checkpoint.pth and one element per atom type");
  if(!allocated)allocate();
  checkpoint_=arg[2];type_to_z_.assign(atom->ntypes+1,0);
  for(int i=1;i<=atom->ntypes;i++) {
    type_to_z_[i]=mfftorch::symbol_to_Z(arg[i+2]);
    if(!type_to_z_[i])error->all(FLERR,"Unknown element or NULL mapping in Ascend mff/torch");
    for(int j=i;j<=atom->ntypes;j++)setflag[i][j]=1;
  }
  load_engine();
}
void PairMFFTorch::load_engine() {
  if(engine_)error->all(FLERR,"Create a new pair style before replacing an initialized Ascend model");
  char host[MPI_MAX_PROCESSOR_NAME]={};int length=0,port=0;
  MPI_Get_processor_name(host,&length);
  // MPI_COMM_TYPE_SHARED may contain only one rank with a socket-only MPI
  // transport. Group actual processor names to assign distinct local NPUs.
  std::vector<char> hosts(static_cast<size_t>(comm->nprocs)*sizeof(host));
  MPI_Allgather(host,sizeof(host),MPI_CHAR,hosts.data(),sizeof(host),MPI_CHAR,world);
  local_device_=0;
  for(int peer=0;peer<comm->me;peer++)
    if(std::strcmp(host,hosts.data()+static_cast<size_t>(peer)*sizeof(host))==0)++local_device_;
  if(comm->me==0)port=free_port();
  MPI_Bcast(host,sizeof(host),MPI_CHAR,0,world);MPI_Bcast(&port,1,MPI_INT,0,world);
  try {
    if(!Py_IsInitialized()) {
      PyConfig config;PyConfig_InitPythonConfig(&config);
      PyStatus status=PyConfig_SetBytesString(&config,&config.program_name,MFF_PYTHON_EXECUTABLE);
      if(!PyStatus_Exception(status))status=Py_InitializeFromConfig(&config);
      PyConfig_Clear(&config);
      if(PyStatus_Exception(status))throw std::runtime_error("Cannot initialize the configured Python runtime");
      PyEval_SaveThread();
    }
    Gil gil;
    Ref module(PyImport_ImportModule("mace_ictc.interfaces.ascend_lammps"));
    // Also cover a failed engine constructor, before the Pair destructor can
    // own an engine. Runtime imports have already registered their destructors.
    static bool exit_registered=false;
    if(!exit_registered){std::atexit(interpreter_exit_gil);exit_registered=true;}
    Ref cls(PyObject_GetAttrString(module,"AscendLAMMPSEngine"));
    engine_=PyObject_CallFunction(cls,"sdiiisi",checkpoint_.c_str(),cutoff_,local_device_,comm->me,comm->nprocs,host,port);
    if(!engine_)python_error("Cannot initialize Ascend model/HCCL");
    Ref depth(PyObject_GetAttrString(engine_,"depth"));depth_=static_cast<int>(PyLong_AsLong(depth));
    if(depth_<1 || (requested_depth_ && requested_depth_!=depth_))throw std::runtime_error("Model message depth does not match pair_style depth");
  } catch(const std::exception& e){error->all(FLERR,e.what());}
}
void PairMFFTorch::init_style() {
  if(!engine_)error->all(FLERR,"mff/torch needs pair_coeff before run");
  if(domain->dimension!=3)error->all(FLERR,"Ascend mff/torch requires a three-dimensional box");
  if(std::string(update->unit_style)!="metal")error->all(FLERR,"Ascend mff/torch requires units metal");
  if(comm->nprocs==1)neighbor->add_request(this,NeighConst::REQ_FULL);
  else {
    neighbor->add_request(this,NeighConst::REQ_FULL|NeighConst::REQ_GHOST);
    comm->cutghostuser=std::max(comm->cutghostuser,depth_*cutoff_+neighbor->skin);
  }
  neighbor_generation_=-1;
}
double PairMFFTorch::init_one(int i,int j) {
  if(!setflag[i][j])error->all(FLERR,"All atom types must be mapped for Ascend mff/torch");
  return cutoff_;
}
int PairMFFTorch::pack_forward_comm(int n,int *list,double *buffer,int,int *) {
  int k=0;for(int i=0;i<n;i++){buffer[k++]=owner_rank_[list[i]];buffer[k++]=owner_index_[list[i]];}return k;
}
void PairMFFTorch::unpack_forward_comm(int n,int first,double *buffer) {
  int k=0;for(int i=first;i<first+n;i++){owner_rank_[i]=static_cast<int64_t>(buffer[k++]);owner_index_[i]=static_cast<int64_t>(buffer[k++]);}
}
void PairMFFTorch::update_topology() {
  const int owned=atom->nlocal,all=owned+atom->nghost;
  owner_rank_.assign(all,-1);owner_index_.assign(all,-1);
  for(int i=0;i<owned;i++){owner_rank_[i]=comm->me;owner_index_[i]=i;}
  comm->forward_comm(this);
  std::vector<char> center(all,0),active(all,0);
  std::vector<int> frontier;
  for(int i=0;i<owned;i++){center[i]=active[i]=1;frontier.push_back(i);}
  if(comm->nprocs>1)for(int hop=1;hop<depth_;hop++) {
    std::vector<int> next;
    for(int i:frontier)for(int k=0;k<list->numneigh[i];k++) {
      const int j=list->firstneigh[i][k]&NEIGHMASK;
      if(!center[j]){center[j]=1;next.push_back(j);}
    }
    frontier.swap(next);
  }
  source_.clear();destination_.clear();shifts_.clear();
  const int centers=list->inum+(comm->nprocs>1?list->gnum:0);
  const double xx=domain->xprd,yy=domain->yprd,zz=domain->zprd;
  for(int row=0;row<centers;row++) {
    const int i=list->ilist[row];if(!center[i])continue;
    for(int k=0;k<list->numneigh[i];k++) {
      const int j=list->firstneigh[i][k]&NEIGHMASK;
      if(comm->nprocs==1) {
        const int owner=static_cast<int>(owner_index_[j]);
        if(owner_rank_[j]!=comm->me || owner<0 || owner>=owned)error->all(FLERR,"Invalid periodic ghost owner mapping");
        const double dz=(atom->x[j][2]-atom->x[owner][2])/zz;
        const double dy=(atom->x[j][1]-atom->x[owner][1]-dz*domain->yz)/yy;
        const double dx=(atom->x[j][0]-atom->x[owner][0]-dy*domain->xy-dz*domain->xz)/xx;
        source_.push_back(owner);destination_.push_back(i);
        shifts_.push_back(-std::nearbyint(dx));shifts_.push_back(-std::nearbyint(dy));shifts_.push_back(-std::nearbyint(dz));
      } else {
        source_.push_back(j);destination_.push_back(i);shifts_.insert(shifts_.end(),3,0.f);
        active[i]=active[j]=1;
      }
    }
  }
  model_to_atom_.clear();std::vector<int> compact(all,-1);
  for(int i=0;i<all;i++)if(active[i]){compact[i]=model_to_atom_.size();model_to_atom_.push_back(i);}
  nodes_=model_to_atom_.size();
  if(comm->nprocs>1){for(auto& i:source_)i=compact[i];for(auto& i:destination_)i=compact[i];}
  numbers_.resize(nodes_);positions_.resize(static_cast<size_t>(nodes_)*3);
  for(int i=0;i<nodes_;i++)numbers_[i]=type_to_z_[atom->type[model_to_atom_[i]]];
  exchange_plan();
  neighbor_generation_=neighbor->ncalls;
  try {
    Gil gil;Ref args(PyTuple_New(12));
    PyTuple_SET_ITEM(args.value,0,PyLong_FromLong(owned));PyTuple_SET_ITEM(args.value,1,PyLong_FromLong(nodes_));
    PyTuple_SET_ITEM(args.value,2,view(numbers_));PyTuple_SET_ITEM(args.value,3,view(source_));
    PyTuple_SET_ITEM(args.value,4,view(destination_));PyTuple_SET_ITEM(args.value,5,view(shifts_));
    PyTuple_SET_ITEM(args.value,6,view(ghosts_));PyTuple_SET_ITEM(args.value,7,view(slots_));
    PyTuple_SET_ITEM(args.value,8,view(receive_indices_));PyTuple_SET_ITEM(args.value,9,view(send_counts_));
    PyTuple_SET_ITEM(args.value,10,view(receive_counts_));PyTuple_SET_ITEM(args.value,11,PyLong_FromLongLong(neighbor_generation_));
    Ref method(PyObject_GetAttrString(engine_,"topology"));Ref result(PyObject_CallObject(method,args));
  }catch(const std::exception& e){error->all(FLERR,e.what());}
}
void PairMFFTorch::exchange_plan() {
  const int size=comm->nprocs;
  std::vector<std::map<int64_t,int64_t>> peers(size);
  for(int m=atom->nlocal;m<nodes_;m++) {
    const int i=model_to_atom_[m],rank=owner_rank_[i];
    if(rank<0 || rank>=size || owner_index_[i]<0)error->all(FLERR,"Incomplete ghost owner metadata");
    peers[rank].emplace(owner_index_[i],0);
  }
  std::vector<int> sc(size),rc(size),sd(size),rd(size);
  std::vector<int64_t> owners;
  for(int peer=0;peer<size;peer++) {
    sd[peer]=owners.size();sc[peer]=peers[peer].size();
    for(auto& item:peers[peer]){item.second=owners.size();owners.push_back(item.first);}
  }
  MPI_Alltoall(sc.data(),1,MPI_INT,rc.data(),1,MPI_INT,world);
  for(int p=1;p<size;p++)rd[p]=rd[p-1]+rc[p-1];
  receive_indices_.resize(rd.back()+rc.back());
  MPI_Alltoallv(owners.data(),sc.data(),sd.data(),MPI_INT64_T,
                receive_indices_.data(),rc.data(),rd.data(),MPI_INT64_T,world);
  ghosts_.clear();slots_.clear();
  for(int m=atom->nlocal;m<nodes_;m++) {
    const int i=model_to_atom_[m];ghosts_.push_back(m);slots_.push_back(peers[owner_rank_[i]].at(owner_index_[i]));
  }
  send_counts_.assign(sc.begin(),sc.end());receive_counts_.assign(rc.begin(),rc.end());
}
void PairMFFTorch::update_positions() {
  for(int m=0;m<nodes_;m++)for(int axis=0;axis<3;axis++)positions_[static_cast<size_t>(m)*3+axis]=atom->x[model_to_atom_[m]][axis];
  const float values[]={static_cast<float>(domain->xprd),0,0,static_cast<float>(domain->xy),static_cast<float>(domain->yprd),0,
                        static_cast<float>(domain->xz),static_cast<float>(domain->yz),static_cast<float>(domain->zprd)};
  std::copy(values,values+9,cell_);
}
void PairMFFTorch::compute(int eflag,int vflag) {
  ev_init(eflag,vflag);
  if(vflag_atom || cvflag_atom)error->all(FLERR,"Ascend mff/torch currently provides global virial, not per-atom stress");
  double started=MPI_Wtime();
  if(neighbor_generation_!=neighbor->ncalls)update_topology();
  double updated=MPI_Wtime();topology_seconds_+=updated-started;
  update_positions();
  positions_seconds_+=MPI_Wtime()-updated;
  try {
    Gil gil;Ref args(PyTuple_New(3));
    PyTuple_SET_ITEM(args.value,0,view(positions_));PyTuple_SET_ITEM(args.value,1,cell_view(cell_));
    PyTuple_SET_ITEM(args.value,2,PyBool_FromLong(vflag_global));
    Ref method(PyObject_GetAttrString(engine_,"compute"));Ref result(PyObject_CallObject(method,args));
    started=MPI_Wtime();
    if(!PyTuple_Check(result) || PyTuple_Size(result)!=4)throw std::runtime_error("Invalid Ascend engine result");
    if(eflag_global)eng_vdwl+=PyFloat_AsDouble(PyTuple_GetItem(result,0));
    force_result_.resize(static_cast<size_t>(atom->nlocal)*3);energy_result_.resize(atom->nlocal);float vir[6];
    buffer_copy(PyTuple_GetItem(result,1),force_result_.data(),force_result_.size());
    buffer_copy(PyTuple_GetItem(result,2),energy_result_.data(),energy_result_.size());
    buffer_copy(PyTuple_GetItem(result,3),vir,6);
    for(int i=0;i<atom->nlocal;i++) {
      for(int axis=0;axis<3;axis++)atom->f[i][axis]+=force_result_[static_cast<size_t>(i)*3+axis];
      if(eflag_atom)eatom[i]+=energy_result_[i];
    }
    if(vflag_global)for(int axis=0;axis<6;axis++)virial[axis]+=vir[axis];
    output_seconds_+=MPI_Wtime()-started;
  } catch(const std::exception& e){error->all(FLERR,e.what());}
}
double PairMFFTorch::memory_usage() {
  return sizeof(float)*(positions_.capacity()+shifts_.capacity()+force_result_.capacity()+energy_result_.capacity())+
         sizeof(int)*(model_to_atom_.capacity())+
         sizeof(int64_t)*(source_.capacity()+destination_.capacity()+numbers_.capacity()+
         ghosts_.capacity()+slots_.capacity()+receive_indices_.capacity()+owner_rank_.capacity()+owner_index_.capacity()+
         type_to_z_.capacity()+send_counts_.capacity()+receive_counts_.capacity());
}
