#include "kernel_launch.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
namespace {
void assignment(const at::Tensor& f,const at::Tensor& batch,const at::Tensor& s,const at::Tensor& mesh,
    const at::Tensor& v,at::Tensor ds,at::Tensor dm,at::Tensor df,at::IntArrayRef shape) {
    TORCH_CHECK(shape.size()==8,"Invalid electrostatic assignment descriptor");
    const int64_t mx=shape[0],my=shape[1],mz=shape[2],graphs=shape[3],order=shape[4],mode=shape[5],slab=shape[6],flags=shape[7];
    TORCH_CHECK(mx>0 && my>0 && mz>0 && graphs>0 && order>=2 && order<=4 && mode>=0 && mode<=3 &&
        (slab==0 || slab==1) && flags>=0 && flags<=7,"Invalid electrostatic assignment dimensions");
    TORCH_CHECK(f.device().type()==c10::DeviceType::PrivateUse1 && f.scalar_type()==at::kFloat && f.dim()==2 && f.size(1)==3,
        "Electrostatic assignment requires NPU FP32 coordinates [N,3]");
    TORCH_CHECK(s.dim()==2 && s.size(0)==f.size(0) && s.size(1)>0,"Expected source [N,C]");
    const int64_t n=f.size(0),c=s.size(1),points=mx*my*mz,total=graphs*points*c;
    for(const auto& value:{f,s,mesh,v,ds,dm,df})TORCH_CHECK(value.device()==f.device() &&
        value.scalar_type()==at::kFloat && value.is_contiguous(),"Expected contiguous FP32 operands on one NPU");
    TORCH_CHECK(batch.device()==f.device() && batch.scalar_type()==at::kLong && batch.is_contiguous() && batch.numel()==n,
        "Expected contiguous int64 batch[N]");
    TORCH_CHECK(mode==0 || mesh.numel()==total,"Invalid input mesh size");
    TORCH_CHECK(mode!=3 || v.sizes()==f.sizes(),"Expected coordinate direction [N,3]");
    TORCH_CHECK((mode!=1 && !(mode==3&&(flags&2))) || ds.sizes()==s.sizes(),"Invalid source gradient size");
    TORCH_CHECK((mode!=0 && !(mode==3&&(flags&4))) || dm.numel()==total,"Invalid mesh output size");
    TORCH_CHECK((mode!=2 && !(mode==3&&(flags&1))) || df.sizes()==f.sizes(),"Invalid coordinate gradient size");
    if(!n)return;
    c10_npu::NPUGuard guard(f.device());auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform,"CANN platform discovery failed");uint64_t ub=0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    TORCH_CHECK(ub>8192,"Insufficient UB for electrostatic assignment");
    const uint32_t limit=std::min<uint64_t>(4096,((ub-4096-128)/(12*4))/64*64);
    const uint32_t tile=std::min<uint64_t>((c+7)/8*8,limit);
    TORCH_CHECK(tile>0 && (12*static_cast<uint64_t>(tile)+32)*4+4096<=ub,"Invalid electrostatic UB tiling");
    const uint32_t blocks=std::min<uint64_t>(platform->GetCoreNumAiv(),n*((c+tile-1)/tile));
    TORCH_CHECK(blocks>0,"No Ascend Vector cores available");
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto launch=[f,batch,s,mesh,v,ds,dm,df,stream,n,c,mx,my,mz,order,mode,slab,flags,tile,blocks]() -> int {
        ictc_ascendc::launch_electrostatic_assignment(blocks,stream,
            reinterpret_cast<uint8_t*>(f.data_ptr()),reinterpret_cast<uint8_t*>(batch.data_ptr()),
            reinterpret_cast<uint8_t*>(s.data_ptr()),reinterpret_cast<uint8_t*>(mesh.data_ptr()),reinterpret_cast<uint8_t*>(v.data_ptr()),
            reinterpret_cast<uint8_t*>(ds.data_ptr()),reinterpret_cast<uint8_t*>(dm.data_ptr()),reinterpret_cast<uint8_t*>(df.data_ptr()),
            n,c,mx,my,mz,order,mode,slab,flags,tile);return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ictc_ascendc_electrostatic_assignment",launch);
}
}
TORCH_LIBRARY_FRAGMENT(ictc_ascendc,m) {
    m.def("electrostatic_assignment(Tensor frac, Tensor batch, Tensor source, Tensor mesh, Tensor direction, Tensor(a!) source_out, Tensor(b!) mesh_out, Tensor(c!) coordinate_out, int[] shape) -> ()");
}
TORCH_LIBRARY_IMPL(ictc_ascendc,PrivateUse1,m) {m.impl("electrostatic_assignment",TORCH_FN(assignment));}
