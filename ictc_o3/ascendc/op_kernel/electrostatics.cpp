#include "kernel_operator.h"
#include "kernel_launch.h"
using namespace AscendC;
namespace {
__aicore__ inline int64_t Floor(float x) {int64_t n=static_cast<int64_t>(x);return n-(static_cast<float>(n)>x);}
__aicore__ inline int64_t Wrap(int64_t x,int64_t m) {x%=m;return x<0?x+m:x;}
__aicore__ inline void Spline(float x,uint32_t order,int64_t& base,float* w,float* d,float* h) {
    if(order==2) {
        base=Floor(x);float t=x-static_cast<float>(base);
        w[0]=1-t;w[1]=t;d[0]=-1;d[1]=1;h[0]=h[1]=0;
    } else if(order==3) {
        base=Floor(x-.5f);float t=x-static_cast<float>(base);
        w[0]=.5f*(1.5f-t)*(1.5f-t);w[1]=.75f-(t-1)*(t-1);w[2]=.5f*(t-.5f)*(t-.5f);
        d[0]=t-1.5f;d[1]=2*(1-t);d[2]=t-.5f;h[0]=1;h[1]=-2;h[2]=1;
    } else {
        int64_t f=Floor(x);base=f-1;float t=x-static_cast<float>(f),u=1-t,t2=t*t,t3=t2*t;
        w[0]=u*u*u/6;w[1]=(3*t3-6*t2+4)/6;w[2]=(-3*t3+3*t2+3*t+1)/6;w[3]=t3/6;
        d[0]=-u*u/2;d[1]=1.5f*t2-2*t;d[2]=-1.5f*t2+t+.5f;d[3]=t2/2;
        h[0]=u;h[1]=3*t-2;h[2]=-3*t+1;h[3]=t;
    }
}
class Assignment {
public:
    __aicore__ inline void Init(GM_ADDR frac,GM_ADDR batch,GM_ADDR source,GM_ADDR mesh,GM_ADDR direction,
        GM_ADDR ds,GM_ADDR dm,GM_ADDR df,uint64_t n,uint32_t c,uint32_t mx,uint32_t my,uint32_t mz,
        uint32_t order,uint32_t mode,uint32_t slab,uint32_t flags,uint32_t tile) {
        n_=n;c_=c;sizes_[0]=mx;sizes_[1]=my;sizes_[2]=mz;order_=order;mode_=mode;slab_=slab;flags_=flags;tile_=tile;
        points_=static_cast<uint64_t>(mx)*my*mz;ct_=(c+tile-1)/tile;
        f_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(frac));b_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(batch));
        s_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source));m_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mesh));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(direction));ds_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ds));
        dm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dm));df_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(df));
        pipe_.InitBuffer(buffer_,static_cast<uint64_t>(12*tile+32)*4);
    }
    __aicore__ inline void Process() {
        auto all=buffer_.Get<float>();
        auto src=all,field=all[tile_],value=all[2*tile_],temporary=all[3*tile_];
        auto sum=all[4*tile_],gx=all[5*tile_],gy=all[6*tile_],gz=all[7*tile_];
        auto product=all[8*tile_],workspace=all[9*tile_],scalar=all[11*tile_];
        for(uint64_t task=GetBlockIdx();task<n_*ct_;task+=GetBlockNum()) {
            uint64_t atom=task/ct_;uint32_t first=(task%ct_)*tile_,count=c_-first<tile_?c_-first:tile_;
            int64_t graph=b_.GetValue(atom);float w[3][4],d[3][4],h[3][4],direction[3]={0,0,0};int64_t ix[3][4];
            for(uint32_t a=0;a<3;a++) {
                int64_t base;Spline(f_.GetValue(3*atom+a)*sizes_[a],order_,base,w[a],d[a],h[a]);
                for(uint32_t u=0;u<order_;u++) {
                    int64_t v=base+u,m=sizes_[a];ix[a][u]=(slab_&&a==2)?(v<0?0:(v>=m?m-1:v)):Wrap(v,m);
                    d[a][u]*=m;h[a][u]*=m*m;
                }
                if(mode_==3)direction[a]=v_.GetValue(3*atom+a);
            }
            if(mode_!=1)Read(src,s_,atom*c_+first,count);
            Duplicate(sum,0.f,tile_);Duplicate(gx,0.f,tile_);Duplicate(gy,0.f,tile_);Duplicate(gz,0.f,tile_);
            PipeBarrier<PIPE_V>();
            for(uint32_t i=0;i<order_;i++)for(uint32_t j=0;j<order_;j++)for(uint32_t k=0;k<order_;k++) {
                uint32_t u[3]={i,j,k};
                uint64_t spatial=(ix[0][i]*sizes_[1]+ix[1][j])*sizes_[2]+ix[2][k];
                uint64_t offset=(static_cast<uint64_t>(graph)*points_+spatial)*c_+first;
                float weight=w[0][i]*w[1][j]*w[2][k];
                if(mode_==0) {
                    Muls(value,src,weight,count);Write(dm_,offset,value,count,true);
                } else {
                    Read(field,m_,offset,count);
                    if(mode_==1) {
                        Muls(value,field,weight,count);PipeBarrier<PIPE_V>();Add(sum,sum,value,count);PipeBarrier<PIPE_V>();
                    } else {
                        float dw[3]={d[0][i]*w[1][j]*w[2][k],w[0][i]*d[1][j]*w[2][k],w[0][i]*w[1][j]*d[2][k]};
                        if(mode_==3) {
                            float directional=direction[0]*dw[0]+direction[1]*dw[1]+direction[2]*dw[2];
                            if(flags_&2) {Muls(value,field,directional,count);PipeBarrier<PIPE_V>();Add(sum,sum,value,count);PipeBarrier<PIPE_V>();}
                            if(flags_&4) {Muls(value,src,directional,count);Write(dm_,offset,value,count,true);}
                        }
                        if(mode_==2 || (flags_&1)) {
                            Mul(product,src,field,count);PipeBarrier<PIPE_V>();
                            for(uint32_t a=0;a<3;a++) {
                                float coefficient=dw[a];
                                if(mode_==3) {
                                    coefficient=0;
                                    for(uint32_t e=0;e<3;e++) {
                                        float dd=1;
                                        for(uint32_t v=0;v<3;v++)dd*=(v==a&&v==e)?h[v][u[v]]:((v==a||v==e)?d[v][u[v]]:w[v][u[v]]);
                                        coefficient+=direction[e]*dd;
                                    }
                                }
                                auto gradient=a==0?gx:(a==1?gy:gz);
                                Muls(temporary,product,coefficient,count);PipeBarrier<PIPE_V>();
                                Add(gradient,gradient,temporary,count);PipeBarrier<PIPE_V>();
                            }
                        }
                    }
                }
            }
            if(mode_==1 || (mode_==3&&(flags_&2)))Write(ds_,atom*c_+first,sum,count,false);
            if(mode_==2 || (mode_==3&&(flags_&1))) {
                ReduceSum(scalar,gx,workspace,count);PipeBarrier<PIPE_V>();
                ReduceSum(scalar[8],gy,workspace,count);PipeBarrier<PIPE_V>();
                ReduceSum(scalar[16],gz,workspace,count);Sync<HardEvent::V_S>();
                float x=scalar.GetValue(0),y=scalar.GetValue(8),z=scalar.GetValue(16);
                scalar.SetValue(0,x);scalar.SetValue(1,y);scalar.SetValue(2,z);Sync<HardEvent::S_MTE3>();
                Write(df_,3*atom,scalar,3,true);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
private:
    template<HardEvent event> __aicore__ inline void Sync() {
        event_t id=static_cast<event_t>(pipe_.FetchEventID(event));SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void Read(LocalTensor<float> value,GlobalTensor<float> source,uint64_t offset,uint32_t count) {
        Sync<HardEvent::V_MTE2>();DataCopyExtParams cp{1,count*4,0,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(value,source[offset],cp,pad);Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Write(GlobalTensor<float> target,uint64_t offset,LocalTensor<float> value,uint32_t count,bool atomic) {
        Sync<HardEvent::V_MTE3>();if(atomic)SetAtomicAdd<float>();DataCopyExtParams cp{1,count*4,0,0,0};
        DataCopyPad(target[offset],value,cp);if(atomic)SetAtomicNone();Sync<HardEvent::MTE3_V>();
    }
    TPipe pipe_;TBuf<TPosition::VECCALC> buffer_;
    GlobalTensor<float> f_,s_,m_,v_,ds_,dm_,df_;GlobalTensor<int64_t>b_;
    uint64_t n_,points_;uint32_t c_,sizes_[3],order_,mode_,slab_,flags_,tile_,ct_;
};
}
extern "C" __global__ __aicore__ void ictc_electrostatic_assignment(GM_ADDR f,GM_ADDR b,GM_ADDR s,GM_ADDR m,GM_ADDR v,
    GM_ADDR ds,GM_ADDR dm,GM_ADDR df,uint64_t n,uint32_t c,uint32_t mx,uint32_t my,uint32_t mz,
    uint32_t order,uint32_t mode,uint32_t slab,uint32_t flags,uint32_t tile) {
    Assignment kernel;kernel.Init(f,b,s,m,v,ds,dm,df,n,c,mx,my,mz,order,mode,slab,flags,tile);kernel.Process();
}
namespace ictc_ascendc {
void launch_electrostatic_assignment(uint32_t blocks,aclrtStream stream,uint8_t* f,uint8_t* b,uint8_t* s,uint8_t* m,uint8_t* v,
    uint8_t* ds,uint8_t* dm,uint8_t* df,uint64_t n,uint32_t c,uint32_t mx,uint32_t my,uint32_t mz,
    uint32_t order,uint32_t mode,uint32_t slab,uint32_t flags,uint32_t tile) {
    ictc_electrostatic_assignment<<<blocks,nullptr,stream>>>(f,b,s,m,v,ds,dm,df,n,c,mx,my,mz,order,mode,slab,flags,tile);
}
}
