#include "kernel_operator.h"
using namespace AscendC;
namespace {
__aicore__ inline uint32_t A8(uint32_t x) { return (x+7)/8*8; }
class Adjoint {
public:
    __aicore__ inline void Init(GM_ADDR l,GM_ADDR r,GM_ADDR g,GM_ADDR hl,GM_ADDR hr,
        GM_ADDR dst,GM_ADDR mask,GM_ADDR meta,GM_ADDR indices,GM_ADDR dl,GM_ADDR dr,GM_ADDR dg,
        uint64_t edges,uint64_t ls,uint64_t rs,uint64_t hls,uint64_t hrs,uint32_t c,
        uint32_t q,uint32_t p,uint32_t m,uint32_t second,uint32_t rows,uint32_t bc,
        uint32_t masked,uint32_t flags) {
        n_=edges;ls_=ls;rs_=rs;hls_=hls;hrs_=hrs;c_=c;q_=q;p_=p;m_=m;
        second_=second;rows_=rows;bc_=bc;masked_=masked;flags_=flags;cb_=(c+bc-1)/bc;
        vec_=rows*q*bc;rc_=rows*bc;pv_=rows*p*bc;
        dl_=flags&1;dr_=flags&2;dg_=flags&4;hl_=flags&8;hr_=flags&16;
        useL_=(!second_ && dr_) || (second_ && dg_ && hr_);
        useR_=(!second_ && dl_) || (second_ && dg_ && hl_);
        useG_=dl_ || dr_;useHL_=second_ && hl_ && (dr_ || dg_);
        useHR_=second_ && hr_ && (dl_ || dg_);
        lo_=0;ro_=lo_+(useL_?vec_:0);go_=ro_+(useR_?pv_:0);
        hlo_=go_+(useG_?vec_:0);hro_=hlo_+(useHL_?vec_:0);
        dstoff_=hro_+(useHR_?pv_:0);maskoff_=dstoff_+A8(2*rows);
        inputSize_=maskoff_+(masked?A8(rows):0);
        dlo_=0;dro_=dlo_+(dl_?vec_:0);dgo_=dro_+(dr_?pv_:0);
        outputSize_=dgo_+(dg_?vec_:0);
        lG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(l));rG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(r));
        gG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));hlG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hl));
        hrG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hr));maskG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mask));
        dstG_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(dst));metaG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(meta));
        ixG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(indices));
        dlG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dl));drG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dr));
        dgG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dg));
        pipe_.InitBuffer(input_,2,inputSize_*4);pipe_.InitBuffer(output_,2,outputSize_*4);
        pipe_.InitBuffer(work_,(vec_+rc_)*4);pipe_.InitBuffer(index_,(2*vec_+2*rc_)*4);
        pipe_.InitBuffer(table_,A8(4*p_)*4);
        Copy(table_.Get<uint32_t>(),metaG_,0,4*p_);Copy(index_.Get<uint32_t>(),ixG_,0,2*vec_+2*rc_);
        Sync<HardEvent::MTE2_S>();Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Process() {
        const uint64_t tasks=((n_+rows_-1)/rows_)*cb_;uint64_t task=GetBlockIdx();
        if(task>=tasks)return;CopyIn(task);
        for(;task<tasks;task+=GetBlockNum()) {
            if(task+GetBlockNum()<tasks)CopyIn(task+GetBlockNum());
            Compute(task);
        }
    }
private:
    template<HardEvent event> __aicore__ inline void Sync() {
        const event_t id=static_cast<event_t>(pipe_.FetchEventID(event));SetFlag<event>(id);WaitFlag<event>(id);
    }
    template<class T> __aicore__ inline void Copy(LocalTensor<T> d,GlobalTensor<T> s,uint64_t begin,uint32_t count) {
        if(!count)return;DataCopyExtParams cp{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{false,0,0,0};DataCopyPad(d,s[begin],cp,pad);
    }
    __aicore__ inline void Columns(LocalTensor<float> d,GlobalTensor<float> s,uint64_t begin,uint32_t angular) {
        DataCopyExtParams cp{static_cast<uint16_t>(angular),cn_*4,(c_-cn_)*4,(bc_-A8(cn_))/8,0};
        DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(d,s[begin],cp,pad);
    }
    __aicore__ inline void Read(LocalTensor<float> d,GlobalTensor<float> s,uint64_t stride,uint32_t width) {
        if(stride==static_cast<uint64_t>(width)*c_)Columns(d,s,firstN_*stride+firstC_,rn_*width);
        else if(cn_==bc_ && bc_==c_) {
            DataCopyExtParams cp{static_cast<uint16_t>(rn_),width*c_*4,static_cast<uint32_t>((stride-width*c_)*4),0,0};
            DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(d,s[firstN_*stride],cp,pad);
        } else for(uint32_t row=0;row<rn_;row++)Columns(d[row*width*bc_],s,(firstN_+row)*stride+firstC_,width);
    }
    __aicore__ inline void Describe(uint64_t task) {
        firstN_=(task/cb_)*rows_;firstC_=(task%cb_)*bc_;
        rn_=n_-firstN_<rows_?n_-firstN_:rows_;cn_=c_-firstC_<bc_?c_-firstC_:bc_;
    }
    __aicore__ inline void CopyIn(uint64_t task) {
        Describe(task);auto x=input_.AllocTensor<float>();auto destinations=x[dstoff_].ReinterpretCast<int64_t>();
        Copy(destinations,dstG_,firstN_,rn_);Sync<HardEvent::MTE2_S>();
        if(useL_)Read(x[lo_],lG_,ls_,q_);if(useR_)Read(x[ro_],rG_,rs_,p_);
        if(useHL_)Read(x[hlo_],hlG_,hls_,q_);if(useHR_)Read(x[hro_],hrG_,hrs_,p_);
        auto table=table_.Get<uint32_t>();
        if(useG_)for(uint32_t row=0;row<rn_;row++)for(uint32_t path=0;path<p_;) {
            const uint32_t src=table.GetValue(path*4),out=table.GetValue(path*4+2);
            uint32_t width=table.GetValue(path*4+3);path++;
            while(path<p_ && table.GetValue(path*4)==src+width && table.GetValue(path*4+2)==out+width) {
                width+=table.GetValue(path*4+3);path++;
            }
            Columns(x[go_+(row*q_+src)*bc_],gG_,(static_cast<uint64_t>(destinations.GetValue(row))*m_+out)*c_+firstC_,width);
        }
        if(masked_)Copy(x[maskoff_],maskG_,firstN_,rn_);input_.EnQue(x);
    }
    __aicore__ inline void EdgeOut(GlobalTensor<float> target,LocalTensor<float> value,uint32_t width) {
        DataCopyExtParams cp{static_cast<uint16_t>(rn_*width),cn_*4,(bc_-A8(cn_))/8,(c_-cn_)*4,0};
        DataCopyPad(target[firstN_*width*c_+firstC_],value,cp);
    }
    __aicore__ inline void Compute(uint64_t task) {
        Describe(task);const uint32_t lanes=rn_*q_*bc_,rc=rn_*bc_;
        auto x=input_.DeQue<float>();auto y=output_.AllocTensor<float>();
        auto ix=index_.Get<uint32_t>();auto table=table_.Get<uint32_t>();
        auto work=work_.Get<float>();auto tmp=work[vec_];
        if(dl_) {Gather(y[dlo_],x[second_?hro_:ro_],ix,0,lanes);Mul(y[dlo_],y[dlo_],x[go_],lanes);}
        if(dg_) {
            if(hl_) {Gather(y[dgo_],x[ro_],ix,0,lanes);Mul(y[dgo_],y[dgo_],x[hlo_],lanes);}
            if(hr_) {
                auto target=hl_?work:y[dgo_];Gather(target,x[hro_],ix,0,lanes);Mul(target,target,x[lo_],lanes);
                if(hl_)Add(y[dgo_],y[dgo_],target,lanes);
            }
        }
        if(masked_ && (dl_ || dg_)) {
            Gather(work,x[maskoff_],ix[vec_],0,lanes);
            if(dl_)Mul(y[dlo_],y[dlo_],work,lanes);if(dg_)Mul(y[dgo_],y[dgo_],work,lanes);
        }
        if(dr_) {
            Mul(work,x[second_?hlo_:lo_],x[go_],lanes);Duplicate(y[dro_],0.f,pv_);
            for(uint32_t path=0;path<p_;path++) {
                const uint32_t src=table.GetValue(path*4),p=table.GetValue(path*4+1),width=table.GetValue(path*4+3);
                for(uint32_t a=0;a<width;a++) {Gather(tmp,work,ix[2*vec_],(src+a)*bc_*4,rc);Add(y[dro_+p*rc_],y[dro_+p*rc_],tmp,rc);}
                if(masked_) {Gather(tmp,x[maskoff_],ix[2*vec_+rc_],0,rc);Mul(y[dro_+p*rc_],y[dro_+p*rc_],tmp,rc);}
            }
        }
        output_.EnQue(y);auto ready=output_.DeQue<float>();
        if(dl_)EdgeOut(dlG_,ready[dlo_],q_);
        if(dr_)for(uint32_t path=0;path<p_;path++) {
            const uint32_t p=table.GetValue(path*4+1);
            DataCopyExtParams cp{static_cast<uint16_t>(rn_),cn_*4,(bc_-A8(cn_))/8,(p_*c_-cn_)*4,0};
            DataCopyPad(drG_[(firstN_*p_+p)*c_+firstC_],ready[dro_+p*rc_],cp);
        }
        if(dg_) {
            auto destinations=x[dstoff_].ReinterpretCast<int64_t>();
            for(uint32_t row=0;row<rn_;row++) {
                const uint64_t node=destinations.GetValue(row);SetAtomicAdd<float>();
                for(uint32_t path=0;path<p_;) {
                    const uint32_t src=table.GetValue(path*4),out=table.GetValue(path*4+2);
                    uint32_t width=table.GetValue(path*4+3);path++;
                    while(path<p_ && table.GetValue(path*4)==src+width && table.GetValue(path*4+2)==out+width) {
                        width+=table.GetValue(path*4+3);path++;
                    }
                    DataCopyExtParams cp{static_cast<uint16_t>(width),cn_*4,(bc_-A8(cn_))/8,(c_-cn_)*4,0};
                    DataCopyPad(dgG_[(node*m_+out)*c_+firstC_],ready[dgo_+(row*q_+src)*bc_],cp);
                }
                SetAtomicNone();
            }
        }
        output_.FreeTensor(ready);input_.FreeTensor(x);
    }
    TPipe pipe_;TQue<TPosition::VECIN,2> input_;TQue<TPosition::VECOUT,2> output_;
    TBuf<TPosition::VECCALC> work_,index_,table_;
    GlobalTensor<float> lG_,rG_,gG_,hlG_,hrG_,maskG_,dlG_,drG_,dgG_;
    GlobalTensor<int64_t> dstG_;GlobalTensor<uint32_t> metaG_,ixG_;
    uint64_t n_,ls_,rs_,hls_,hrs_,firstN_;
    uint32_t c_,q_,p_,m_,second_,rows_,bc_,masked_,flags_,cb_,vec_,rc_,pv_;
    uint32_t lo_,ro_,go_,hlo_,hro_,dstoff_,maskoff_,dlo_,dro_,dgo_,inputSize_,outputSize_,rn_,cn_,firstC_;
    bool dl_,dr_,dg_,hl_,hr_,useL_,useR_,useG_,useHL_,useHR_;
};
}
extern "C" __global__ __aicore__ void ictc_epilogue_adjoint(
    GM_ADDR l,GM_ADDR r,GM_ADDR g,GM_ADDR hl,GM_ADDR hr,GM_ADDR dst,GM_ADDR mask,GM_ADDR meta,GM_ADDR ix,
    GM_ADDR dl,GM_ADDR dr,GM_ADDR dg,uint64_t n,uint64_t ls,uint64_t rs,uint64_t hls,uint64_t hrs,
    uint32_t c,uint32_t q,uint32_t p,uint32_t m,uint32_t second,uint32_t rows,uint32_t bc,uint32_t masked,uint32_t flags) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Adjoint kernel;kernel.Init(l,r,g,hl,hr,dst,mask,meta,ix,dl,dr,dg,n,ls,rs,hls,hrs,c,q,p,m,second,rows,bc,masked,flags);kernel.Process();
}
