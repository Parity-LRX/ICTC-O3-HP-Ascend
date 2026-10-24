#include "kernel_operator.h"
using namespace AscendC;
namespace {
__aicore__ inline uint32_t A8(uint32_t x) { return (x+7)/8*8; }
class Epilogue {
public:
    __aicore__ inline void Init(GM_ADDR left,GM_ADDR right,GM_ADDR grad,GM_ADDR dst,
        GM_ADDR mask,GM_ADDR meta,GM_ADDR indices,GM_ADDR out,uint64_t edges,
        uint64_t ls,uint64_t rs,uint32_t c,uint32_t q,uint32_t p,uint32_t m,
        uint32_t paths,uint32_t mode,uint32_t rows,uint32_t bc,uint32_t masked,uint32_t dense) {
        n_=edges;ls_=ls;rs_=rs;c_=c;q_=q;p_=p;m_=m;paths_=paths;mode_=mode;
        rows_=rows;bc_=bc;masked_=masked;dense_=dense;cb_=(c+bc-1)/bc;
        vec_=rows*q*bc;rc_=rows*bc;
        lsize_=mode==1 ? 0 : vec_;rsize_=mode==2 ? 0 : rows*p*bc;
        gsize_=mode==0 ? 0 : rows*(dense ? m : q)*bc;
        inputSize_=lsize_+rsize_+gsize_+A8(2*rows)+(masked ? A8(rows) : 0);
        outputSize_=mode==2 ? rows*p*bc : vec_;
        l_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(left));
        r_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(right));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(grad));
        dst_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(dst));
        mask_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mask));
        meta_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(meta));
        ix_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(indices));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        pipe_.InitBuffer(input_,2,inputSize_*4);pipe_.InitBuffer(output_,2,outputSize_*4);
        pipe_.InitBuffer(work_,(vec_+rc_)*4);pipe_.InitBuffer(index_,((2+dense)*vec_+2*rc_)*4);
        pipe_.InitBuffer(table_,A8(4*paths_)*4);
        Copy(table_.Get<uint32_t>(),meta_,0,4*paths_);
        Copy(index_.Get<uint32_t>(),ix_,0,(2+dense)*vec_+2*rc_);
        Sync<HardEvent::MTE2_S>();Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Process() {
        const uint64_t tasks=((n_+rows_-1)/rows_)*cb_;
        uint64_t task=GetBlockIdx();if(task>=tasks)return;
        CopyIn(task);
        for(;task<tasks;task+=GetBlockNum()) {
            if(task+GetBlockNum()<tasks)CopyIn(task+GetBlockNum());
            Compute(task);
        }
    }
private:
    template<HardEvent event> __aicore__ inline void Sync() {
        const event_t id=static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    template<class T> __aicore__ inline void Copy(LocalTensor<T> d,GlobalTensor<T> s,uint64_t begin,uint32_t count) {
        if(!count)return;DataCopyExtParams cp{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{false,0,0,0};DataCopyPad(d,s[begin],cp,pad);
    }
    __aicore__ inline void ReadColumns(LocalTensor<float> d,GlobalTensor<float> s,
        uint64_t begin,uint32_t angular,uint32_t channels) {
        DataCopyExtParams cp{static_cast<uint16_t>(angular),channels*4,(c_-channels)*4,(bc_-A8(channels))/8,0};
        DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(d,s[begin],cp,pad);
    }
    __aicore__ inline void Describe(uint64_t task) {
        firstN_=(task/cb_)*rows_;firstC_=(task%cb_)*bc_;
        rn_=n_-firstN_<rows_ ? n_-firstN_ : rows_;
        cn_=c_-firstC_<bc_ ? c_-firstC_ : bc_;
    }
    __aicore__ inline void CopyIn(uint64_t task) {
        Describe(task);auto x=input_.AllocTensor<float>();
        auto destinations=x[lsize_+rsize_+gsize_].ReinterpretCast<int64_t>();
        Copy(destinations,dst_,firstN_,rn_);Sync<HardEvent::MTE2_S>();
        auto table=table_.Get<uint32_t>();
        const bool linearL=lsize_ && ls_==static_cast<uint64_t>(q_)*c_;
        const bool linearR=rsize_ && cn_==bc_ && bc_==c_;
        if(linearL)ReadColumns(x,l_,firstN_*ls_+firstC_,rn_*q_,cn_);
        if(linearR) {
            DataCopyExtParams cp{static_cast<uint16_t>(rn_),p_*c_*4,static_cast<uint32_t>((rs_-p_*c_)*4),0,0};
            DataCopyPadExtParams<float> pad{false,0,0,0};
            DataCopyPad(x[lsize_],r_[firstN_*rs_],cp,pad);
        }
        for(uint32_t row=0;row<rn_;row++) {
            if(lsize_ && !linearL)ReadColumns(x[row*q_*bc_],l_,(firstN_+row)*ls_+firstC_,q_,cn_);
            if(rsize_ && !linearR)ReadColumns(x[lsize_+row*p_*bc_],r_,(firstN_+row)*rs_+firstC_,p_,cn_);
            if(gsize_ && dense_)ReadColumns(x[lsize_+rsize_+row*m_*bc_],g_,
                static_cast<uint64_t>(destinations.GetValue(row))*m_*c_+firstC_,m_,cn_);
            if(gsize_ && !dense_)for(uint32_t path=0;path<paths_;) {
                const uint32_t src=table.GetValue(path*4),out=table.GetValue(path*4+2);
                uint32_t width=table.GetValue(path*4+3);path++;
                while(path<paths_ && table.GetValue(path*4)==src+width && table.GetValue(path*4+2)==out+width) {
                    width+=table.GetValue(path*4+3);path++;
                }
                ReadColumns(x[lsize_+rsize_+(row*q_+src)*bc_],g_,
                    (static_cast<uint64_t>(destinations.GetValue(row))*m_+out)*c_+firstC_,width,cn_);
            }
        }
        if(masked_)Copy(x[lsize_+rsize_+gsize_+A8(2*rows_)],mask_,firstN_,rn_);
        input_.EnQue(x);
    }
    __aicore__ inline void Compute(uint64_t task) {
        Describe(task);const uint32_t lanes=rn_*q_*bc_,rc=rn_*bc_;
        auto x=input_.DeQue<float>();auto y=output_.AllocTensor<float>();
        auto ix=index_.Get<uint32_t>();auto table=table_.Get<uint32_t>();
        auto work=work_.Get<float>();auto tmp=work[vec_];
        auto left=x;auto right=x[lsize_];auto grad=x[lsize_+rsize_];
        auto destinations=x[lsize_+rsize_+gsize_].ReinterpretCast<int64_t>();
        auto mask=x[lsize_+rsize_+gsize_+A8(2*rows_)];
        if(mode_==2) {
            if(dense_) {Gather(work,grad,ix[2*vec_+2*rc_],0,lanes);Mul(work,work,left,lanes);}
            else Mul(work,left,grad,lanes);
            Duplicate(y,0.f,outputSize_);
            for(uint32_t path=0;path<paths_;path++) {
                const uint32_t src=table.GetValue(path*4),p=table.GetValue(path*4+1),width=table.GetValue(path*4+3);
                for(uint32_t q=0;q<width;q++) {
                    Gather(tmp,work,ix[2*vec_],(src+q)*bc_*4,rc);
                    Add(y[p*rc_],y[p*rc_],tmp,rc);
                }
                if(masked_) {
                    Gather(tmp,mask,ix[2*vec_+rc_],0,rc);
                    Mul(y[p*rc_],y[p*rc_],tmp,rc);
                }
            }
        } else {
            Gather(y,right,ix,0,lanes);
            if(mode_==1 && dense_) {Gather(work,grad,ix[2*vec_+2*rc_],0,lanes);Mul(y,y,work,lanes);}
            else Mul(y,y,mode_==0 ? left : grad,lanes);
            if(masked_) {Gather(work,mask,ix[vec_],0,lanes);Mul(y,y,work,lanes);}
        }
        output_.EnQue(y);auto ready=output_.DeQue<float>();
        if(mode_==2) {
            for(uint32_t path=0;path<paths_;path++) {
                const uint32_t p=table.GetValue(path*4+1);
                DataCopyExtParams cp{static_cast<uint16_t>(rn_),cn_*4,(bc_-A8(cn_))/8,(p_*c_-cn_)*4,0};
                DataCopyPad(out_[(firstN_*p_+p)*c_+firstC_],ready[p*rc_],cp);
            }
        } else if(mode_==1) {
            for(uint32_t row=0;row<rn_;row++) {
                DataCopyExtParams cp{static_cast<uint16_t>(q_),cn_*4,(bc_-A8(cn_))/8,(c_-cn_)*4,0};
                DataCopyPad(out_[((firstN_+row)*q_)*c_+firstC_],ready[row*q_*bc_],cp);
            }
        } else for(uint32_t row=0;row<rn_;row++) {
            const uint64_t node=destinations.GetValue(row);
            SetAtomicAdd<float>();
            for(uint32_t path=0;path<paths_;) {
                const uint32_t src=table.GetValue(path*4),out=table.GetValue(path*4+2);
                uint32_t width=table.GetValue(path*4+3);path++;
                while(path<paths_ && table.GetValue(path*4)==src+width && table.GetValue(path*4+2)==out+width) {
                    width+=table.GetValue(path*4+3);path++;
                }
                DataCopyExtParams cp{static_cast<uint16_t>(width),cn_*4,(bc_-A8(cn_))/8,(c_-cn_)*4,0};
                DataCopyPad(out_[(node*m_+out)*c_+firstC_],ready[(row*q_+src)*bc_],cp);
            }
            SetAtomicNone();
        }
        output_.FreeTensor(ready);input_.FreeTensor(x);
    }
    TPipe pipe_;TQue<TPosition::VECIN,2> input_;TQue<TPosition::VECOUT,2> output_;
    TBuf<TPosition::VECCALC> work_,index_,table_;
    GlobalTensor<float> l_,r_,g_,mask_,out_;GlobalTensor<int64_t> dst_;
    GlobalTensor<uint32_t> meta_,ix_;
    uint64_t n_,ls_,rs_,firstN_;
    uint32_t c_,q_,p_,m_,paths_,mode_,rows_,bc_,masked_,dense_,cb_,vec_,rc_;
    uint32_t lsize_,rsize_,gsize_,inputSize_,outputSize_,rn_,cn_,firstC_;
};
}
extern "C" __global__ __aicore__ void ictc_epilogue(
    GM_ADDR l,GM_ADDR r,GM_ADDR g,GM_ADDR dst,GM_ADDR mask,GM_ADDR meta,GM_ADDR ix,GM_ADDR out,
    uint64_t n,uint64_t ls,uint64_t rs,uint32_t c,uint32_t q,uint32_t p,uint32_t m,
    uint32_t paths,uint32_t mode,uint32_t rows,uint32_t bc,uint32_t masked,uint32_t dense) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Epilogue kernel;kernel.Init(l,r,g,dst,mask,meta,ix,out,n,ls,rs,c,q,p,m,paths,mode,rows,bc,masked,dense);kernel.Process();
}
