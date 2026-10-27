#include "kernel_operator.h"
#include "kernel_launch.h"
using namespace AscendC;
namespace {
__aicore__ inline uint32_t A8(uint32_t n) {return (n+7)/8*8;}
class BatchedVector {
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR out,GM_ADDR indices,
        uint64_t n,uint64_t as,uint64_t bs,uint32_t q,uint32_t i,uint32_t c,uint32_t mode,uint32_t rows,uint32_t bc) {
        n_=n;as_=as;bs_=bs;q_=q;i_=i;c_=c;mode_=mode;rows_=rows;bc_=bc;cb_=(c+bc-1)/bc;
        vec_=rows*q*bc;ap_=A8(q*i);aSize_=mode==1?vec_:rows*ap_;
        inputSize_=aSize_+rows*i*bc;outputSize_=mode==1?A8(rows*q*i):vec_;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));ix_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(indices));
        pipe_.InitBuffer(input_,2,inputSize_*4);pipe_.InitBuffer(output_,2,outputSize_*4);
        pipe_.InitBuffer(work_,(2*vec_+i*A8(rows*q))*4);pipe_.InitBuffer(index_,(2*vec_+A8(rows*q*i))*4);
        DataCopy(index_.Get<uint32_t>(),ix_,2*vec_+A8(rows*q*i));
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
    __aicore__ inline void Describe(uint64_t task) {
        firstN_=(task/cb_)*rows_;firstC_=(task%cb_)*bc_;
        rn_=n_-firstN_<rows_?n_-firstN_:rows_;cn_=c_-firstC_<bc_?c_-firstC_:bc_;
    }
    __aicore__ inline void Columns(LocalTensor<float> d,GlobalTensor<float> s,uint64_t offset,uint32_t count) {
        DataCopyExtParams cp{static_cast<uint16_t>(count),cn_*4,(c_-cn_)*4,(bc_-A8(cn_))/8,0};
        DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(d,s[offset],cp,pad);
    }
    __aicore__ inline void Read(LocalTensor<float> d,GlobalTensor<float> s,uint64_t stride,uint32_t width) {
        if(stride==static_cast<uint64_t>(width)*c_)Columns(d,s,firstN_*stride+firstC_,rn_*width);
        else for(uint32_t row=0;row<rn_;row++)Columns(d[row*width*bc_],s,(firstN_+row)*stride+firstC_,width);
    }
    __aicore__ inline void CopyIn(uint64_t task) {
        Describe(task);auto x=input_.AllocTensor<float>();
        if(mode_==1)Read(x,a_,as_,q_);
        else {
            DataCopyExtParams cp{static_cast<uint16_t>(rn_),q_*i_*4,static_cast<uint32_t>((as_-q_*i_)*4),0,0};
            DataCopyPadExtParams<float> pad{false,0,0,0};DataCopyPad(x,a_[firstN_*as_],cp,pad);
        }
        Read(x[aSize_],b_,bs_,i_);input_.EnQue(x);
    }
    __aicore__ inline void Compute(uint64_t task) {
        Describe(task);const uint32_t lanes=rn_*q_*bc_,rqs=rn_*q_;
        auto x=input_.DeQue<float>();auto y=output_.AllocTensor<float>();
        auto work=work_.Get<float>();auto v=work[vec_];auto reduced=work[2*vec_];auto ix=index_.Get<uint32_t>();
        for(uint32_t i=0;i<i_;i++) {
            Gather(v,x[aSize_],ix[vec_],i*bc_*4,lanes);
            if(mode_==1) {
                Mul(v,v,x,lanes);
                WholeReduceSum(reduced[i*A8(rows_*q_)],v,cn_,rqs,1,1,bc_/8);
            } else {
                Gather(work,x,ix,i*(mode_==2?q_:1)*4,lanes);Mul(work,work,v,lanes);
                if(i==0)Adds(y,work,0.f,lanes);else Add(y,y,work,lanes);
            }
        }
        if(mode_==1)Gather(y,reduced,ix[2*vec_],0,rqs*i_);
        output_.EnQue(y);auto ready=output_.DeQue<float>();
        if(mode_==1) {
            if(cb_>1)SetAtomicAdd<float>();
            DataCopyExtParams cp{1,rqs*i_*4,0,0,0};
            DataCopyPad(out_[firstN_*q_*i_],ready,cp);
            if(cb_>1)SetAtomicNone();
        } else {
            DataCopyExtParams cp{static_cast<uint16_t>(rqs),cn_*4,(bc_-A8(cn_))/8,(c_-cn_)*4,0};
            DataCopyPad(out_[firstN_*q_*c_+firstC_],ready,cp);
        }
        output_.FreeTensor(ready);input_.FreeTensor(x);
    }
    TPipe pipe_;TQue<TPosition::VECIN,2> input_;TQue<TPosition::VECOUT,2> output_;
    TBuf<TPosition::VECCALC> work_,index_;GlobalTensor<float> a_,b_,out_;GlobalTensor<uint32_t> ix_;
    uint64_t n_,as_,bs_,firstN_;
    uint32_t q_,i_,c_,mode_,rows_,bc_,cb_,vec_,ap_,aSize_,inputSize_,outputSize_,rn_,cn_,firstC_;
};
}
extern "C" __global__ __aicore__ void ictc_batched_vector(GM_ADDR a,GM_ADDR b,GM_ADDR out,GM_ADDR ix,
    uint64_t n,uint64_t as,uint64_t bs,uint32_t q,uint32_t i,uint32_t c,uint32_t mode,uint32_t rows,uint32_t bc) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchedVector kernel;kernel.Init(a,b,out,ix,n,as,bs,q,i,c,mode,rows,bc);kernel.Process();
}

// Host launcher; compiled by the ASC toolchain.

namespace ictc_ascendc {
void launch_batched_vector(uint32_t blocks, aclrtStream stream,
    uint8_t* a, uint8_t* b, uint8_t* out, uint8_t* ix, uint64_t n, uint64_t as, uint64_t bs,
    uint32_t q, uint32_t i, uint32_t c, uint32_t mode, uint32_t rows, uint32_t bc) {
    ictc_batched_vector<<<blocks, nullptr, stream>>>(
        a, b, out, ix, n, as, bs, q, i, c, mode, rows, bc);
}
} // namespace ictc_ascendc
