#include "kernel_operator.h"

using namespace AscendC;

class Binary {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint64_t size, uint32_t tile, uint32_t mode,
                                float alpha, float beta,GM_ADDR indices,uint64_t columns) {
        size_ = size; tile_ = tile; mode_ = mode; alpha_=alpha; beta_=beta;
        columns_=columns;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        z_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z));
        pipe_.InitBuffer(xq_, 2, tile * sizeof(float));
        pipe_.InitBuffer(yq_, 2, tile * sizeof(float));
        pipe_.InitBuffer(zq_, 2, tile * sizeof(float));
        pipe_.InitBuffer(work_, tile * sizeof(float));
        if(mode_==10) {
            pipe_.InitBuffer(index_,tile*sizeof(uint32_t));
            GlobalTensor<uint32_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(indices));
            DataCopy(index_.Get<uint32_t>(),gm,tile);
            const auto event=static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(event); WaitFlag<HardEvent::MTE2_V>(event);
        }
    }
    __aicore__ inline void Process() {
        const uint64_t tiles = mode_==11 ? (size_/columns_)*((columns_+tile_-1)/tile_) :
                               (size_ + tile_ - 1) / tile_;
        uint64_t task = GetBlockIdx();
        if (task >= tiles) return;
        CopyIn(task);
        for (; task < tiles; task += GetBlockNum()) {
            const uint64_t next = task + GetBlockNum();
            if (next < tiles) CopyIn(next);
            const uint32_t count = Count(task);
            auto x = xq_.DeQue<float>();
            auto y = yq_.DeQue<float>();
            auto z = zq_.AllocTensor<float>();
            if (mode_ == 0) Add(z, x, y, count);
            else if(mode_ == 1) Mul(z, x, y, count);
            else if(mode_==10) {
                auto expanded=work_.Get<float>();
                Gather(expanded,y,index_.Get<uint32_t>(),0,count);
                Add(z,x,expanded,count);
            } else if(mode_==11) Add(z,x,y,count);
            else if(mode_ == 4) {
                Muls(z,x,alpha_,count); Adds(z,z,beta_,count);
            } else {
                auto s=work_.Get<float>();
                Muls(s,x,-1.f,count); Exp(s,s,count); Adds(s,s,1.f,count);
                Duplicate(z,1.f,count); Div(s,z,s,count);
                if(mode_ == 2) Muls(z,s,1.f,count);
                else if(mode_ == 3) {
                    Mul(z,x,s,count); Muls(z,z,alpha_,count);
                } else if(mode_ == 6) {
                    Muls(z,s,-1.f,count); Adds(z,z,1.f,count);
                    Mul(z,z,x,count); Adds(z,z,1.f,count); Mul(z,z,s,count);
                    Mul(z,z,y,count); Muls(z,z,alpha_,count);
                } else if(mode_ == 7) {
                    Muls(z,s,-2.f,count); Adds(z,z,1.f,count);
                    Mul(z,z,x,count); Adds(z,z,2.f,count);
                    Muls(x,s,-1.f,count); Adds(x,x,1.f,count); Mul(x,x,s,count);
                    Mul(z,z,x,count); Mul(z,z,y,count); Muls(z,z,alpha_,count);
                } else {
                    Muls(z,s,-1.f,count); Adds(z,z,1.f,count); Mul(z,z,s,count);
                    if(mode_ == 9) {
                        Muls(x,s,-2.f,count); Adds(x,x,1.f,count); Mul(z,z,x,count);
                    }
                    Mul(z,z,y,count);
                }
            }
            zq_.EnQue(z);
            xq_.FreeTensor(x); yq_.FreeTensor(y);
            auto out = zq_.DeQue<float>();
            DataCopyExtParams p {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(z_[Offset(task)], out, p);
            zq_.FreeTensor(out);
        }
    }
private:
    __aicore__ inline uint32_t Count(uint64_t task) const {
        const uint64_t remaining = mode_==11 ? columns_-(task%((columns_+tile_-1)/tile_))*tile_ :
                                   size_ - task * tile_;
        return remaining < tile_ ? static_cast<uint32_t>(remaining) : tile_;
    }
    __aicore__ inline uint64_t Offset(uint64_t task) const {
        if(mode_!=11) return task*tile_;
        const uint64_t chunks=(columns_+tile_-1)/tile_;
        return (task/chunks)*columns_+(task%chunks)*tile_;
    }
    __aicore__ inline void CopyIn(uint64_t task) {
        auto x = xq_.AllocTensor<float>();
        auto y = yq_.AllocTensor<float>();
        DataCopyExtParams p {1, static_cast<uint32_t>(Count(task) * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad {false, 0, 0, 0};
        DataCopyPad(x, x_[Offset(task)], p, pad);
        if(mode_==10) {
            DataCopyExtParams cp {1,static_cast<uint32_t>(columns_*sizeof(float)),0,0,0};
            DataCopyPad(y,y_,cp,pad);
        } else DataCopyPad(y,y_[mode_==11 ? Offset(task)%columns_ : Offset(task)],p,pad);
        xq_.EnQue(x); yq_.EnQue(y);
    }
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> xq_, yq_;
    TQue<TPosition::VECOUT, 2> zq_;
    TBuf<TPosition::VECCALC> work_;
    TBuf<TPosition::VECCALC> index_;
    GlobalTensor<float> x_, y_, z_;
    uint64_t size_;
    uint64_t columns_;
    uint32_t tile_, mode_;
    float alpha_,beta_;
};

extern "C" __global__ __aicore__ void ictc_binary(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t size, uint32_t tile, uint32_t mode,
    float alpha,float beta,GM_ADDR indices,uint64_t columns) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Binary kernel;
    kernel.Init(x, y, z, size, tile, mode,alpha,beta,indices,columns);
    kernel.Process();
}
