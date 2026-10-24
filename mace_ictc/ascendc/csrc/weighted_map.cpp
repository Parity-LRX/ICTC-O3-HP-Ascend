#include "kernel_operator.h"
#include "map_params.h"
using namespace AscendC;

namespace {
__aicore__ inline uint32_t Align(uint32_t x) { return (x+7)/8*8; }

class WeightedMap {
public:
    __aicore__ inline void Init(GM_ADDR l, GM_ADDR r, GM_ADDR w, GM_ADDR g,
        GM_ADDR species, GM_ADDR meta, GM_ADDR ix0, GM_ADDR ix1,GM_ADDR source,
        GM_ADDR destination,GM_ADDR mask, GM_ADDR out,
        IctcMapParams params) {
        p_=params;
        l_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(l));
        r_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(r));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        species64_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(species));
        species32_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(species));
        metadata_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(meta));
        indices0_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ix0));
        indices1_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ix1));
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(source));
        destination_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(destination));
        mask_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mask));
        cb_=(p_.channels+p_.tile_channels-1)/p_.tile_channels;
        shared_=p_.mode==2 && p_.right_channels==1;
        cached_=p_.terms<=p_.local_term_stride;
        const uint32_t nc=Align(p_.tile_rows*p_.tile_channels);
        const uint32_t outputRows=shared_ ? p_.tile_rows : p_.tile_rows*p_.tile_channels;
        const uint32_t inputRows=RowPitches(p_.tile_channels);
        pipe_.InitBuffer(input_,2,(p_.tile_rows*inputRows+(p_.graph ? 2*Align(2*p_.tile_rows) : 0))*sizeof(float));
        pipe_.InitBuffer(output_,2,Align(outputRows)*8*sizeof(float));
        pipe_.InitBuffer(raw_,8*nc*sizeof(float));
        pipe_.InitBuffer(work_,3*nc*sizeof(float));
        pipe_.InitBuffer(index_,(4*nc+8*outputRows+Align(p_.tile_channels)+(p_.masked ? nc : 0))*sizeof(uint32_t));
        pipe_.InitBuffer(meta_,(p_.head+5*p_.local_term_stride)*sizeof(uint32_t));
        auto table=meta_.Get<uint32_t>();
        Copy(table,metadata_,0,cached_ ? p_.head+5*p_.local_term_stride : p_.head);
        Sync<HardEvent::MTE2_S>();
    }

    __aicore__ inline void Process() {
        const uint64_t tasks=((p_.nodes+p_.tile_rows-1)/p_.tile_rows)*cb_;
        uint64_t task=GetBlockIdx();
        if (task>=tasks) return;
        CopyIn(task);
        for (;task<tasks;task+=GetBlockNum()) {
            const uint64_t next=task+GetBlockNum();
            if(next<tasks) CopyIn(next);
            Compute(task);
        }
    }
private:
    template<HardEvent event> __aicore__ inline void Sync() {
        const event_t id=static_cast<event_t>(pipe_.FetchEventID(event));
        SetFlag<event>(id); WaitFlag<event>(id);
    }
    template<typename T> __aicore__ inline void Copy(LocalTensor<T> dst,
        GlobalTensor<T> src,uint64_t begin,uint32_t count) {
        if(!count) return;
        DataCopyExtParams cp{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{false,0,0,0};
        DataCopyPad(dst,src[begin],cp,pad);
    }
    __aicore__ inline uint32_t RowPitches(uint32_t c) {
        lp_=p_.mode==1 ? 0 : Align(c*p_.left_width);
        rp_=(p_.mode==0 || p_.mode==1 || p_.mode==3) ?
            Align((p_.right_channels==1 ? 1 : c)*p_.right_width) : 0;
        wp_=p_.mode==3 ? 0 : Align(c*p_.paths);
        gp_=p_.mode==0 ? 0 : Align(c*p_.grad_width);
        return lp_+rp_+wp_+gp_+(p_.masked ? 8 : 0);
    }
    __aicore__ inline uint64_t Species(uint64_t node) {
        if(!p_.bank) return node;
        if(!p_.indexed) return 0;
        return p_.indexed==2 ? static_cast<uint64_t>(species32_.GetValue(node)) :
                              static_cast<uint64_t>(species64_.GetValue(node));
    }
    __aicore__ inline void CopyRows(LocalTensor<float> dst,GlobalTensor<float> src,
        uint64_t begin,uint32_t width,uint32_t stride,uint32_t rows) {
        if(!width) return;
        DataCopyExtParams cp{static_cast<uint16_t>(rows),width*4,(stride-width)*4,0,0};
        DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(dst,src[begin],cp,pad);
    }
    __aicore__ inline void CopyIn(uint64_t task) {
        const uint32_t cb=task%cb_;
        const uint64_t node=(task/cb_)*p_.tile_rows;
        const uint32_t rows=p_.nodes-node<p_.tile_rows ? p_.nodes-node : p_.tile_rows;
        const uint32_t firstC=cb*p_.tile_channels;
        const uint32_t c=p_.channels-firstC<p_.tile_channels ? p_.channels-firstC : p_.tile_channels;
        const uint32_t stride=RowPitches(c);
        auto data=input_.AllocTensor<float>();
        auto sources=data[p_.graph ? p_.tile_rows*stride : 0].ReinterpretCast<int64_t>();
        auto destinations=data[p_.graph ? p_.tile_rows*stride+Align(2*p_.tile_rows) : 0].ReinterpretCast<int64_t>();
        if(p_.graph) {
            Copy(sources,source_,node,rows);
            Copy(destinations,destination_,node,rows);
            Sync<HardEvent::MTE2_S>();
        }
        uint32_t offset=0;
        if(lp_) {
            if(p_.graph) {
                for(uint32_t row=0;row<rows;row++)
                    Copy(data[row*lp_],l_,(static_cast<uint64_t>(sources.GetValue(row))*p_.channels+firstC)*p_.left_width,c*p_.left_width);
            } else CopyRows(data,l_,(node*p_.channels+firstC)*p_.left_width,
                            c*p_.left_width,p_.channels*p_.left_width,rows);
        }
        offset+=p_.tile_rows*lp_;
        if(rp_) CopyRows(data[offset],r_,(node*p_.right_channels+(p_.right_channels==1 ? 0 : firstC))*p_.right_width,
                        (p_.right_channels==1 ? 1 : c)*p_.right_width,p_.right_channels*p_.right_width,rows);
        offset+=p_.tile_rows*rp_;
        if(wp_) {
            for(uint32_t row=0;row<rows;row++)
                Copy(data[offset+row*wp_],w_,(Species(node+row)*p_.channels+firstC)*p_.paths,c*p_.paths);
        }
        offset+=p_.tile_rows*wp_;
        if(gp_) {
            if(p_.graph) {
                for(uint32_t row=0;row<rows;row++)
                    Copy(data[offset+row*gp_],g_,(static_cast<uint64_t>(destinations.GetValue(row))*p_.channels+firstC)*p_.grad_width,c*p_.grad_width);
            } else CopyRows(data[offset],g_,(node*p_.channels+firstC)*p_.grad_width,
                            c*p_.grad_width,p_.channels*p_.grad_width,rows);
        }
        offset+=p_.tile_rows*gp_;
        if(p_.masked) CopyRows(data[offset],mask_,node,1,1,rows);
        input_.EnQue(data);
    }
    __aicore__ inline void LoadTerms(uint32_t start,uint32_t count) {
        auto table=meta_.Get<uint32_t>();
        Sync<HardEvent::S_MTE2>();
        for(uint32_t column=0;column<5;column++)
            Copy(table[p_.head+column*p_.local_term_stride],metadata_,
                 p_.head+column*p_.global_term_stride+start,count);
        Sync<HardEvent::MTE2_S>();
    }
    __aicore__ inline void Compute(uint64_t task) {
        const uint32_t cb=task%cb_;
        const uint64_t firstN=(task/cb_)*p_.tile_rows;
        const uint32_t rows=p_.nodes-firstN<p_.tile_rows ? p_.nodes-firstN : p_.tile_rows;
        const uint32_t firstC=cb*p_.tile_channels;
        const uint32_t c=p_.channels-firstC<p_.tile_channels ? p_.channels-firstC : p_.tile_channels;
        const uint32_t stride=RowPitches(c);
        const uint32_t cap=Align(p_.tile_rows*c),count=rows*c;
        const uint32_t outputCap=shared_ ? p_.tile_rows : p_.tile_rows*c;
        const uint32_t outputCount=shared_ ? rows : count;
        const uint32_t ixWords=4*cap+8*outputCap+Align(c)+(p_.masked ? cap : 0);
        auto ix=index_.Get<uint32_t>();
        if(lastC_!=c) {
            Sync<HardEvent::V_MTE2>();
            Copy(ix,c==p_.tile_channels ? indices0_ : indices1_,0,ixWords);
            Sync<HardEvent::MTE2_V>();
            lastC_=c;
        }
        auto data=input_.DeQue<float>();
        auto left=data;
        auto right=data[p_.tile_rows*lp_];
        auto weight=data[p_.tile_rows*(lp_+rp_)];
        auto grad=data[p_.tile_rows*(lp_+rp_+wp_)];
        auto mask=data[p_.tile_rows*(lp_+rp_+wp_+gp_)];
        auto sources=data[p_.graph ? p_.tile_rows*stride : 0].ReinterpretCast<int64_t>();
        auto destinations=data[p_.graph ? p_.tile_rows*stride+Align(2*p_.tile_rows) : 0].ReinterpretCast<int64_t>();
        auto raw=raw_.Get<float>();
        auto work=work_.Get<float>();
        auto v0=work, v1=work[Align(p_.tile_rows*p_.tile_channels)],
             v2=work[2*Align(p_.tile_rows*p_.tile_channels)];
        auto table=meta_.Get<uint32_t>();
        auto coefs=meta_.Get<float>();
        for(uint32_t begin=0;begin<p_.output_width;begin+=8) {
            const uint32_t qCount=p_.output_width-begin<8 ? p_.output_width-begin : 8;
            Duplicate(raw,0.f,8*cap);
            for(uint32_t q=0;q<qCount;q++) {
                const uint32_t start=table.GetValue(begin+q),end=table.GetValue(begin+q+1);
                for(uint32_t chunk=start;chunk<end;) {
                    const uint32_t terms=cached_ ? end-chunk :
                        (end-chunk<p_.local_term_stride ? end-chunk : p_.local_term_stride);
                    if(!cached_) LoadTerms(chunk,terms);
                    const uint32_t local=cached_ ? chunk : 0;
                    for(uint32_t k=local;k<local+terms;k++) {
                        const uint32_t a=table.GetValue(p_.head+k);
                        const uint32_t b=table.GetValue(p_.head+p_.local_term_stride+k);
                        const uint32_t path=table.GetValue(p_.head+2*p_.local_term_stride+k);
                        const uint32_t g=table.GetValue(p_.head+3*p_.local_term_stride+k);
                        const float coef=coefs.GetValue(p_.head+4*p_.local_term_stride+k);
                        if(p_.mode==1) {
                            Gather(v0,right,ix[cap],b*4,count);
                            Gather(v1,grad,ix[3*cap],g*4,count);
                        } else {
                            Gather(v0,left,ix,a*4,count);
                            if(p_.mode==2) Gather(v1,grad,ix[3*cap],g*4,count);
                            else Gather(v1,right,ix[cap],b*4,count);
                        }
                        if(p_.mode==3) Gather(v2,grad,ix[3*cap],g*4,count);
                        else Gather(v2,weight,ix[2*cap],path*4,count);
                        Mul(v0,v0,v1,count);
                        Muls(v2,v2,coef,count);
                        Mul(v0,v0,v2,count);
                        Add(raw[q*cap],raw[q*cap],v0,count);
                    }
                    chunk+=terms;
                }
                if(p_.masked) {
                    Gather(v0,mask,ix[4*cap+8*outputCap+Align(c)],0,count);
                    Mul(raw[q*cap],raw[q*cap],v0,count);
                }
                if(shared_) {
                    for(uint32_t row=0;row<rows;row++) {
                        Gather(v0,raw[q*cap],ix[4*cap+8*outputCap],row*c*4,c);
                        ReduceSum(raw[q*cap+row],v0,v1,c);
                    }
                }
            }
            auto result=output_.AllocTensor<float>();
            Gather(result,raw,ix[4*cap],0,outputCount*8);
            output_.EnQue(result);
            auto ready=output_.DeQue<float>();
            if(shared_) {
                DataCopyExtParams cp{static_cast<uint16_t>(rows),qCount*4,0,
                    (cb_*p_.output_width-qCount)*4,0};
                DataCopyPad(out_[(firstN*cb_+cb)*p_.output_width+begin],ready,cp);
            } else if((p_.mode==3 && p_.bank) || (p_.graph && p_.mode<2)) {
                for(uint32_t row=0;row<rows;row++) {
                    DataCopyExtParams cp{static_cast<uint16_t>(c),qCount*4,0,(p_.output_width-qCount)*4,0};
                    const uint64_t target=(p_.graph && p_.mode<2) ? static_cast<uint64_t>(p_.mode==0 ?
                        destinations.GetValue(row) : sources.GetValue(row)) : Species(firstN+row);
                    SetAtomicAdd<float>();
                    DataCopyPad(out_[(target*p_.channels+firstC)*p_.output_width+begin],ready[row*c*8],cp);
                    SetAtomicNone();
                }
            } else {
                if(c==p_.channels) {
                    DataCopyExtParams cp{static_cast<uint16_t>(count),qCount*4,0,(p_.output_width-qCount)*4,0};
                    DataCopyPad(out_[(firstN*p_.channels+firstC)*p_.output_width+begin],ready,cp);
                } else for(uint32_t row=0;row<rows;row++) {
                    DataCopyExtParams cp{static_cast<uint16_t>(c),qCount*4,0,(p_.output_width-qCount)*4,0};
                    DataCopyPad(out_[((firstN+row)*p_.channels+firstC)*p_.output_width+begin],ready[row*c*8],cp);
                }
            }
            output_.FreeTensor(ready);
        }
        input_.FreeTensor(data);
    }
    IctcMapParams p_;
    TPipe pipe_;
    TQue<TPosition::VECIN,2> input_;
    TQue<TPosition::VECOUT,2> output_;
    TBuf<TPosition::VECCALC> raw_,work_,index_,meta_;
    GlobalTensor<float> l_,r_,w_,g_,out_;
    GlobalTensor<int64_t> species64_;
    GlobalTensor<int64_t> source_,destination_;
    GlobalTensor<float> mask_;
    GlobalTensor<int32_t> species32_;
    GlobalTensor<uint32_t> metadata_,indices0_,indices1_;
    uint32_t cb_,lp_,rp_,wp_,gp_,lastC_=0;
    bool shared_,cached_;
};
} // namespace

extern "C" __global__ __aicore__ void ictc_weighted_map(
    GM_ADDR l,GM_ADDR r,GM_ADDR w,GM_ADDR g,GM_ADDR species,GM_ADDR meta,
    GM_ADDR ix0,GM_ADDR ix1,GM_ADDR source,GM_ADDR destination,GM_ADDR mask,GM_ADDR out,
    uint64_t n,uint32_t c,uint32_t rc,
    uint32_t a,uint32_t b,uint32_t paths,uint32_t m,uint32_t o,uint32_t mode,
    uint32_t bank,uint32_t indexed,uint32_t rows,uint32_t channels,
    uint32_t terms,uint32_t head,uint32_t globalStride,uint32_t localStride,
    uint64_t outputNodes,uint32_t graph,uint32_t masked) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    WeightedMap kernel;
    kernel.Init(l,r,w,g,species,meta,ix0,ix1,source,destination,mask,out,
        {n,c,rc,a,b,paths,m,o,mode,bank,indexed,rows,channels,terms,head,globalStride,localStride,outputNodes,graph,masked});
    kernel.Process();
}
