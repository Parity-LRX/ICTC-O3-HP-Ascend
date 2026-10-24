#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include <vector>
#include <map>
#include <mutex>
#include <numeric>
#include <tuple>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_ictc_binary.h"
#include "aclrtlaunch_ictc_weighted_map.h"
#include "map_params.h"

namespace {
std::vector<int64_t> hardware() {
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform, "CANN platform discovery failed");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    return {static_cast<int64_t>(ub), platform->GetCoreNumAiv()};
}

at::Tensor binary(const at::Tensor& x, const at::Tensor& y, int64_t mode, int64_t tile,
                  double alpha,double beta) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && y.device() == x.device(),
                "Expected two tensors on the same NPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
                "Ascend C kernels require FP32");
    const bool biased=mode==10;
    TORCH_CHECK((biased ? (y.dim()==1 && x.dim()>=1 && x.size(-1)==y.numel()) : x.sizes()==y.sizes()) &&
                x.is_contiguous() && y.is_contiguous(),
                "Expected equal contiguous shapes");
    TORCH_CHECK((mode>=0 && mode<=4) || (mode>=6 && mode<=10), "Invalid vector mode");
    c10_npu::NPUGuard guard(x.device());
    auto out=at::empty_like(x);
    if(!x.numel()) return out;
    const auto info = hardware();
    TORCH_CHECK(info[0] > 4096 && info[1] > 0, "Invalid UB or Vector Core capacity");
    const uint64_t maximum = ((info[0] - 4096) / ((biased ? 8 : 7) * sizeof(float)) / 64) * 64;
    const uint64_t columns=biased ? y.numel() : 1;
    const uint64_t quantum=biased ? (columns/std::gcd<uint64_t>(columns,64))*64 : 64;
    if(tile==0) {
        const uint64_t target=std::min<uint64_t>(4096,maximum);
        tile=(biased && quantum<=maximum) ? std::max<uint64_t>(quantum,(target/quantum)*quantum) : target;
    }
    TORCH_CHECK(tile > 0 && tile % 64 == 0 && static_cast<uint64_t>(tile) <= maximum,
                "Tile exceeds UB capacity or is not a multiple of 64");
    auto indices=x;
    if(biased && tile%columns==0) {
        static std::map<std::tuple<int64_t,int64_t,int64_t>,at::Tensor> cache;
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        const auto key=std::make_tuple(x.device().index(),columns,tile);
        auto item=cache.find(key);
        if(item==cache.end()) {
            auto cpu=at::empty({tile},at::TensorOptions().dtype(at::kInt).device(at::kCPU));
            auto* data=cpu.data_ptr<int32_t>();
            for(int64_t i=0;i<tile;i++) data[i]=static_cast<int32_t>((i%columns)*4);
            if(cache.size()>=64) cache.erase(cache.begin());
            item=cache.emplace(key,cpu.to(x.device())).first;
        }
        indices=item->second;
    } else if(biased) mode=11;
    const uint64_t tasks=mode==11 ? (x.numel()/columns)*((columns+tile-1)/tile) : (x.numel()+tile-1)/tile;
    const uint32_t blocks = std::min<uint64_t>(info[1],tasks);
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [x,y,out,indices,stream,blocks,tile,mode,alpha,beta,columns]() -> int {
        ACLRT_LAUNCH_KERNEL(ictc_binary)(blocks, stream,
            reinterpret_cast<uint8_t*>(x.data_ptr()), reinterpret_cast<uint8_t*>(y.data_ptr()),
            reinterpret_cast<uint8_t*>(out.data_ptr()), static_cast<uint64_t>(x.numel()),
            static_cast<uint32_t>(tile), static_cast<uint32_t>(mode),
            static_cast<float>(alpha),static_cast<float>(beta),
            reinterpret_cast<uint8_t*>(indices.data_ptr()),columns);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ictc_ascendc_binary", launch);
    return out;
}

uint64_t aligned(uint64_t n) { return (n+7)/8*8; }

at::Tensor weighted_map(const at::Tensor& l,const at::Tensor& r,const at::Tensor& w,
    const at::Tensor& g,const at::Tensor& species,const at::Tensor& meta,
    const at::Tensor& ix0,const at::Tensor& ix1,const at::Tensor& source,
    const at::Tensor& destination,const at::Tensor& mask,at::IntArrayRef shape) {
    TORCH_CHECK(shape.size()==20,"Invalid native map descriptor");
    IctcMapParams p{static_cast<uint64_t>(shape[0]),static_cast<uint32_t>(shape[1]),
        static_cast<uint32_t>(shape[2]),static_cast<uint32_t>(shape[3]),static_cast<uint32_t>(shape[4]),
        static_cast<uint32_t>(shape[5]),static_cast<uint32_t>(shape[6]),static_cast<uint32_t>(shape[7]),
        static_cast<uint32_t>(shape[8]),static_cast<uint32_t>(shape[9]),static_cast<uint32_t>(shape[10]),
        static_cast<uint32_t>(shape[11]),static_cast<uint32_t>(shape[12]),static_cast<uint32_t>(shape[13]),
        static_cast<uint32_t>(shape[14]),static_cast<uint32_t>(shape[15]),static_cast<uint32_t>(shape[16]),
        static_cast<uint64_t>(shape[17]),static_cast<uint32_t>(shape[18]),static_cast<uint32_t>(shape[19])};
    TORCH_CHECK(l.device().type()==c10::DeviceType::PrivateUse1,"Expected NPU inputs");
    for(const auto& value : {l,r,w,g})
        TORCH_CHECK(value.device()==l.device() && value.scalar_type()==at::kFloat && value.is_contiguous(),
                    "Native map requires contiguous FP32 operands on one NPU");
    for(const auto& value : {meta,ix0,ix1})
        TORCH_CHECK(value.device()==l.device() && value.scalar_type()==at::kInt && value.is_contiguous(),
                    "Native metadata must be contiguous NPU int32");
    if(p.graph) {
        for(const auto& value : {source,destination})
            TORCH_CHECK(value.device()==l.device() && value.scalar_type()==at::kLong &&
                        value.is_contiguous() && value.numel()==static_cast<int64_t>(p.nodes),
                        "Graph indices must be contiguous NPU int64 edge vectors");
    }
    if(p.masked) TORCH_CHECK(mask.device()==l.device() && mask.scalar_type()==at::kFloat &&
        mask.is_contiguous() && mask.numel()==static_cast<int64_t>(p.nodes),"Invalid graph mask");
    TORCH_CHECK(p.mode<4 && p.tile_rows>0 && p.tile_channels>0 && p.tile_channels<=p.channels,
                "Invalid native map tiling");
    TORCH_CHECK(static_cast<uint64_t>(p.tile_rows)*p.tile_channels<=4095,"DMA block count exceeds capacity");
    c10_npu::NPUGuard guard(l.device());
    const auto info=hardware();
    const bool shared=p.mode==2 && p.right_channels==1;
    const uint64_t nc=aligned(p.tile_rows*p.tile_channels);
    const uint64_t outputRows=shared ? p.tile_rows : p.tile_rows*p.tile_channels;
    const uint64_t pitches=(p.mode==1 ? 0 : aligned(p.tile_channels*p.left_width))+
        ((p.mode==0 || p.mode==1 || p.mode==3) ? aligned((p.right_channels==1 ? 1 : p.tile_channels)*p.right_width) : 0)+
        (p.mode==3 ? 0 : aligned(p.tile_channels*p.paths))+
        (p.mode==0 ? 0 : aligned(p.tile_channels*p.grad_width))+(p.masked ? 8 : 0);
    const uint64_t bytes=4*(2*p.tile_rows*pitches+(p.graph ? 4*aligned(2*p.tile_rows) : 0)+
        16*aligned(outputRows)+11*nc+4*nc+8*outputRows+aligned(p.tile_channels)+
        (p.masked ? nc : 0)+p.head+5*p.local_term_stride);
    TORCH_CHECK(bytes+4096<=static_cast<uint64_t>(info[0]),"Native map tile exceeds UB capacity");
    const uint32_t cb=(p.channels+p.tile_channels-1)/p.tile_channels;
    auto out=(p.mode==3 && p.bank) ? at::zeros_like(w) : (p.graph && p.mode<2) ?
        at::zeros({static_cast<int64_t>(p.output_nodes),static_cast<int64_t>(p.channels),
                   static_cast<int64_t>(p.output_width)},l.options()) :
        at::empty({static_cast<int64_t>(p.nodes),static_cast<int64_t>(shared ? cb : p.channels),
                   static_cast<int64_t>(p.output_width)},l.options());
    if(!p.nodes || !p.output_width) return shared ? out.sum({1},true) : out;
    const uint64_t tasks=((p.nodes+p.tile_rows-1)/p.tile_rows)*cb;
    const uint32_t blocks=std::min<uint64_t>(info[1],tasks);
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto launch=[l,r,w,g,species,meta,ix0,ix1,source,destination,mask,out,stream,p,blocks]() -> int {
        ACLRT_LAUNCH_KERNEL(ictc_weighted_map)(blocks,stream,
            reinterpret_cast<uint8_t*>(l.data_ptr()),reinterpret_cast<uint8_t*>(r.data_ptr()),
            reinterpret_cast<uint8_t*>(w.data_ptr()),reinterpret_cast<uint8_t*>(g.data_ptr()),
            reinterpret_cast<uint8_t*>(species.data_ptr()),reinterpret_cast<uint8_t*>(meta.data_ptr()),
            reinterpret_cast<uint8_t*>(ix0.data_ptr()),reinterpret_cast<uint8_t*>(ix1.data_ptr()),
            reinterpret_cast<uint8_t*>(source.data_ptr()),reinterpret_cast<uint8_t*>(destination.data_ptr()),
            reinterpret_cast<uint8_t*>(mask.data_ptr()),
            reinterpret_cast<uint8_t*>(out.data_ptr()),p.nodes,p.channels,p.right_channels,
            p.left_width,p.right_width,p.paths,p.grad_width,p.output_width,p.mode,p.bank,p.indexed,
            p.tile_rows,p.tile_channels,p.terms,p.head,p.global_term_stride,p.local_term_stride,
            p.output_nodes,p.graph,p.masked);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ictc_ascendc_weighted_map",launch);
    return shared && cb>1 ? out.sum({1},true) : out;
}
} // namespace

at::Tensor ictc_ascendc_add_bias(const at::Tensor& x,const at::Tensor& bias) {
    return binary(x,bias,10,0,1.,0.);
}

TORCH_LIBRARY(ictc_ascendc, m) {
    m.def("hardware() -> int[]", &hardware);
    m.def("binary(Tensor x, Tensor y, int mode=0, int tile=0, float alpha=1., float beta=0.) -> Tensor");
    m.def("weighted_map(Tensor l, Tensor r, Tensor w, Tensor g, Tensor species, Tensor metadata, Tensor ix0, Tensor ix1, Tensor source, Tensor destination, Tensor mask, int[] shape) -> Tensor");
}
TORCH_LIBRARY_IMPL(ictc_ascendc, PrivateUse1, m) {
    m.impl("binary", TORCH_FN(binary));
    m.impl("weighted_map", TORCH_FN(weighted_map));
}
