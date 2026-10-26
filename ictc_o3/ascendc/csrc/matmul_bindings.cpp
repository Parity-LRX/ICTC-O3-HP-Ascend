#include <ATen/ATen.h>
#include <torch/library.h>
#include <climits>
#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_ictc_matmul.h"

at::Tensor ictc_ascendc_add_bias(const at::Tensor&,const at::Tensor&);

namespace {
struct Plan { at::Tensor data; uint32_t cores; uint64_t workspace; };
using Key=std::tuple<int64_t,int64_t,int64_t,int64_t,bool,bool,bool>;
std::map<Key,Plan> plans;
std::mutex planMutex;

Plan plan(const at::Tensor& a,int64_t m,int64_t n,int64_t k,bool ta,bool tb,bool bias) {
    const Key key{a.device().index(),m,n,k,ta,tb,bias};
    std::lock_guard<std::mutex> lock(planMutex);
    auto found=plans.find(key);
    if(found!=plans.end()) return found->second;
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform,"Ascend C platform discovery failed");
    matmul_tiling::MultiCoreMatmulTiling tiler(*platform);
    using namespace matmul_tiling;
    tiler.SetDim(platform->GetCoreNumAic());
    tiler.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT,ta);
    tiler.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT,tb);
    tiler.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT);
    tiler.SetOrgShape(m,n,k);
    tiler.SetShape(m,n,k);
    tiler.SetBias(bias);
    if(bias) tiler.SetBiasType(TPosition::GM,CubeFormat::ND,DataType::DT_FLOAT);
    uint64_t l1=0,l0c=0,ub=0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    tiler.SetBufferSpace(l1,l0c,ub>4096 ? ub-4096 : ub);
    optiling::TCubeTiling descriptor;
    TORCH_CHECK(tiler.GetTiling(descriptor)!=-1,"Native FP32 matmul tiling failed for ",m,"x",n,"x",k);
    auto cpu=at::empty({static_cast<int64_t>(descriptor.GetDataSize())},at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    descriptor.SaveToBuffer(cpu.data_ptr<uint8_t>(),cpu.numel());
    Plan result{cpu.to(a.device()),static_cast<uint32_t>(descriptor.get_usedCoreNum()),
                std::max<uint64_t>(1,platform->GetLibApiWorkSpaceSize())};
    if(plans.size()>=128) plans.erase(plans.begin());
    plans.emplace(key,result);
    return result;
}

at::Tensor mm(const at::Tensor& a,const at::Tensor& b,bool ta,bool tb,
              const std::optional<at::Tensor>& bias) {
    TORCH_CHECK(a.device().type()==c10::DeviceType::PrivateUse1 && b.device()==a.device(),
                "Native matmul requires operands on the same NPU");
    TORCH_CHECK(a.scalar_type()==at::kFloat && b.scalar_type()==at::kFloat && a.dim()==2 && b.dim()==2 &&
                a.is_contiguous() && b.is_contiguous(),"Native matmul requires contiguous FP32 matrices");
    const int64_t m=a.size(ta ? 1 : 0),k=a.size(ta ? 0 : 1),n=b.size(tb ? 0 : 1);
    TORCH_CHECK(k==b.size(tb ? 1 : 0) && m<=INT_MAX && n<=INT_MAX && k<=INT_MAX,
                "Invalid native matmul dimensions");
    if(bias) TORCH_CHECK(bias->device()==a.device() && bias->scalar_type()==at::kFloat &&
                        bias->is_contiguous() && bias->dim()==1 && bias->numel()==n,"Invalid FP32 bias");
    c10_npu::NPUGuard guard(a.device());
    if(!m || !n || !k) {
        auto out=at::zeros({m,n},a.options());
        return bias ? ictc_ascendc_add_bias(out,*bias) : out;
    }
    const auto schedule=plan(a,m,n,k,ta,tb,false);
    auto out=at::empty({m,n},a.options());
    auto scratch=at::empty({static_cast<int64_t>(schedule.workspace)},a.options().dtype(at::kByte));
    auto biasTensor=bias ? *bias : a;
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto launch=[a,b,biasTensor,out,scratch,schedule,stream,ta,tb]() -> int {
        ACLRT_LAUNCH_KERNEL(ictc_matmul)(schedule.cores,stream,
            reinterpret_cast<uint8_t*>(a.data_ptr()),reinterpret_cast<uint8_t*>(b.data_ptr()),
            reinterpret_cast<uint8_t*>(biasTensor.data_ptr()),reinterpret_cast<uint8_t*>(out.data_ptr()),
            reinterpret_cast<uint8_t*>(scratch.data_ptr()),reinterpret_cast<uint8_t*>(schedule.data.data_ptr()),
            static_cast<uint32_t>(ta),static_cast<uint32_t>(tb),static_cast<uint32_t>(0));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ictc_ascendc_matmul",launch);
    return bias ? ictc_ascendc_add_bias(out,*bias) : out;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(ictc_ascendc,m) {
    m.def("mm(Tensor a, Tensor b, bool transpose_a=False, bool transpose_b=False, Tensor? bias=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(ictc_ascendc,PrivateUse1,m) {
    m.impl("mm",TORCH_FN(mm));
}
