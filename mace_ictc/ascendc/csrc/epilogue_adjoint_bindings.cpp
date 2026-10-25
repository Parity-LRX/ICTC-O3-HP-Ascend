#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_ictc_epilogue_adjoint.h"
namespace {
uint64_t align8(uint64_t n) { return (n+7)/8*8; }
std::vector<at::Tensor> epilogue_adjoint(const at::Tensor& l,const at::Tensor& r,const at::Tensor& g,
    const at::Tensor& hl,const at::Tensor& hr,const at::Tensor& dst,const at::Tensor& mask,
    const at::Tensor& meta,const at::Tensor& ix,at::IntArrayRef s) {
    TORCH_CHECK((s.size()==11 || s.size()==12),"Invalid fused epilogue adjoint descriptor");
    for(auto v:s)TORCH_CHECK(v>=0,"Negative fused adjoint dimension");
    const uint32_t hoist=s.size()==12?s[11]:0;TORCH_CHECK((hoist&~57u)==0,"Invalid transfer policy");
    const uint64_t n=s[0],nodes=s[9];
    const uint32_t c=s[1],q=s[2],p=s[3],m=s[4],second=s[5],rows=s[6],bc=s[7],masked=s[8],flags=s[10];
    const bool dl=flags&1,dr=flags&2,dg=flags&4,hasHL=flags&8,hasHR=flags&16;
    TORCH_CHECK(c && q && p && m && q<=4095 && p<=4095 && second<=1 && rows && rows<=4095 &&
        bc && bc%8==0 && static_cast<uint64_t>(rows)*q*bc<=8192 && masked<=1 && flags<32 && (dl || dr || dg),
        "Invalid fused epilogue adjoint tile");
    TORCH_CHECK(second ? ((!dl || hasHR) && (!dr || hasHL) && (!dg || hasHL || hasHR)) :
        (!dg && !hasHL && !hasHR),"Invalid fused adjoint dependencies");
    TORCH_CHECK(l.device().type()==c10::DeviceType::PrivateUse1,"Expected NPU activations");
    for(const auto& value:{l,r,g,hl,hr})TORCH_CHECK(value.device()==l.device() && value.scalar_type()==at::kFloat,
        "Expected FP32 operands on one NPU");
    auto layout=[&](const at::Tensor& v,uint32_t width) {
        TORCH_CHECK(v.dim()==3 && v.size(0)==n && v.size(1)==width && v.size(2)==c &&
            v.stride(2)==1 && v.stride(1)==c && v.stride(0)>=static_cast<int64_t>(width)*c,"Invalid fused adjoint edge layout");
    };
    layout(l,q);layout(r,p);if(hasHL)layout(hl,q);if(hasHR)layout(hr,p);
    if(dl || dr)TORCH_CHECK(g.dim()==3 && g.size(0)==nodes && g.size(1)==m && g.size(2)==c && g.is_contiguous(),
        "Invalid fused adjoint node cotangent layout");
    TORCH_CHECK(dst.device()==l.device() && dst.scalar_type()==at::kLong && dst.is_contiguous() && dst.numel()==n,
        "Expected contiguous int64 destinations");
    for(const auto& v:{meta,ix})TORCH_CHECK(v.device()==l.device() && v.scalar_type()==at::kInt && v.is_contiguous(),
        "Expected contiguous int32 fused adjoint schedules");
    if(masked)TORCH_CHECK(mask.device()==l.device() && mask.scalar_type()==at::kFloat && mask.is_contiguous() && mask.numel()==n,
        "Invalid fixed fused adjoint mask");
    c10_npu::NPUGuard guard(l.device());
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"CANN platform discovery failed");
    uint64_t ub=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    const uint64_t vec=static_cast<uint64_t>(rows)*q*bc,rc=static_cast<uint64_t>(rows)*bc,pv=static_cast<uint64_t>(rows)*p*bc;
    const bool useL=(!second && dr) || (second && dg && hasHR);
    const bool useR=(!second && dl) || (second && dg && hasHL);
    const bool useHL=second && hasHL && (dr || dg),useHR=second && hasHR && (dl || dg);
    const uint64_t input=(useL?vec:0)+(useR?pv:0)+((dl || dr)?vec:0)+(useHL?vec:0)+(useHR?pv:0)+
        align8(2*rows)+(masked?align8(rows):0);
    const uint64_t output=(dl?vec:0)+(dr?pv:0)+(dg?vec:0);
    const uint64_t bytes=4*((hoist&32?1:2)*input+(hoist&16?1:2)*output+(3+bool(hoist&8))*vec+3*rc+align8(4*p));
    TORCH_CHECK(bytes+4096<=ub,"Fused adjoint exceeds UB capacity");
    TORCH_CHECK(meta.numel()>=4*p && ix.numel()>=(2+bool(hoist&8))*vec+2*rc,"Incomplete fused adjoint schedules");
    auto dlT=dl?at::empty({static_cast<int64_t>(n),q,c},l.options()):at::empty({0},l.options());
    auto drT=dr?at::empty({static_cast<int64_t>(n),p,c},l.options()):at::empty({0},l.options());
    auto dgT=dg?at::zeros({static_cast<int64_t>(nodes),m,c},l.options()):at::empty({0},l.options());
    if(!n)return {dlT,drT,dgT};
    const uint32_t blocks=std::min<uint64_t>(platform->GetCoreNumAiv(),((n+rows-1)/rows)*((c+bc-1)/bc));
    const uint64_t ls=l.stride(0),rs=r.stride(0),hls=hasHL?hl.stride(0):0,hrs=hasHR?hr.stride(0):0;
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto launch=[l,r,g,hl,hr,dst,mask,meta,ix,dlT,drT,dgT,stream,n,ls,rs,hls,hrs,c,q,p,m,second,rows,bc,masked,flags,hoist,blocks]() -> int {
        ACLRT_LAUNCH_KERNEL(ictc_epilogue_adjoint)(blocks,stream,
            reinterpret_cast<uint8_t*>(l.data_ptr()),reinterpret_cast<uint8_t*>(r.data_ptr()),reinterpret_cast<uint8_t*>(g.data_ptr()),
            reinterpret_cast<uint8_t*>(hl.data_ptr()),reinterpret_cast<uint8_t*>(hr.data_ptr()),reinterpret_cast<uint8_t*>(dst.data_ptr()),
            reinterpret_cast<uint8_t*>(mask.data_ptr()),reinterpret_cast<uint8_t*>(meta.data_ptr()),reinterpret_cast<uint8_t*>(ix.data_ptr()),
            reinterpret_cast<uint8_t*>(dlT.data_ptr()),reinterpret_cast<uint8_t*>(drT.data_ptr()),reinterpret_cast<uint8_t*>(dgT.data_ptr()),
            n,ls,rs,hls,hrs,c,q,p,m,second,rows,bc,masked,flags,hoist);return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ictc_ascendc_epilogue_adjoint",launch);
    return {dlT,drT,dgT};
}
}
TORCH_LIBRARY_FRAGMENT(ictc_ascendc,m) {
    m.def("epilogue_adjoint(Tensor left, Tensor right, Tensor grad, Tensor hleft, Tensor hright, Tensor destination, Tensor mask, Tensor metadata, Tensor indices, int[] shape) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(ictc_ascendc,PrivateUse1,m) {m.impl("epilogue_adjoint",TORCH_FN(epilogue_adjoint));}
