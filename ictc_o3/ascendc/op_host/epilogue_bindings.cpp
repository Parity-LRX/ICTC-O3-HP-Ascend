#include "kernel_launch.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
namespace {
uint64_t align8(uint64_t x) { return (x+7)/8*8; }
at::Tensor epilogue_impl(const at::Tensor& l,const at::Tensor& r,const at::Tensor& g,
    const at::Tensor& dst,const at::Tensor& mask,const at::Tensor& meta,const at::Tensor& ix,at::IntArrayRef s,const at::Tensor& target) {
    TORCH_CHECK((s.size()==12 || s.size()==13),"Invalid epilogue descriptor");
    const uint32_t hoist=s.size()==13?s[12]:0;TORCH_CHECK((hoist&~121u)==0,"Invalid transfer policy");
    const uint64_t n=s[0],nodes=s[10];
    const uint32_t c=s[1],q=s[2],p=s[3],m=s[4],paths=s[5],mode=s[6],rows=s[7],bc=s[8],masked=s[9];
    const uint32_t dense=s[11];
    TORCH_CHECK(l.device().type()==c10::DeviceType::PrivateUse1,"Expected NPU activations");
    for(const auto& value:{l,r,g})
        TORCH_CHECK(value.device()==l.device() && value.scalar_type()==at::kFloat,"Expected FP32 operands on one NPU");
    TORCH_CHECK(l.dim()==3 && l.size(0)==n && l.size(1)==q && l.size(2)==c &&
        l.stride(2)==1 && l.stride(1)==c && l.stride(0)>=static_cast<int64_t>(q)*c,"Invalid epilogue left layout");
    TORCH_CHECK(r.dim()==3 && r.size(0)==n && r.size(1)==p && r.size(2)==c &&
        r.stride(2)==1 && r.stride(1)==c && r.stride(0)>=static_cast<int64_t>(p)*c,"Invalid epilogue right layout");
    if(mode)TORCH_CHECK(g.dim()==3 && g.size(0)==nodes && g.size(1)==m && g.size(2)==c &&
        g.is_contiguous(),"Invalid node cotangent layout");
    TORCH_CHECK(dst.device()==l.device() && dst.scalar_type()==at::kLong && dst.is_contiguous() && dst.numel()==n,
        "Expected contiguous int64 destinations");
    for(const auto& value:{meta,ix})TORCH_CHECK(value.device()==l.device() && value.scalar_type()==at::kInt &&
        value.is_contiguous(),"Expected contiguous int32 epilogue schedules");
    if(masked)TORCH_CHECK(mask.device()==l.device() && mask.scalar_type()==at::kFloat && mask.is_contiguous() &&
        mask.numel()==n,"Invalid fixed epilogue mask");
    TORCH_CHECK(c && q && p && m && paths==p && mode<3 && rows && bc && bc%8==0 &&
        q<=4095 && p<=4095 && (!dense || m<=4095) && dense<=1 && rows<=4095 &&
        static_cast<uint64_t>(rows)*q*bc<=8192,"Invalid epilogue tile");
    c10_npu::NPUGuard guard(l.device());
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform,"CANN platform discovery failed");
    uint64_t ub=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    const uint64_t vec=static_cast<uint64_t>(rows)*q*bc,rc=static_cast<uint64_t>(rows)*bc;
    const uint64_t input=(mode==1 ? 0 : vec)+(mode==2 ? 0 : static_cast<uint64_t>(rows)*p*bc)+
        (mode==0 ? 0 : static_cast<uint64_t>(rows)*(dense ? m : q)*bc)+align8(2*rows)+(masked ? align8(rows) : 0);
    const uint64_t output=mode==2 ? static_cast<uint64_t>(rows)*p*bc : vec;
    const uint64_t bytes=4*((hoist&32?1:2)*input+(hoist&16?1:2)*output+(3+dense+bool(hoist&8))*vec+3*rc+align8(4*paths));
    TORCH_CHECK(bytes+4096<=ub,"Epilogue exceeds UB capacity");
    TORCH_CHECK(meta.numel()>=4*paths && ix.numel()>=(2+dense+bool(hoist&8))*vec+2*rc,"Incomplete epilogue metadata");
    at::Tensor out;
    if(target.defined()) {
        TORCH_CHECK(mode==0 && target.device()==l.device() && target.scalar_type()==at::kFloat &&
            target.is_contiguous() && target.dim()==3 && target.size(0)==nodes &&
            target.size(1)==m && target.size(2)==c,"Invalid epilogue accumulation buffer");
        for(const auto& value:{l,r,g,dst,mask,meta,ix})
            TORCH_CHECK(!target.is_alias_of(value),"Epilogue accumulation must not alias its operands");
        out=target;
    } else out=mode==0 ? at::zeros({static_cast<int64_t>(nodes),m,c},l.options()) :
        at::empty({static_cast<int64_t>(n),mode==1 ? q : p,c},l.options());
    if(!n)return out;
    const uint64_t tasks=((n+rows-1)/rows)*((c+bc-1)/bc);
    const uint32_t blocks=std::min<uint64_t>(platform->GetCoreNumAiv(),tasks);
    const uint64_t ls=l.stride(0),rs=r.stride(0);
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto launch=[l,r,g,dst,mask,meta,ix,out,stream,n,ls,rs,c,q,p,m,paths,mode,rows,bc,masked,dense,hoist,blocks]() -> int {
        ictc_ascendc::launch_epilogue(blocks,stream,
            reinterpret_cast<uint8_t*>(l.data_ptr()),reinterpret_cast<uint8_t*>(r.data_ptr()),
            reinterpret_cast<uint8_t*>(g.data_ptr()),reinterpret_cast<uint8_t*>(dst.data_ptr()),
            reinterpret_cast<uint8_t*>(mask.data_ptr()),reinterpret_cast<uint8_t*>(meta.data_ptr()),
            reinterpret_cast<uint8_t*>(ix.data_ptr()),reinterpret_cast<uint8_t*>(out.data_ptr()),
            n,ls,rs,c,q,p,m,paths,mode,rows,bc,masked,dense,hoist);return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ictc_ascendc_epilogue",launch);
    return out;
}
at::Tensor epilogue(const at::Tensor& l,const at::Tensor& r,const at::Tensor& g,
    const at::Tensor& dst,const at::Tensor& mask,const at::Tensor& meta,const at::Tensor& ix,at::IntArrayRef s) {
    return epilogue_impl(l,r,g,dst,mask,meta,ix,s,at::Tensor());
}
at::Tensor epilogue_into(const at::Tensor& l,const at::Tensor& r,const at::Tensor& g,
    const at::Tensor& dst,const at::Tensor& mask,const at::Tensor& meta,const at::Tensor& ix,
    const at::Tensor& out,at::IntArrayRef s) {
    return epilogue_impl(l,r,g,dst,mask,meta,ix,s,out);
}
}
TORCH_LIBRARY_FRAGMENT(ictc_ascendc,m) {
    m.def("epilogue(Tensor left, Tensor right, Tensor grad, Tensor destination, Tensor mask, Tensor metadata, Tensor indices, int[] shape) -> Tensor");
    m.def("epilogue_into(Tensor left, Tensor right, Tensor grad, Tensor destination, Tensor mask, Tensor metadata, Tensor indices, Tensor(a!) output, int[] shape) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(ictc_ascendc,PrivateUse1,m) {
    m.impl("epilogue",TORCH_FN(epilogue));
    m.impl("epilogue_into",TORCH_FN(epilogue_into));
}
