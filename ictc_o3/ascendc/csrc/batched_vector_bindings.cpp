#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_ictc_batched_vector.h"
namespace {
uint64_t A8(uint64_t n) {return (n+7)/8*8;}
at::Tensor batched_vector(const at::Tensor& a,const at::Tensor& b,const at::Tensor& ix,at::IntArrayRef s) {
    TORCH_CHECK(s.size()==7,"Invalid Vector BMM descriptor");
    for(auto v:s)TORCH_CHECK(v>=0,"Negative Vector BMM dimension");
    const uint64_t n=s[0];const uint32_t q=s[1],i=s[2],c=s[3],mode=s[4],rows=s[5],bc=s[6];
    TORCH_CHECK(q && i && c && mode<=2 && rows && bc && bc<=64 && bc%8==0 && rows*q<=255 &&
        static_cast<uint64_t>(rows)*q*bc<=8192 && static_cast<uint64_t>(rows)*q*i<=8192,"Invalid Vector BMM tile");
    TORCH_CHECK(a.device().type()==c10::DeviceType::PrivateUse1 && a.scalar_type()==at::kFloat && b.device()==a.device() &&
        b.scalar_type()==at::kFloat && ix.device()==a.device() && ix.scalar_type()==at::kInt && ix.is_contiguous(),"Invalid Vector BMM tensors");
    const uint64_t aq=mode==2?i:q,ac=mode==1?c:(mode==2?q:i);
    TORCH_CHECK(a.dim()==3 && a.size(0)==n && a.size(1)==aq && a.size(2)==ac &&
        a.stride(2)==1 && a.stride(1)==ac && a.stride(0)>=aq*ac,"Invalid Vector BMM left layout");
    TORCH_CHECK(b.dim()==3 && b.size(0)==n && b.size(1)==i && b.size(2)==c &&
        b.stride(2)==1 && b.stride(1)==c && b.stride(0)>=static_cast<uint64_t>(i)*c,"Invalid Vector BMM right layout");
    c10_npu::NPUGuard guard(a.device());auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform,"CANN platform discovery failed");uint64_t ub=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    const uint64_t vec=static_cast<uint64_t>(rows)*q*bc;
    const uint64_t input=(mode==1?vec:rows*A8(q*i))+static_cast<uint64_t>(rows)*i*bc;
    const uint64_t output=mode==1?A8(static_cast<uint64_t>(rows)*q*i):vec;
    const uint64_t bytes=4*(2*input+2*output+4*vec+i*A8(rows*q)+A8(static_cast<uint64_t>(rows)*q*i));
    TORCH_CHECK(bytes+4096<=ub && ix.numel()>=2*vec+A8(static_cast<uint64_t>(rows)*q*i),"Incomplete or excessive Vector BMM UB schedule");
    auto out=mode==1 ? (c>bc ? at::zeros({static_cast<int64_t>(n),q,i},a.options()):at::empty({static_cast<int64_t>(n),q,i},a.options())) :
        at::empty({static_cast<int64_t>(n),q,c},a.options());
    if(!n)return out;
    const auto stream=c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t blocks=std::min<uint64_t>(platform->GetCoreNumAiv(),((n+rows-1)/rows)*((c+bc-1)/bc));
    const uint64_t as=a.stride(0),bs=b.stride(0);
    auto launch=[a,b,ix,out,stream,n,as,bs,q,i,c,mode,rows,bc,blocks]() -> int {
        ACLRT_LAUNCH_KERNEL(ictc_batched_vector)(blocks,stream,
            reinterpret_cast<uint8_t*>(a.data_ptr()),reinterpret_cast<uint8_t*>(b.data_ptr()),
            reinterpret_cast<uint8_t*>(out.data_ptr()),reinterpret_cast<uint8_t*>(ix.data_ptr()),n,as,bs,q,i,c,mode,rows,bc);return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ictc_ascendc_batched_vector",launch);return out;
}
}
TORCH_LIBRARY_FRAGMENT(ictc_ascendc,m) {m.def("batched_vector(Tensor a, Tensor b, Tensor indices, int[] shape) -> Tensor");}
TORCH_LIBRARY_IMPL(ictc_ascendc,PrivateUse1,m) {m.impl("batched_vector",TORCH_FN(batched_vector));}
