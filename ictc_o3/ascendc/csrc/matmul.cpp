#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;
// ND2NZ row strides above 65535 elements require the SDK's cyclic move path.
constexpr MatmulConfig ICTC_FP32_CONFIG=GetNormalConfig(true);

template<bool transposeA,bool transposeB>
__aicore__ inline void Calculate(GM_ADDR a,GM_ADDR b,GM_ADDR bias,GM_ADDR c,
                                 GM_ADDR workspace,TCubeTiling& tiling,uint32_t hasBias) {
    if(GetBlockIdx()>=tiling.usedCoreNum) return;
    const uint32_t mBlocks=(tiling.M+tiling.singleCoreM-1)/tiling.singleCoreM;
    const uint64_t row=static_cast<uint64_t>(GetBlockIdx()%mBlocks)*tiling.singleCoreM;
    const uint64_t column=static_cast<uint64_t>(GetBlockIdx()/mBlocks)*tiling.singleCoreN;
    if(row>=tiling.M || column>=tiling.N) return;
    GlobalTensor<float> ag,bg,cg,biasg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    biasg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    // For FP32 the template, host tiling and runtime transpose flags must
    // match exactly; the half-precision runtime-switching rule does not apply.
    using A=MatmulType<TPosition::GM,CubeFormat::ND,float,transposeA>;
    using B=MatmulType<TPosition::GM,CubeFormat::ND,float,transposeB>;
    using C=MatmulType<TPosition::GM,CubeFormat::ND,float>;
    using Bias=MatmulType<TPosition::GM,CubeFormat::ND,float>;
    TPipe pipe;
    Matmul<A,B,C,Bias,ICTC_FP32_CONFIG> mm;
    REGIST_MATMUL_OBJ(&pipe,workspace,mm,&tiling);
    mm.SetTensorA(ag[transposeA ? row : row*tiling.Ka],transposeA);
    mm.SetTensorB(bg[transposeB ? column*tiling.Kb : column],transposeB);
    const uint32_t tailM=tiling.M-row<tiling.singleCoreM ? tiling.M-row : tiling.singleCoreM;
    const uint32_t tailN=tiling.N-column<tiling.singleCoreN ? tiling.N-column : tiling.singleCoreN;
    mm.SetTail(tailM,tailN,tiling.Ka);
    if(hasBias) mm.SetBias(biasg[column]);
    mm.SetHF32(false);
    mm.IterateAll(cg[row*tiling.N+column]);
    mm.End();
}

extern "C" __global__ __aicore__ void ictc_matmul(
    GM_ADDR a,GM_ADDR b,GM_ADDR bias,GM_ADDR c,GM_ADDR workspace,GM_ADDR metadata,
    uint32_t transposeA,uint32_t transposeB,uint32_t hasBias) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if(g_coreType==AscendC::AIV) return;
    TCubeTiling tiling;
    auto* target=reinterpret_cast<uint32_t*>(&tiling);
    auto* source=reinterpret_cast<__gm__ uint32_t*>(metadata);
    for(uint32_t i=0;i<sizeof(TCubeTiling)/sizeof(uint32_t);i++) target[i]=source[i];
    if(transposeA) {
        if(transposeB) Calculate<true,true>(a,b,bias,c,workspace,tiling,hasBias);
        else Calculate<true,false>(a,b,bias,c,workspace,tiling,hasBias);
    } else {
        if(transposeB) Calculate<false,true>(a,b,bias,c,workspace,tiling,hasBias);
        else Calculate<false,false>(a,b,bias,c,workspace,tiling,hasBias);
    }
}
