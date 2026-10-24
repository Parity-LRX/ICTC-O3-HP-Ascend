# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Edge-owned joint geometric VJP plus a source-owned charged-input VJP.

Each edge CTA reduces its entire harmonic/phase cotangent before writing it.
Unlike the source-owned joint schedule, no harmonic/phase atomic scatter is
needed. The source feature adjoint is separately CSR-owned, without atomics.
"""
from mace_ictc.models.generated_cartesian_message import _literal, _sum


def source(p, projection, columns, outputs):
    c, hdim = p["c"], projection["hdim"]
    heads, phases = p["phase_heads"], p["phase_count"]
    threads = 32 if c <= 32 else 128
    warps, tiles = threads//32, (c+31)//32
    cotangents = ",".join(f"const float* V{l}" for l, _, _ in outputs)

    def vloads(degree, base, width, ob, m):
        lines = []
        for mag in range(m):
            offset = mag*(width//m)+(ob-base)//m
            for stream, prefix in enumerate(("v", "vr", "vi")):
                lines.append(f"const float {prefix}{mag}=valid?V{degree}[(dst*3+{stream})*{width}+{offset}+channel]:0.f;")
            lines.append(f"const float b{mag}=v{mag}+cr*vr{mag}+ci*vi{mag};")
        return lines

    # One CTA owns all gate/harmonic/phase derivatives of an edge. It may
    # traverse more than one channel tile, but shared reduction stays fixed.
    lines = ["""
__global__ void coherent_edge_adjoint_kernel(const float* X,const float* H,const float* W,
 const float* G,const float* R,const float* I,const int64_t* SRC,const int64_t* DST,
 const float* MASK,"""+cotangents+""",float* DH,float* DG,float* DR,float* DI,
 int64_t E,bool masked,bool needh,bool needg,bool needr,bool needi) {
 const int64_t e=blockIdx.x,src=SRC[e],dst=DST[e];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const float mask=masked?MASK[e]:1.f;
"""]
    for h in range(hdim):
        lines += [f"const float h{h}=H[e*{hdim}+{h}];", f"float dh{h}=0.f;"]
    for phase in range(phases):
        lines += [f"float dr{phase}=0.f,di{phase}=0.f;"]
    lines += [f"for(int channel=threadIdx.x;channel<{c};channel+={threads}){{",
              "const bool valid=true;", f"const int head=channel/{c//heads};"]
    grouped = {}
    for row in p["path_specs"]:
        grouped.setdefault((row[2], row[1]), []).append(row)
    for (xb, ni), paths in grouped.items():
        lines.append("{")
        for j in range(ni):
            lines.append(f"const float x{j}=X[src*{p['xdim']}+{xb+j*c}+channel];")
        for path, _, _, bb, bs, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            phase = degree*heads if p["phase_angular"] else 0
            lines += ["{", f"const int phase={phase}+head;",
                      f"const float cr=R[e*{phases}+phase],ci=I[e*{phases}+phase];",
                      f"const float gate=G[e*{p['p']*c}+{path*c}+channel];",
                      f"const float scale=W[{path*c}+channel]*mask;"]
            lines += vloads(degree, base, width, ob, m)
            for j in range(ni):
                for mag in range(m):
                    terms = columns[bb+j*bs+mag]
                    if terms:
                        lines.append(f"const float q{j}_{mag}="+_sum(f"{_literal(v)}*h{h}" for h, v in terms)+";")
            for mag in range(m):
                terms = [f"x{j}*q{j}_{mag}" for j in range(ni) if columns[bb+j*bs+mag]]
                lines.append(f"const float t{mag}="+_sum(terms)+";")
            value = _sum(f"b{mag}*t{mag}" for mag in range(m))
            lines.append(f"if(needg)DG[e*{p['p']*c}+{path*c}+channel]=({value})*scale;")
            for head in range(heads):
                dr = _sum(f"vr{mag}*t{mag}" for mag in range(m))
                di = _sum(f"vi{mag}*t{mag}" for mag in range(m))
                lines += [f"if(head=={head}){{dr{phase+head}+=({dr})*gate*scale;",
                          f"di{phase+head}+=({di})*gate*scale;}}"]
            lines.append("if(needh){")
            for j in range(ni):
                for mag in range(m):
                    terms = columns[bb+j*bs+mag]
                    if terms:
                        lines.append(f"const float db{j}_{mag}=x{j}*b{mag}*gate*scale;")
                        for h, value in terms:
                            lines.append(f"dh{h}+=({_literal(value)})*db{j}_{mag};")
            lines += ["}", "}"]
        lines.append("}")
    lines.append("}")
    slots = hdim+2*phases
    lines.append(f"__shared__ float partial[{slots*warps}];")
    reductions = [(f"dh{h}", "DH", h, hdim, "needh") for h in range(hdim)]
    reductions += [(f"dr{i}", "DR", i, phases, "needr") for i in range(phases)]
    reductions += [(f"di{i}", "DI", i, phases, "needi") for i in range(phases)]
    for slot, (value, _, _, _, need) in enumerate(reductions):
        lines += [f"if({need}){{",
                  f"for(int d=16;d>0;d/=2){value}+=__shfl_down_sync(0xffffffff,{value},d);",
                  f"if(lane==0)partial[{slot*warps}+warp]={value};", "}"]
    lines.append("__syncthreads();")
    for slot, (_, target, offset, stride, need) in enumerate(reductions):
        value = _sum(f"partial[{slot*warps+warp}]" for warp in range(warps))
        lines.append(f"if(threadIdx.x=={slot%threads}&&{need}){target}[e*{stride}+{offset}]={value};")
    lines += ["}", ""]

    # Node-feature gradients are independent of X itself. The separate kernel
    # reads only H, gates, phases and the joint output cotangents.
    lines += ["""
__global__ void coherent_source_adjoint_kernel(const float* H,const float* W,const float* G,
 const float* R,const float* I,const int64_t* DST,const float* MASK,
 """+cotangents+""",const int64_t* PTR,const int64_t* ORDER,float* DX,
 int64_t N,bool masked) {
 const int lane=threadIdx.x%32;
 const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
""", f"if(task>=N*{tiles*len(grouped)})return;",
              f"const int channel=(task%{tiles})*32+lane;if(channel>={c})return;",
              "const bool valid=true;", f"const int head=channel/{c//heads};",
              f"const int group=(task/{tiles})%{len(grouped)};",
              f"const int64_t src=task/{tiles*len(grouped)};"]
    for number, ((xb, ni), paths) in enumerate(grouped.items()):
        lines.append(f"if(group=={number}){{")
        for j in range(ni):
            lines.append(f"float dx{j}=0.f;")
        lines += ["for(int64_t index=PTR[src];index<PTR[src+1];++index){",
                  "const int64_t e=ORDER[index],dst=DST[e];const float mask=masked?MASK[e]:1.f;"]
        for h in range(hdim):
            lines.append(f"const float h{h}=H[e*{hdim}+{h}];")
        for path, _, _, bb, bs, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            phase = degree*heads if p["phase_angular"] else 0
            lines += ["{", f"const int phase={phase}+head;",
                      f"const float cr=R[e*{phases}+phase],ci=I[e*{phases}+phase];",
                      f"const float gate=G[e*{p['p']*c}+{path*c}+channel]*W[{path*c}+channel]*mask;"]
            lines += vloads(degree, base, width, ob, m)
            for j in range(ni):
                terms = [f"b{mag}*("+_sum(f"{_literal(v)}*h{h}" for h, v in columns[bb+j*bs+mag])+")"
                         for mag in range(m) if columns[bb+j*bs+mag]]
                lines.append(f"dx{j}+=gate*("+_sum(terms)+");")
            lines.append("}")
        lines.append("}")
        for j in range(ni):
            lines.append(f"DX[src*{p['xdim']}+{xb+j*c}+channel]=dx{j};")
        lines.append("}")
    lines.append("}")
    wrapper_args = ",".join(f"torch::Tensor v{l}" for l, _, _ in outputs)
    pointer_args = ",".join(f"v{l}.data_ptr<float>()" for l, _, _ in outputs)
    lines.append(f"""
void coherent_adjoint(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor dst,torch::Tensor mask,{wrapper_args},
 torch::Tensor ptr,torch::Tensor order,torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,
 torch::Tensor dr,torch::Tensor di,bool needx,bool needh,bool needg,bool needr,bool needi) {{
 const c10::cuda::CUDAGuard guard(x.device());
 auto stream=at::cuda::getCurrentCUDAStream();
 if(src.numel()&&(needh||needg||needr||needi)){{
 coherent_edge_adjoint_kernel<<<src.numel(),{threads},0,stream>>>(x.data_ptr<float>(),h.data_ptr<float>(),
 w.data_ptr<float>(),g.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),
 src.data_ptr<int64_t>(),dst.data_ptr<int64_t>(),mask.data_ptr<float>(),{pointer_args},
 dh.data_ptr<float>(),dg.data_ptr<float>(),dr.data_ptr<float>(),di.data_ptr<float>(),
 src.numel(),mask.numel()!=0,needh,needg,needr,needi);
 C10_CUDA_KERNEL_LAUNCH_CHECK();}}
 if(needx&&x.size(0)){{
 const int64_t blocks=(x.size(0)*{tiles*len(grouped)}+3)/4;
 coherent_source_adjoint_kernel<<<blocks,128,0,stream>>>(h.data_ptr<float>(),w.data_ptr<float>(),
 g.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),dst.data_ptr<int64_t>(),
 mask.data_ptr<float>(),{pointer_args},ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),
 dx.data_ptr<float>(),x.size(0),mask.numel()!=0);
 C10_CUDA_KERNEL_LAUNCH_CHECK();}}
}}
""")
    return "\n".join(lines)
