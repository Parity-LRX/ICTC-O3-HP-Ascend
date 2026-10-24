# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Destination-owned live JVP for the output-cotangent mixed derivative.

Each warp owns a node/channel/path tile and writes its complete three-stream
result. This trades some repeated TP work for removal of E-by-output floating
atomics. Only dependency metadata is specialized; N/E and all values stay live.
"""
from mace_ictc.models.generated_cartesian_message import _literal, _sum


_EXTENSIONS = {}


def _source(p, projection, tangents):
    from .coherent import _HEADERS, _columns, _outputs
    c, hd, phases = p["c"], projection["hdim"], p["phase_count"]
    columns, outputs = _columns(p, projection), _outputs(p)
    groups, current, width = [], [], 0
    for row in p["path_specs"]:
        if current and width+row[-1] > 8:
            groups.append(current)
            current, width = [], 0
        current.append(row)
        width += row[-1]
    if current:
        groups.append(current)
    tiles = (c+31)//32
    outargs = ",".join(f"float* O{l}" for l, _, _ in outputs)
    flags = ("hx", "hh", "hg", "hr", "hi")
    constants = ";".join(f"constexpr bool {flag}={'true' if active else 'false'}"
                         for flag, active in zip(flags, tangents))+";"
    lines = [_HEADERS, f"""
__global__ void coherent_destination_jvp_kernel(const float* X,const float* H,const float* W,
 const float* G,const float* R,const float* I,const int64_t* SRC,const float* MASK,
 const int64_t* PTR,const int64_t* ORDER,const float* TX,const float* TH,const float* TG,
 const float* TR,const float* TI,{outargs},int64_t N,bool masked) {{
 {constants}
 const int lane=threadIdx.x%32;
 const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
 if(task>=N*{tiles*len(groups)})return;
 const int channel=(task%{tiles})*32+lane;
 if(channel>={c})return;
 const int group=(task/{tiles})%{len(groups)};
 const int64_t dst=task/{tiles*len(groups)};
 const int head=channel/{c//p['phase_heads']};
"""]
    for group, paths in enumerate(groups):
        lines.append(f"if(group=={group}){{")
        for path, _, _, _, _, _, m in paths:
            lines.append(f"const float w{path}=W[{path*c}+channel];")
            for mag in range(m):
                lines.append(f"float a{path}_{mag}=0.f,r{path}_{mag}=0.f,i{path}_{mag}=0.f;")
        lines += ["for(int64_t index=PTR[dst];index<PTR[dst+1];++index){",
                  "const int64_t e=ORDER[index],src=SRC[e];const float mask=masked?MASK[e]:1.f;"]
        for h in range(hd):
            lines += [f"const float h{h}=H[e*{hd}+{h}];",
                      f"const float th{h}=hh?TH[e*{hd}+{h}]:0.f;"]
        grouped = {}
        for row in paths:
            grouped.setdefault((row[2], row[1]), []).append(row)
        for (xb, ni), input_paths in grouped.items():
            lines.append("{")
            for j in range(ni):
                lines += [f"const float x{j}=X[src*{p['xdim']}+{xb+j*c}+channel];",
                          f"const float tx{j}=hx?TX[src*{p['xdim']}+{xb+j*c}+channel]:0.f;"]
            for path, _, _, bb, bs, _, m in input_paths:
                phase = (m-1)//2*p["phase_heads"] if p["phase_angular"] else 0
                lines += ["{", f"const int phase={phase}+head;",
                          f"const float cr=R[e*{phases}+phase],ci=I[e*{phases}+phase];",
                          f"const float tr=hr?TR[e*{phases}+phase]:0.f,ti=hi?TI[e*{phases}+phase]:0.f;",
                          f"const float gate=G[e*{p['p']*c}+{path*c}+channel];",
                          f"const float tg=hg?TG[e*{p['p']*c}+{path*c}+channel]:0.f;",
                          f"const float scale=w{path}*mask;"]
                for mag in range(m):
                    for j in range(ni):
                        terms = columns[bb+j*bs+mag]
                        if terms:
                            lines += [f"const float b{j}_{mag}="+_sum(f"{_literal(v)}*h{h}" for h, v in terms)+";",
                                      f"const float tb{j}_{mag}="+_sum(f"{_literal(v)}*th{h}" for h, v in terms)+";"]
                    active = [j for j in range(ni) if columns[bb+j*bs+mag]]
                    lines += [f"const float u{mag}="+_sum(f"x{j}*b{j}_{mag}" for j in active)+";",
                              f"const float tu{mag}="+_sum(f"tx{j}*b{j}_{mag}+x{j}*tb{j}_{mag}" for j in active)+";",
                              f"const float v{mag}=gate*tu{mag}+tg*u{mag};",
                              f"a{path}_{mag}+=scale*v{mag};",
                              f"r{path}_{mag}+=scale*(cr*v{mag}+tr*gate*u{mag});",
                              f"i{path}_{mag}+=scale*(ci*v{mag}+ti*gate*u{mag});"]
                lines.append("}")
            lines.append("}")
        lines.append("}")
        for path, _, _, _, _, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            for mag in range(m):
                offset = mag*(width//m)+(ob-base)//m
                for stream, prefix in enumerate(("a", "r", "i")):
                    lines.append(f"O{degree}[(dst*3+{stream})*{width}+{offset}+channel]={prefix}{path}_{mag};")
        lines.append("}")
    lines.append("}")
    wrappers = ",".join(f"torch::Tensor out{l}" for l, _, _ in outputs)
    pointers = ",".join(f"out{l}.data_ptr<float>()" for l, _, _ in outputs)
    lines.append(f"""
void coherent_destination_jvp(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor mask,
 torch::Tensor ptr,torch::Tensor order,torch::Tensor tx,torch::Tensor th,torch::Tensor tg,
 torch::Tensor tr,torch::Tensor ti,{wrappers}) {{
 const c10::cuda::CUDAGuard guard(x.device());
 const int64_t blocks=(x.size(0)*{tiles*len(groups)}+3)/4;
 if(!blocks)return;
 auto stream=at::cuda::getCurrentCUDAStream();
 coherent_destination_jvp_kernel<<<blocks,128,0,stream>>>(x.data_ptr<float>(),h.data_ptr<float>(),
 w.data_ptr<float>(),g.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),
 src.data_ptr<int64_t>(),mask.data_ptr<float>(),ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),
 tx.data_ptr<float>(),th.data_ptr<float>(),tg.data_ptr<float>(),tr.data_ptr<float>(),ti.data_ptr<float>(),
 {pointers},x.size(0),mask.numel()!=0);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
""")
    return "\n".join(lines)


def destination_jvp(x, sh, w, g, r, i, src, dst, mask, p, projection, tangents, active, out):
    from torch.utils.cpp_extension import load_inline
    from .coherent import _outputs
    from mace_ictc.models.message_topology import topology
    key = (p["path_specs"], p["xdim"], p["odim"], p["c"], p["phase_heads"], p["phase_angular"],
           tuple(tuple(row) for row in projection["static_rows"]), active)
    if key not in _EXTENSIONS:
        name = (f"ictc_coherent_training_jvp_v1_{p['c']}_{p['xdim']}_{p['odim']}"
                f"_h{p['phase_heads']}_a{int(p['phase_angular'])}_dep"
                +"".join("1" if need else "0" for need in active))
        declaration = "#include <torch/extension.h>\nvoid coherent_destination_jvp("+",".join(
            ["torch::Tensor"]*(15+len(_outputs(p))))+");\n"
        _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration,
            cuda_sources=_source(p, projection, active), functions=["coherent_destination_jvp"],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)
    _EXTENSIONS[key].coherent_destination_jvp(x, sh, w, g, r, i, src, mask,
        *topology(dst, x.shape[0]), *tangents, *out)
