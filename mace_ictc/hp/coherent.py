# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Native three-stream ICTC graph TP and joint coordinate adjoint.

One Cartesian TP value feeds the neutral, real and imaginary accumulators.
Only the original E*P*C gate bank is stored; coefficients remain E*phase.
No cuEq/OpenEq code, descriptors or runtime is used. A joint mixed adjoint
supports force training; third/higher derivatives retain the exact expression.
"""
import torch
from .operators import EdgeTensorProduct, _eager, _operand
from mace_ictc.models.generated_cartesian_message import _literal, _sum


_EXTENSIONS = {}
_HEADERS = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
"""


def _columns(p, projection):
    columns = [[] for _ in range(p["bdim"])]
    for harmonic, rows in enumerate(projection["static_rows"]):
        for col, value in rows:
            columns[col].append((harmonic, value))
    return columns


def _outputs(p):
    """(degree, flat offset, width), preserving the original path order."""
    counts = {}
    for row in p["path_specs"]:
        degree = (row[-1]-1)//2
        counts[degree] = counts.get(degree, 0)+1
    offset, specs = 0, []
    for degree, count in sorted(counts.items()):
        width = count*p["c"]*(2*degree+1)
        specs.append((degree, offset, width))
        offset += width
    return tuple(specs)


def _forward_source(p, projection, *, accumulator_limit=16, neighbor_warps=1):
    c, hdim, odim = p["c"], projection["hdim"], p["odim"]
    tiles = (c+31)//32
    groups, current, width = [], [], 0
    for row in p["path_specs"]:
        if current and width+row[-1] > accumulator_limit:
            groups.append(current)
            current, width = [], 0
        current.append(row)
        width += row[-1]
    if current:
        groups.append(current)
    columns = _columns(p, projection)
    outputs = _outputs(p)
    output_args = ",".join(f"float* O{l}" for l, _, _ in outputs)
    lines = [_HEADERS, """
__global__ void coherent_forward_kernel(const float* X,const float* H,const float* W,
 const float* G,const float* R,const float* I,const int64_t* SRC,const float* MASK,
 const int64_t* PTR,const int64_t* ORDER,"""+output_args+""",int64_t N,int64_t E,bool masked) {
 const int lane=threadIdx.x%32;
""", ('const int64_t task=blockIdx.x;const int shard=threadIdx.x/32;' if neighbor_warps > 1 else
        'const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;'),
        f"if(task>=N*{tiles*len(groups)})return;",
        f"const int channel=(task%{tiles})*32+lane;",
        f"const int group=(task/{tiles})%{len(groups)};",
        f"const int64_t dst=task/{tiles*len(groups)};",
        f"if(channel>={c})return;const int head=channel/{c//p['phase_heads']};"]
    shared = 3*max(sum(row[-1] for row in paths) for paths in groups)
    if neighbor_warps > 1:lines.append(f'__shared__ float partial[{neighbor_warps*shared*32}];')
    for number, paths in enumerate(groups):
        lines.append(f"if(group=={number}){{")
        for path, ni, xb, bb, bs, ob, m in paths:
            lines.append(f"const float w{path}=W[{path*c}+channel];")
            for mag in range(m):
                lines.append(f"float a{path}_{mag}=0.f,r{path}_{mag}=0.f,i{path}_{mag}=0.f;")
        lines += [f"for(int64_t index=PTR[dst]"+('+shard' if neighbor_warps > 1 else '')+
                  ';index<PTR[dst+1];'+(f'index+={neighbor_warps}' if neighbor_warps > 1 else '++index')+'){' ,
                  "const int64_t e=ORDER[index],src=SRC[e];const float mask=masked?MASK[e]:1.f;"]
        for harmonic in range(hdim):
            lines.append(f"const float h{harmonic}=H[e*{hdim}+{harmonic}];")
        by_input = {}
        for row in paths:
            by_input.setdefault((row[2], row[1]), []).append(row)
        for (xb, ni), input_paths in by_input.items():
            lines.append("{")
            for j in range(ni):
                lines.append(f"const float x{j}=X[src*{p['xdim']}+{xb+j*c}+channel];")
            for path, _, _, bb, bs, ob, m in input_paths:
                phase = ((m-1)//2)*p["phase_heads"] if p["phase_angular"] else 0
                lines += ["{", f"const int phase={phase}+head;",
                          f"const float cr=R[e*{p['phase_count']}+phase],ci=I[e*{p['phase_count']}+phase];",
                          f"const float gate=G[e*{p['p']*c}+{path*c}+channel]*w{path}*mask;"]
                for mag in range(m):
                    terms = [f"x{j}*("+_sum(f"{_literal(v)}*h{h}" for h, v in columns[bb+j*bs+mag])+")"
                             for j in range(ni) if columns[bb+j*bs+mag]]
                    lines += [f"const float v{mag}=("+_sum(terms)+")*gate;",
                              f"a{path}_{mag}+=v{mag};r{path}_{mag}+=v{mag}*cr;i{path}_{mag}+=v{mag}*ci;"]
                lines.append("}")
            lines.append("}")
        lines.append("}")
        if neighbor_warps > 1:
            slot = 0
            for path, ni, xb, bb, bs, ob, m in paths:
                for mag in range(m):
                    for prefix in ('a', 'r', 'i'):
                        lines.append(f'partial[(shard*{shared}+{slot})*32+lane]={prefix}{path}_{mag};')
                        slot += 1
            lines += ['__syncthreads();', 'if(shard==0){']
        slot = 0
        for path, ni, xb, bb, bs, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            for mag in range(m):
                # Public [N,3,path*C,m] is an angular-major strided view.
                # Keeping channels adjacent both coalesces the warp stores and
                # lets the following shared channel GEMM consume it directly.
                offset = mag*(width//m)+(ob-base)//m
                for stream, prefix in enumerate(("a", "r", "i")):
                    value = (_sum(f'partial[({s}*{shared}+{slot})*32+lane]' for s in range(neighbor_warps))
                             if neighbor_warps > 1 else f'{prefix}{path}_{mag}')
                    lines.append(f"O{degree}[(dst*3+{stream})*{width}+{offset}+channel]={value};")
                    slot += 1
        if neighbor_warps > 1:lines.append('}')
        lines.append("}")
    lines.append("}")
    wrapper_args = ",".join(f"torch::Tensor out{l}" for l, _, _ in outputs)
    pointer_args = ",".join(f"out{l}.data_ptr<float>()" for l, _, _ in outputs)
    lines.append(f"""
void coherent_forward(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor mask,
 torch::Tensor ptr,torch::Tensor order,{wrapper_args}) {{
 const c10::cuda::CUDAGuard guard(x.device());
 auto stream=at::cuda::getCurrentCUDAStream();
 const int64_t blocks={f'x.size(0)*{tiles*len(groups)}' if neighbor_warps > 1 else f'(x.size(0)*{tiles*len(groups)}+3)/4'};
 if(!blocks)return;
 coherent_forward_kernel<<<blocks,{neighbor_warps*32 if neighbor_warps > 1 else 128},0,stream>>>(x.data_ptr<float>(),h.data_ptr<float>(),
 w.data_ptr<float>(),g.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),
 src.data_ptr<int64_t>(),mask.data_ptr<float>(),ptr.data_ptr<int64_t>(),
 order.data_ptr<int64_t>(),{pointer_args},x.size(0),src.numel(),mask.numel()!=0);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
""")
    return "\n".join(lines)


def _adjoint_groups(p):
    groups = {}
    for row in p['path_specs']:
        groups.setdefault((row[2], row[1]), []).append(row)
    limit = p.get('destination_group_paths', 0)
    return [(key, paths[start:start+limit] if limit else paths)
            for key, paths in groups.items()
            for start in (range(0, len(paths), limit) if limit else (0,))]


def _componentwise_first_path(p, columns, ni, bb, bs, ob, m, outputs):
    """Consume each original CG column before constructing the next column.

    This is the first VJP itself, not the mixed kernel with dummy tangents.
    Only scalar gate/phase accumulators survive across output components.
    """
    c, path = p['c'], p['_first_path']
    degree = (m-1)//2
    _, base, width = next(row for row in outputs if row[0] == degree)
    phase = degree*p['phase_heads'] if p['phase_angular'] else 0
    lines = ['float gate_grad=0.f,real_grad=0.f,imag_grad=0.f;']
    for mag in range(m):
        active = [j for j in range(ni) if columns[bb+j*bs+mag]]
        offset = mag*(width//m)+(ob-base)//m
        lines.append('{')
        for j in active:
            lines.append(f'const float basis{j}='+_sum(
                f'{_literal(value)}*h{h}' for h, value in columns[bb+j*bs+mag])+';')
        for prefix in ('v', 'vr', 'vi'):
            lines.append(f'const float {prefix}={prefix}p{path}_{mag};')
        lines += ['const float cotangent=v+cr*vr+ci*vi;',
                  'const float product='+_sum(f'x{j}*basis{j}' for j in active)+';',
                  'gate_grad+=product*cotangent;real_grad+=product*vr;imag_grad+=product*vi;',
                  'const float weighted=gate_scale*cotangent;']
        for j in active:
            lines.append(f'if(needx)dx{j}+=weighted*basis{j};')
            lines.append(f'if(needh){{const float basis_grad=x{j}*weighted;')
            for h, value in columns[bb+j*bs+mag]:
                lines.append(f'dh{h}+=({_literal(value)})*basis_grad;')
            lines.append('}')
        lines.append('}')
    lines += [f"if(needg&&valid)DG[e*{p['p']*c}+{path*c}+channel]=scale*gate_grad;",
              f"pr{phase//p['phase_heads']}+=gate_scale*real_grad;pi{phase//p['phase_heads']}+=gate_scale*imag_grad;"]
    return lines


def _csr_adjoint_source(p, projection, *, owned_edges=False, destination_owned=False):
    if destination_owned and not owned_edges:
        raise ValueError('Destination scheduling requires owned edge partials')
    c, hdim, odim = p["c"], projection["hdim"], p["odim"]
    factored = bool(p.get('factor_edge_scalars', False))
    coalesced = bool(p.get('coalesced_edge_partials', False))
    tiles = (c+31)//32
    columns = _columns(p, projection)
    groups = _adjoint_groups(p if destination_owned else {**p, 'destination_group_paths': 0})
    outputs = _outputs(p)
    cotangent_args = ",".join(f"const float* V{l}" for l, _, _ in outputs)
    lines = ["""
__global__ void coherent_adjoint_kernel(const float* X,const float* H,const float* W,
 const float* G,const float* R,const float* I,const int64_t* DST,const float* MASK,
 """+cotangent_args+""",const int64_t* PTR,const int64_t* ORDER,float* DX,float* DH,
 float* DG,float* DR,float* DI,int64_t N,int64_t E,bool masked,
 bool needx,bool needh,bool needg,bool needr,bool needi) {
 const int lane=threadIdx.x%32;
 const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
""", f"if(task>=N*{tiles*len(groups)})return;",
        f"const int channel=(task%{tiles})*32+lane;const bool valid=channel<{c};",
        f"const int group=(task/{tiles})%{len(groups)};",
        f"const int64_t {'dst' if destination_owned else 'src'}=task/{tiles*len(groups)};",
        f"const int head=channel/{c//p['phase_heads']};"]
    for number, ((xb, ni), paths) in enumerate(groups):
        lines.append(f"if(group=={number}){{")
        if not destination_owned:
            for j in range(ni):
                lines += [f"const float x{j}=valid?X[src*{p['xdim']}+{xb+j*c}+channel]:0.f;",
                          f"float dx{j}=0.f;"]
        for path, *_ in paths:
            lines.append(f"const float w{path}=valid?W[{path*c}+channel]:0.f;")
        if destination_owned:
            # Every incident edge uses this same destination cotangent. Load it
            # once, instead of rereading the wide three-stream bank per edge.
            for path, _, _, bb, bs, ob, m in paths:
                degree = (m-1)//2
                _, base, width = next(row for row in outputs if row[0] == degree)
                for mag in range(m):
                    offset = mag*(width//m)+(ob-base)//m
                    for stream, prefix in enumerate(('v', 'vr', 'vi')):
                        lines.append(f"const float {prefix}p{path}_{mag}=valid?V{degree}[(dst*3+{stream})*{width}+{offset}+channel]:0.f;")
            lines += ['for(int64_t index=PTR[dst];index<PTR[dst+1];++index){',
                      'const int64_t e=ORDER[index],src=DST[e];const float mask=masked?MASK[e]:1.f;']
            for j in range(ni):
                lines += [f"const float x{j}=valid?X[src*{p['xdim']}+{xb+j*c}+channel]:0.f;", f'float dx{j}=0.f;']
        else:
            lines += ["for(int64_t index=PTR[src];index<PTR[src+1];++index){",
                      "const int64_t e=ORDER[index],dst=DST[e];const float mask=masked?MASK[e]:1.f;"]
        for h in range(hdim):
            lines += [f"const float h{h}=H[e*{hdim}+{h}];", f"float dh{h}=0.f;"]
        if owned_edges:
            phase_bases = range((p['phase_count']//p['phase_heads']) if p['phase_angular'] else 1)
            for q in phase_bases:
                lines.append(f"float pr{q}=0.f,pi{q}=0.f;")
        for path, _, _, bb, bs, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            phase = ((m-1)//2)*p["phase_heads"] if p["phase_angular"] else 0
            lines += ["{", f"const int phase={phase}+head;",
                      f"const float cr=valid?R[e*{p['phase_count']}+phase]:0.f;",
                      f"const float ci=valid?I[e*{p['phase_count']}+phase]:0.f;",
                      f"const float gate=valid?G[e*{p['p']*c}+{path*c}+channel]:0.f;",
                      f"const float scale=w{path}*mask;float dr=0.f,di=0.f;"]
            if factored:lines.append('const float gate_scale=gate*scale;')
            if p.get('componentwise_first', False):
                if not (destination_owned and owned_edges and factored):
                    raise ValueError('Componentwise first VJP requires factored destination ownership')
                lines += _componentwise_first_path({**p, '_first_path': path}, columns,
                    ni, bb, bs, ob, m, outputs)
                lines.append('}')
                continue
            for mag in range(m):
                offset = mag*(width//m)+(ob-base)//m
                for stream, prefix in enumerate(("v", "vr", "vi")):
                    if destination_owned:
                        lines.append(f'const float {prefix}{mag}={prefix}p{path}_{mag};')
                    else:
                        lines.append(f"const float {prefix}{mag}=valid?V{degree}[(dst*3+{stream})*{width}+{offset}+channel]:0.f;")
                lines.append(f"const float b{mag}=v{mag}+cr*vr{mag}+ci*vi{mag};")
                for j in range(ni):
                    terms = columns[bb+j*bs+mag]
                    if terms:
                        lines.append(f"const float q{j}_{mag}="+_sum(f"{_literal(v)}*h{h}" for h, v in terms)+";")
                terms = [f"x{j}*q{j}_{mag}" for j in range(ni) if columns[bb+j*bs+mag]]
                lines.append(f"const float t{mag}="+_sum(terms)+";")
                if factored:
                    lines.append(f'dr+=vr{mag}*t{mag};di+=vi{mag}*t{mag};')
                else:
                    lines.append(f"dr+=vr{mag}*t{mag}*gate*scale;di+=vi{mag}*t{mag}*gate*scale;")
            for j in range(ni):
                terms = [f"b{mag}*q{j}_{mag}" for mag in range(m) if columns[bb+j*bs+mag]]
                lines += [f"const float z{j}="+_sum(terms)+";",
                          f"if(needx)dx{j}+=gate*scale*z{j};"]
            value = _sum(f"x{j}*z{j}" for j in range(ni))
            lines.append(f"if(needg&&valid)DG[e*{p['p']*c}+{path*c}+channel]=({value})*scale;")
            lines.append("if(needh){")
            if factored:
                for h in range(hdim):
                    outer = []
                    for j in range(ni):
                        inner = []
                        for mag in range(m):
                            inner.extend(f'{_literal(value)}*b{mag}' for harmonic, value in columns[bb+j*bs+mag]
                                         if harmonic == h)
                        if inner:outer.append(f'x{j}*('+_sum(inner)+')')
                    if outer:lines.append(f'dh{h}+=gate_scale*('+_sum(outer)+');')
            else:
                for j in range(ni):
                    for mag in range(m):
                        if not columns[bb+j*bs+mag]:
                            continue
                        lines.append(f"const float db{j}_{mag}=x{j}*b{mag}*gate*scale;")
                        for h, v in columns[bb+j*bs+mag]:
                            lines.append(f"dh{h}+=({_literal(v)})*db{j}_{mag};")
            lines.append("}")
            if factored:lines.append('dr*=gate_scale;di*=gate_scale;')
            if owned_edges:
                lines.append(f"pr{phase//p['phase_heads']}+=dr;pi{phase//p['phase_heads']}+=di;")
            else:
                for head in range(p["phase_heads"]):
                    lines += ["{", f"float rr=(valid&&head=={head})?dr:0.f,ii=(valid&&head=={head})?di:0.f;",
                              "for(int d=16;d>0;d/=2){rr+=__shfl_down_sync(0xffffffff,rr,d);ii+=__shfl_down_sync(0xffffffff,ii,d);}",
                              f"if(lane==0){{if(needr)atomicAdd(DR+e*{p['phase_count']}+{phase+head},rr);",
                              f"if(needi)atomicAdd(DI+e*{p['phase_count']}+{phase+head},ii);}}", "}"]
            lines.append("}")
        if owned_edges:
            slots = tiles*len(groups)
            if coalesced:lines.append('float pr_value=0.f,pi_value=0.f;')
            for q in phase_bases:
                for head in range(p['phase_heads']):
                    phase_index = q*p['phase_heads']+head
                    lines += ['{', f'float rr=(valid&&head=={head})?pr{q}:0.f,ii=(valid&&head=={head})?pi{q}:0.f;',
                        'for(int d=16;d>0;d/=2){rr+=__shfl_down_sync(0xffffffff,rr,d);ii+=__shfl_down_sync(0xffffffff,ii,d);}']
                    if coalesced:
                        lines += ['rr=__shfl_sync(0xffffffff,rr,0);ii=__shfl_sync(0xffffffff,ii,0);',
                            f'if(lane=={phase_index}){{pr_value=rr;pi_value=ii;}}', '}']
                    else:
                        lines += [f'if(lane==0){{DR[(e*{p["phase_count"]}+{phase_index})*{slots}+group*{tiles}+channel/32]=rr;',
                            f'DI[(e*{p["phase_count"]}+{phase_index})*{slots}+group*{tiles}+channel/32]=ii;}}', '}']
            if coalesced:
                lines += [f'if(lane<{p["phase_count"]}){{DR[(e*{slots}+group*{tiles}+channel/32)*{p["phase_count"]}+lane]=pr_value;',
                          f'DI[(e*{slots}+group*{tiles}+channel/32)*{p["phase_count"]}+lane]=pi_value;}}']
        lines.append("if(needh){")
        if owned_edges and coalesced:lines.append('float dh_value=0.f;')
        for h in range(hdim):
            lines.append(f"for(int d=16;d>0;d/=2)dh{h}+=__shfl_down_sync(0xffffffff,dh{h},d);")
            if owned_edges and coalesced:
                lines += [f'const float dh_full{h}=__shfl_sync(0xffffffff,dh{h},0);', f'if(lane=={h})dh_value=dh_full{h};']
            elif owned_edges:
                lines.append(f"if(lane==0)DH[(e*{hdim}+{h})*{slots}+group*{tiles}+channel/32]=dh{h};")
            else:
                lines.append(f"if(lane==0)atomicAdd(DH+e*{hdim}+{h},dh{h});")
        if owned_edges and coalesced:
            lines.append(f'if(lane<{hdim})DH[(e*{slots}+group*{tiles}+channel/32)*{hdim}+lane]=dh_value;')
        lines.append('}')
        if destination_owned:
            for j in range(ni):
                lines.append(f"if(needx&&valid)DX[e*{p['xdim']}+{xb+j*c}+channel]=dx{j};")
        lines.append('}')
        if not destination_owned:
            for j in range(ni):
                lines.append(f"if(needx&&valid)DX[src*{p['xdim']}+{xb+j*c}+channel]=dx{j};")
        lines.append("}")
    lines.append("}")
    wrapper_args = ",".join(f"torch::Tensor v{l}" for l, _, _ in outputs)
    pointer_args = ",".join(f"v{l}.data_ptr<float>()" for l, _, _ in outputs)
    lines.append(f"""
void coherent_adjoint(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor dst,torch::Tensor mask,{wrapper_args},
 torch::Tensor ptr,torch::Tensor order,torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,
 torch::Tensor dr,torch::Tensor di,bool needx,bool needh,bool needg,bool needr,bool needi) {{
 const c10::cuda::CUDAGuard guard(x.device());
 const int64_t blocks=(x.size(0)*{tiles*len(groups)}+3)/4;
 if(!blocks)return;
 auto stream=at::cuda::getCurrentCUDAStream();
 coherent_adjoint_kernel<<<blocks,128,0,stream>>>(x.data_ptr<float>(),h.data_ptr<float>(),
 w.data_ptr<float>(),g.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),
 dst.data_ptr<int64_t>(),mask.data_ptr<float>(),{pointer_args},ptr.data_ptr<int64_t>(),
 order.data_ptr<int64_t>(),dx.data_ptr<float>(),dh.data_ptr<float>(),dg.data_ptr<float>(),
 dr.data_ptr<float>(),di.data_ptr<float>(),x.size(0),dst.numel(),mask.numel()!=0,
 needx,needh,needg,needr,needi);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
""")
    return "\n".join(lines)


def _extension(p, projection):
    key = (p["path_specs"], p["xdim"], p["odim"], p["c"], p["phase_heads"], p["phase_angular"],
           tuple(tuple(row) for row in projection["static_rows"]))
    if key not in _EXTENSIONS:
        from torch.utils.cpp_extension import load_inline
        name = f"ictc_coherent_hybrid_v5_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}_a{int(p['phase_angular'])}"
        outputs = len(_outputs(p))
        declarations = "#include <torch/extension.h>\nvoid coherent_forward("+",".join(["torch::Tensor"]*(10+outputs))+");\n"
        declarations += "void coherent_adjoint("+",".join(["torch::Tensor"]*(16+outputs)+["bool"]*5)+");\n"
        declarations += "void coherent_csr_adjoint("+",".join(["torch::Tensor"]*(15+outputs)+["bool"]*5)+");\n"
        from ._coherent_adjoint import source as adjoint_source
        _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declarations,
            cuda_sources=_forward_source(p, projection)+adjoint_source(
                p, projection, _columns(p, projection), _outputs(p))+_csr_adjoint_source(
                    p, projection).replace("void coherent_adjoint(", "void coherent_csr_adjoint("),
            functions=["coherent_forward", "coherent_adjoint", "coherent_csr_adjoint"], extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    return _EXTENSIONS[key]


def _reference(x, sh, w, g, real, imag, src, dst, mask, p, projection):
    from mace_ictc.models.rematerialized_message import _native
    paths = [None]*p["p"]
    for row in p["path_specs"]:
        paths[row[0]] = ((row[-1]-1)//2)*p["phase_heads"] if p["phase_angular"] else 0
    index = torch.tensor([base+head for base in paths for head in range(p["phase_heads"])],
                         device=x.device, dtype=torch.long)
    # Generated schedules can reorder paths; the external gate index is row[0].
    def modulate(coefficient):
        factor = coefficient.index_select(1, index).reshape(g.shape[0], p["p"], p["phase_heads"], 1)
        return (g.reshape(g.shape[0], p["p"], p["phase_heads"], p["c"]//p["phase_heads"])*factor).reshape_as(g)
    values = [_native(x, sh, w, gate.reshape(g.shape[0], p["p"], p["c"]), src, dst, mask, p, projection)
              for gate in (g, modulate(real), modulate(imag))]
    return torch.cat(values, dim=-1)


class _Coherent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, w, g, real, imag, src, dst, mask, p, projection):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, sh, w, g, real, imag, src, dst, mask)
        ctx.p, ctx.projection = p, projection
        out = tuple((x.new_empty if src.numel() else x.new_zeros)(
            x.shape[0], 3, 2*l+1, width//(2*l+1)).transpose(-1, -2)
            for l, _, width in _outputs(p))
        if src.numel():
            from mace_ictc.models.message_topology import topology
            _extension(p, projection).coherent_forward(x, sh, w, g, real, imag, src, mask,
                                                       *topology(dst, x.shape[0]), *out)
        return out

    @staticmethod
    def backward(ctx, *grads):
        x, sh, w, g, real, imag, src, dst, mask = ctx.saved_tensors
        p, projection = ctx.p, ctx.projection
        from mace_ictc.models.adjoint_requirements import required_gradients
        needed = required_gradients((x, sh, g, real, imag), tuple(
            ctx.needs_input_grad[i] for i in (0, 1, 3, 4, 5)))
        if torch.is_grad_enabled():
            from ._coherent_training import coherent_adjoint
            dx, dh, dg, dr, di = coherent_adjoint(
                x, sh, w, g, real, imag, src, dst, mask, p, projection, needed, grads)
        elif (p.get('first_schedule', 'legacy') != 'legacy' or
              ((p.get('training_split_first', False) or p.get('training_csr_splits', 0)
                or p.get('training_destination_first', 0) or p.get('first_cache', False)) and needed[0])):
            from ._coherent_training import _first
            dx, dh, dg, dr, di = _first(x, sh, w, g, real, imag, src, dst, mask,
                                        p, projection, needed, grads)
        elif all(needed[1:]) and p.get('destination_first_adjoint', False):
            from ._coherent_destination import adjoint
            dx, dh, dg, dr, di = adjoint(x, sh, w, g, real, imag, src, dst, mask,
                p, projection, needed, grads, splits=p.get('destination_splits', 1), atomic_x=True, factored=True,
                coalesced=True, group_paths=p.get('destination_group_paths', 0))
        elif all(needed) and p.get("adjoint_splits", 1) > 1:
            from ._coherent_csr_split import adjoint
            dx, dh, dg, dr, di = adjoint(x, sh, w, g, real, imag, src, dst, mask,
                p, projection, needed, grads, splits=p["adjoint_splits"])
        else:
            empty = x.new_empty(0)
            dx = torch.zeros_like(x) if needed[0] else empty
            allocate = torch.zeros_like if needed[0] else torch.empty_like
            dh = allocate(sh) if needed[1] else empty
            dg = (torch.empty_like(g) if src.numel() else torch.zeros_like(g)) if needed[2] else empty
            dr = allocate(real) if needed[3] else empty
            di = allocate(imag) if needed[4] else empty
            if src.numel():
                from mace_ictc.models.message_topology import topology
                cotangents = [grad.transpose(-1, -2).contiguous() if grad is not None else x.new_zeros(
                    x.shape[0], 3, 2*l+1, width//(2*l+1))
                    for (l, _, width), grad in zip(_outputs(p), grads)]
                extension = _extension(p, projection)
                if needed[0]:
                    # The joint source-owned schedule reuses TP work for DX;
                    # splitting it into a second edge kernel was slower.
                    extension.coherent_csr_adjoint(x, sh, w, g, real, imag, dst, mask,
                        *cotangents, *topology(src, x.shape[0]), dx, dh, dg, dr, di, *needed)
                else:
                    extension.coherent_adjoint(x, sh, w, g, real, imag, src, dst, mask,
                        *cotangents, src.new_empty(0), src.new_empty(0), dx, dh, dg, dr, di, *needed)
            dx, dh, dg, dr, di = [value if need else None for value, need in zip((dx, dh, dg, dr, di), needed)]
        return dx, dh, None, dg, dr, di, None, None, None, None, None


class CoherentGraphMessage(torch.nn.Module):
    """Three native ICTC streams, one gate bank and one joint first adjoint.

    Inputs follow GraphMessage, plus real/imag scalar coefficients [E,F,1].
    F=heads*(output_lmax+1) for angular heads, otherwise heads. Returns each
    degree as [N,3,path_count*C,2l+1], with streams neutral/real/imag.
    Its physical layout is [N,3,2l+1,path_count*C], so the next shared
    channel GEMM needs no permutation copy. Contiguity is not part of the API.
    adjoint_splits=0 selects destination-cached first VJP for large sm80
    C128/L2 graphs, CSR4 below that size, and the joint schedule elsewhere.
    large_graph_fusion=False disables the destination schedule; explicit
    adjoint_splits preserves the selected CSR/joint schedule.
    This changes reduction ownership, not the TP paths or gradient targets.
    """
    def __init__(self, tp, *, input_lmax, edge_lmax, heads=1, angular=True, adjoint_splits=0,
                 large_graph_fusion=True, mixed_threads=0, training_split_first=False,
                 mixed_grouped=False, training_csr_splits=0, mixed_destination=False,
                 training_destination_first=0, mixed_compact=False, mixed_cache=0, first_cache=False,
                 mixed_fuse_jvp=False, first_schedule='legacy'):
        super().__init__()
        if not isinstance(heads, int) or not 1 <= heads <= 4 or tp.channels % heads:
            raise ValueError("Coherent graph heads must divide channels and lie in 1..4")
        if max(input_lmax, edge_lmax, tp.lmax) > 3:
            raise ValueError("Coherent graph schedule currently supports L<=3")
        self.edge = EdgeTensorProduct(tp, input_lmax=input_lmax, edge_lmax=edge_lmax, layout="ir_mul")
        self.heads, self.angular = heads, bool(angular)
        self.output_lmax = max(path[2] for path in tp.paths)
        if isinstance(adjoint_splits, bool) or adjoint_splits not in (0, 1, 2, 4, 8):
            raise ValueError("adjoint_splits must be 0 (structural auto), 1, 2, 4 or 8")
        self.adjoint_splits = adjoint_splits
        if not isinstance(large_graph_fusion, bool):
            raise TypeError('large_graph_fusion must be a boolean')
        self.large_graph_fusion = large_graph_fusion
        if isinstance(mixed_threads, bool) or mixed_threads not in (0, 32, 64, 128):
            raise ValueError('mixed_threads must be 0 (default), 32, 64 or 128')
        self.mixed_threads = mixed_threads
        if not isinstance(training_split_first, bool):
            raise TypeError('training_split_first must be a boolean')
        self.training_split_first = training_split_first
        if not isinstance(mixed_grouped, bool):
            raise TypeError('mixed_grouped must be a boolean')
        self.mixed_grouped = mixed_grouped
        if not isinstance(mixed_compact, bool):
            raise TypeError('mixed_compact must be a boolean')
        self.mixed_compact = mixed_compact
        if isinstance(mixed_cache, bool) or mixed_cache not in (0, 8, 16, 32, 64):
            raise ValueError('mixed_cache must be 0 (off), 8, 16, 32 or 64 components')
        if mixed_cache and tp.channels % 32:
            raise ValueError('Destination-cached mixed adjoint requires channel multiples of 32')
        self.mixed_cache = mixed_cache
        if not isinstance(mixed_fuse_jvp, bool):
            raise TypeError('mixed_fuse_jvp must be a boolean')
        if mixed_fuse_jvp and not mixed_cache:
            raise ValueError('mixed_fuse_jvp requires a destination-cached mixed schedule')
        self.mixed_fuse_jvp = mixed_fuse_jvp
        if not isinstance(first_cache, bool):
            raise TypeError('first_cache must be a boolean')
        if first_cache and tp.channels % 32:
            raise ValueError('Cached first adjoint requires channel multiples of 32')
        self.first_cache = first_cache
        if first_schedule not in ('legacy', 'pruned', 'grouped', 'compact', 'compact_grouped'):
            raise ValueError('Invalid first-adjoint schedule')
        if first_schedule != 'legacy' and (tp.channels % 32 or first_cache):
            raise ValueError('Fused first schedule requires channel multiples of 32 and excludes first_cache')
        self.first_schedule = first_schedule
        if isinstance(training_csr_splits, bool) or training_csr_splits not in (0, 1, 2, 4, 8):
            raise ValueError('training_csr_splits must be 0, 1, 2, 4 or 8')
        if training_csr_splits and training_split_first:
            raise ValueError('CSR-neighbor and edge/source split schedules are mutually exclusive')
        self.training_csr_splits = training_csr_splits
        if not isinstance(mixed_destination, bool):
            raise TypeError('mixed_destination must be a boolean')
        self.mixed_destination = mixed_destination
        if isinstance(training_destination_first, bool) or training_destination_first not in (0, 1, 2, 4):
            raise ValueError('training_destination_first must be 0, 1, 2 or 4')
        if training_destination_first and (training_split_first or training_csr_splits):
            raise ValueError('Only one explicit first-adjoint schedule may be selected')
        self.training_destination_first = training_destination_first
        # Explicit candidate switch until full-model qualification. Only a
        # genuinely unused frozen X cotangent may be omitted by the adjoint.
        self.frozen_input_fusion = False
        self.destination_splits = 1
        self.destination_group_paths = 0
        self.distance_destination = False
        self.distance_projection = 'gemm'
        self.distance_cached_gates = False
        self._auto_split_supported = torch.cuda.get_device_capability(tp.weight.device) == (8, 0)

    def radial(self, blocks, harmonics, hidden, weight, src, dst, mask, real, imag, *,
               tile_edges=4096, gate_pipeline=False):
        """Fuse the frozen radial output linear and its large gate cotangent."""
        from ._radial_coherent import radial_coherent
        return radial_coherent(self, blocks, harmonics, hidden, weight, src, dst, mask, real, imag,
                               tile_edges=tile_edges, gate_pipeline=gate_pipeline)

    def radial_training(self, blocks, harmonics, hidden, bank, src, dst, mask, real, imag, *,
                        tile_edges=8192):
        """Live radial-output bank, bounded mixed gate VJP and inline gate JVP."""
        from ._radial_training import radial_training
        return radial_training(self, blocks, harmonics, hidden, bank, src, dst, mask,
                               real, imag, tile_edges=tile_edges)

    def distance(self, blocks, harmonics, distance, radial, src, dst, mask, real, imag, *,
                 tile_edges=65536, gate_pipeline=False):
        """Exact one-dimensional radial directional derivative for frozen E/F."""
        from ._radial_distance import distance_coherent
        return distance_coherent(self, blocks, harmonics, distance, radial, src, dst, mask, real, imag,
                                  tile_edges=tile_edges, gate_pipeline=gate_pipeline)

    def forward(self, blocks, harmonics, gates, src, dst, mask, real, imag):
        _eager(blocks[0])
        p0, projection = self.edge._prepare()
        phases = self.heads*(self.output_lmax+1 if self.angular else 1)
        n, e = blocks[0].shape[0], src.numel()
        for name, value in (("src", src), ("dst", dst)):
            if value.shape != (e,) or value.dtype != torch.int64 or value.device != gates.device:
                raise ValueError(f"{name} must be int64 [E] on the execution device")
        for value in (real, imag):
            _operand(value, (e, phases, 1), self.edge.tp.weight, "phase coefficient")
        _operand(gates, (e, self.edge.gate_width), self.edge.tp.weight, "gates")
        if mask is not None:
            if mask.requires_grad or mask.shape not in ((e,), (e, 1)):
                raise ValueError("Coherent mask must be fixed [E] or [E,1]")
            _operand(mask, mask.shape, self.edge.tp.weight, "mask")
        p = {**p0, "native_packed": True, "phase_heads": self.heads,
             "phase_angular": self.angular, "phase_count": phases}
        p['mixed_threads'] = self.mixed_threads
        p['training_split_first'] = self.training_split_first
        p['mixed_grouped'] = self.mixed_grouped
        p['mixed_compact'] = self.mixed_compact
        p['mixed_cache'] = self.mixed_cache
        p['mixed_fuse_jvp'] = self.mixed_fuse_jvp
        p['first_cache'] = self.first_cache
        p['first_schedule'] = self.first_schedule
        p['training_csr_splits'] = self.training_csr_splits
        p['mixed_destination'] = self.mixed_destination
        p['training_destination_first'] = self.training_destination_first
        # The large C128/L2 live-input VJP is measured faster with four CSR
        # neighbor owners. Keep other structures on the established schedule.
        p["adjoint_splits"] = self.adjoint_splits or (4 if (
            self._auto_split_supported and e >= 32768 and p["c"] == 128 and p["xdim"] == 1152
            and p["odim"] == 4480 and self.heads == 1 and self.angular) else 1)
        p['destination_first_adjoint'] = bool(self.large_graph_fusion and self.adjoint_splits == 0
            and self._auto_split_supported and e >= 327680 and p['c'] == 128
            and p['xdim'] == 1152 and p['odim'] == 4480 and self.heads == 1 and self.angular)
        p['destination_first_adjoint'] |= bool(self.frozen_input_fusion
            and self._auto_split_supported and e >= 327680 and p['c'] == 128
            and p['xdim'] == 128 and p['odim'] == 1152 and self.heads == 1 and self.angular)
        p['destination_splits'] = self.destination_splits
        p['destination_group_paths'] = self.destination_group_paths if p['xdim'] > 128 else 0
        x = self.edge.pack_input(blocks)
        sh = torch.cat([harmonics[l].squeeze(-2) for l in projection["keys"]], -1).contiguous()
        _operand(sh, (e, projection["hdim"]), self.edge.tp.weight, "harmonics")
        m = gates.new_empty(0) if mask is None else mask.reshape(e).contiguous()
        output = _Coherent.apply(x, sh, self.edge.tp.weight, gates.contiguous(), real.squeeze(-1).contiguous(),
            imag.squeeze(-1).contiguous(), src.contiguous(), dst.contiguous(), m, p, projection)
        return {l: value for (l, _, _), value in zip(_outputs(p), output)}
