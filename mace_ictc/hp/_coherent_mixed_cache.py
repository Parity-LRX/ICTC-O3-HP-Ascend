# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Destination-cached high-l mixed TP adjoint with live tangent operands.

A warp owns a destination/path group/channel tile. Its three-stream output
cotangents are loaded once and reused across neighbors. Exact ICTC CG columns
are consumed component by component. Source features use FP32 atomics; edge
harmonic/phase partials have unique writers and an explicit final reduction.
"""
import torch

from mace_ictc.models.generated_cartesian_message import _sum

_EXTENSIONS = {}


def _split_groups(p):
    return bool(p.get("mixed_grouped", False)) and not p.get("radial_cached", False)


def _groups(p):
    grouped = {}
    for row in p["path_specs"]:
        grouped.setdefault((row[2], row[1]), []).append(row)
    result = []
    for (xb, ni), paths in grouped.items():
        current, width = [], 0
        for row in paths:
            if current and width+row[-1] > p["mixed_cache"]:
                result.append((xb, ni, current))
                current, width = [], 0
            current.append(row)
            width += row[-1]
        if current:
            result.append((xb, ni, current))
    return result


def _specialize_groups(code, p):
    """Independent register allocation for each exact input/path group."""
    tiles, groups = p['c']//32, len(_groups(p))
    slots = tiles*groups
    replacements = (
        ('__global__ void coherent_cached_mixed_kernel', 'template<int GROUP>\n__global__ void coherent_cached_mixed_kernel'),
        (f'if(task>=N*{slots})return;', f'if(task>=N*{tiles})return;'),
        (f'const int group=(task/{tiles})%{groups};', 'constexpr int group=GROUP;'),
        (f'const int64_t dst=task/{slots};', f'const int64_t dst=task/{tiles};'),
    )
    for old, new in replacements:
        if code.count(old) != 1:
            raise RuntimeError('Cached mixed group specialization marker changed')
        code = code.replace(old, new, 1)
    before, call = code.split('\n coherent_cached_mixed_kernel<<<', 1)
    call, after = call.split(';\n coherent_cached_mixed_reduce', 1)
    call = call.replace(f'x.size(0)*{slots}', f'x.size(0)*{tiles}', 1)
    launches = '\n'.join(f' coherent_cached_mixed_kernel<{group}><<<{call};'
                         for group in range(groups))
    return before+'\n'+launches+'\n coherent_cached_mixed_reduce'+after


def _source(p, projection, dependencies):
    from .coherent import _HEADERS, _columns, _outputs
    from ._coherent_training import _compact_path
    c, hd, phases = p["c"], projection["hdim"], p["phase_count"]
    outputs, columns, groups = _outputs(p), _columns(p, projection), _groups(p)
    tiles, slots = c//32, c//32*len(groups)
    fuse_jvp = bool(p.get("mixed_fuse_jvp", False))
    flags = ("hx", "hh", "hg", "hr", "hi", "needx", "needh", "needg", "needr", "needi")
    if fuse_jvp:
        flags += ("needy",)
    constants = ";".join(f"constexpr bool {name}={str(bool(value)).lower()}"
                         for name, value in zip(flags, dependencies))+";"
    if not fuse_jvp:
        constants += "constexpr bool needy=false;"
    vargs = ",".join(f"const float* V{l}" for l, _, _ in outputs)
    yargs = ",".join(f"float* DY{l}" for l, _, _ in outputs)+"," if fuse_jvp else ""
    unused_dy = "" if fuse_jvp else ";".join(f"float* DY{l}=nullptr" for l, _, _ in outputs)+";"
    lines = [_HEADERS, f"""
__global__ void coherent_cached_mixed_kernel(const float* X,const float* H,const float* W,
 const float* G,const float* R,const float* I,const int64_t* SRC,const float* MASK,
 {vargs},{yargs}const int64_t* PTR,const int64_t* ORDER,
 const float* TX,const float* TH,const float* TG,const float* TR,const float* TI,
 float* DX,float* PH,float* DG,float* PR,float* PI,int64_t N,bool masked) {{
 {constants}{unused_dy}
 const int lane=threadIdx.x%32;
 const int64_t task=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
 if(task>=N*{slots})return;
 const int tile=task%{tiles},channel=tile*32+lane;
 const int group=(task/{tiles})%{len(groups)};
 const int64_t dst=task/{slots};
 const int head=channel/{c//p['phase_heads']};
"""]
    for number, (xb, ni, paths) in enumerate(groups):
        lines.append(f"if(group=={number}){{")
        for path, _, _, _, _, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            for mag in range(m):
                offset = mag*(width//m)+(ob-base)//m
                for stream, prefix in enumerate(("v", "vr", "vi")):
                    lines.append(f"const float c{prefix}{path}_{mag}=V{degree}[(dst*3+{stream})*{width}+{offset}+channel];")
                if fuse_jvp:
                    for stream in range(3):
                        lines.append(f"float y{path}_{mag}_{stream}=0.f;")
        lines += ["for(int64_t index=PTR[dst];index<PTR[dst+1];++index){",
                  "const int64_t e=ORDER[index],src=SRC[e];const float mask=masked?MASK[e]:1.f;"]
        for h in range(hd):
            lines += [f"const float h{h}=H[e*{hd}+{h}],th{h}=hh?TH[e*{hd}+{h}]:0.f;",
                      f"float dh{h}=0.f;"]
        for phase in range(phases):
            lines.append(f"float dr{phase}=0.f,di{phase}=0.f;")
        for j in range(ni):
            lines += [f"const float x{j}=X[src*{p['xdim']}+{xb+j*c}+channel];",
                      f"const float tx{j}=hx?TX[src*{p['xdim']}+{xb+j*c}+channel]:0.f;float dx{j}=0.f;"]
        for path, _, _, bb, bs, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            phase = degree*p["phase_heads"] if p["phase_angular"] else 0
            lines += ["{", f"const int phase={phase}+head;",
                f"const float cr=R[e*{phases}+phase],ci=I[e*{phases}+phase];",
                f"const float tr=hr?TR[e*{phases}+phase]:0.f,ti=hi?TI[e*{phases}+phase]:0.f;",
                f"const float gate=G[e*{p['p']*c}+{path*c}+channel];",
                f"const float tg=hg?TG[e*{p['p']*c}+{path*c}+channel]:0.f;",
                f"const float scale=W[{path*c}+channel]*mask;"]
            body = "\n".join(_compact_path({**p, "_compact_path": path}, columns, ni, bb, bs,
                degree, (ob-base)//m, width, m, phase))
            if p.get("cache_first", False):
                # With TG=G and all other tangents zero, the X/H/phase mixed
                # targets equal the first VJP. Its gate target is u*vef, not
                # the zero gate-gate second derivative.
                body = body.replace("dg_path+=tu*vef+u*tv;", "dg_path+=u*vef;")
            for mag in range(m):
                offset = mag*(width//m)+(ob-base)//m
                for stream, prefix in enumerate(("v", "vr", "vi")):
                    marker = f"const float {prefix}=V{degree}[(dst*3+{stream})*{width}+{offset}+channel];"
                    if body.count(marker) != 1:
                        raise RuntimeError("Mixed cache cotangent marker changed")
                    body = body.replace(marker, f"const float {prefix}=c{prefix}{path}_{mag};")
                if fuse_jvp:
                    for stream, value in enumerate(("a", "cr*a+tr*gate*u", "ci*a+ti*gate*u")):
                        marker = f"atomicAdd(DY{degree}+(dst*3+{stream})*{width}+{offset}+channel,scale*({value}));"
                        if body.count(marker) != 1:
                            raise RuntimeError("Mixed cache fused JVP marker changed")
                        body = body.replace(marker, f"y{path}_{mag}_{stream}+=scale*({value});")
            lines += [body, "}"]
        for j in range(ni):
            lines.append(f"if(needx)atomicAdd(DX+src*{p['xdim']}+{xb+j*c}+channel,dx{j});")
        for name, target, width, count, need in (("dh", "PH", hd, hd, "needh"),
                ("dr", "PR", phases, phases, "needr"), ("di", "PI", phases, phases, "needi")):
            for index in range(count):
                lines += [f"if({need}){{for(int d=16;d>0;d/=2){name}{index}+=__shfl_down_sync(0xffffffff,{name}{index},d);",
                    f"if(lane==0){target}[(e*{slots}+{number*tiles}+tile)*{width}+{index}]={name}{index};}}"]
        lines.append("}")
        if fuse_jvp:
            for path, _, _, _, _, ob, m in paths:
                degree = (m-1)//2
                _, base, width = next(row for row in outputs if row[0] == degree)
                for mag in range(m):
                    offset = mag*(width//m)+(ob-base)//m
                    for stream in range(3):
                        lines.append(f"if(needy)DY{degree}[(dst*3+{stream})*{width}+{offset}+channel]=y{path}_{mag}_{stream};")
        lines.append("}")
    lines += ["}", f"""
__global__ void coherent_cached_mixed_reduce(const float* PH,const float* PR,const float* PI,
 float* DH,float* DR,float* DI,int64_t E) {{
 {constants}
 const int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(needh&&index<E*{hd}){{
  const int64_t e=index/{hd},col=index%{hd};float value=0.f;
  #pragma unroll 4
  for(int s=0;s<{slots};++s)value+=PH[(e*{slots}+s)*{hd}+col];
  DH[index]=value;
 }}
 if((needr||needi)&&index<E*{phases}){{
  const int64_t e=index/{phases},col=index%{phases};float r=0.f,i=0.f;
  #pragma unroll 4
  for(int s=0;s<{slots};++s){{
   if(needr)r+=PR[(e*{slots}+s)*{phases}+col];
   if(needi)i+=PI[(e*{slots}+s)*{phases}+col];
  }}
  if(needr)DR[index]=r;if(needi)DI[index]=i;
 }}
}}
"""]
    wrappers = ",".join(f"torch::Tensor v{l}" for l, _, _ in outputs)
    pointers = ",".join(f"v{l}.data_ptr<float>()" for l, _, _ in outputs)
    ywrappers = ",".join(f"torch::Tensor dy{l}" for l, _, _ in outputs)+"," if fuse_jvp else ""
    ypointers = ",".join(f"dy{l}.data_ptr<float>()" for l, _, _ in outputs)+"," if fuse_jvp else ""
    needed_y = ",bool needy" if fuse_jvp else ""
    lines.append(f"""
void coherent_cached_mixed(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor mask,{wrappers},{ywrappers}
 torch::Tensor ptr,torch::Tensor order,torch::Tensor tx,torch::Tensor th,torch::Tensor tg,
 torch::Tensor tr,torch::Tensor ti,torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,
 torch::Tensor dr,torch::Tensor di,torch::Tensor ph,torch::Tensor pr,torch::Tensor pi,
 bool hx,bool hh,bool hg,bool hr,bool hi,bool needx,bool needh,bool needg,bool needr,bool needi{needed_y}) {{
 const c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
 if(!src.numel())return;
 coherent_cached_mixed_kernel<<<(x.size(0)*{slots}+3)/4,128,0,stream>>>(
 x.data_ptr<float>(),h.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
 r.data_ptr<float>(),i.data_ptr<float>(),src.data_ptr<int64_t>(),mask.data_ptr<float>(),
 {pointers},{ypointers}ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),
 tx.data_ptr<float>(),th.data_ptr<float>(),tg.data_ptr<float>(),tr.data_ptr<float>(),ti.data_ptr<float>(),
 dx.data_ptr<float>(),ph.data_ptr<float>(),dg.data_ptr<float>(),pr.data_ptr<float>(),pi.data_ptr<float>(),
 x.size(0),mask.numel()!=0);
 coherent_cached_mixed_reduce<<<(src.numel()*{max(hd,phases)}+255)/256,256,0,stream>>>(
 ph.data_ptr<float>(),pr.data_ptr<float>(),pi.data_ptr<float>(),dh.data_ptr<float>(),dr.data_ptr<float>(),di.data_ptr<float>(),src.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
""")
    code = "\n".join(lines)
    if _split_groups(p):
        # Separate template instantiations let each input-degree/path group use
        # its own register allocation. In the monolithic warp-uniform switch,
        # every group pays the register footprint of the widest CG block.
        code = _specialize_groups(code, p)
    return code.replace("float* ", "float* __restrict__ ").replace("int64_t* ", "int64_t* __restrict__ ")


def mixed(x, sh, w, gates, r, i, src, dst, mask, p, projection, needs, grads, incoming):
    from .coherent import _outputs
    from ._coherent_training import _physical_gradients
    from ._coherent_training_jvp import destination_jvp
    from mace_ictc.models.message_topology import topology
    from torch.utils.cpp_extension import load_inline
    active = tuple(value is not None for value in incoming)
    dependencies = (*active, *needs[:5])
    fuse_jvp = bool(p.get("mixed_fuse_jvp", False))
    if fuse_jvp:
        dependencies += (any(needs[5:]),)
    key = (p["path_specs"], p["xdim"], p["odim"], p["c"], p["phase_heads"], p["phase_angular"],
           p["mixed_cache"], p.get("cache_first", False), fuse_jvp, _split_groups(p),
           tuple(tuple(row) for row in projection["static_rows"]), dependencies)
    if key not in _EXTENSIONS:
        name = (f"ictc_mixed_cache_v1_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}"
                f"_a{int(p['phase_angular'])}_g{p['mixed_cache']}_dep"
                +"".join(str(int(value)) for value in dependencies))
        if p.get("cache_first", False):
            name += "_first"
        if fuse_jvp:
            name += "_jvp"
        if _split_groups(p):
            name += "_splitgroups"
        declaration = "#include <torch/extension.h>\nvoid coherent_cached_mixed("+",".join(
            ["torch::Tensor"]*(23+len(_outputs(p))*(2 if fuse_jvp else 1))+
            ["bool"]*len(dependencies))+");\n"
        _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration,
            cuda_sources=_source(p, projection, dependencies), functions=["coherent_cached_mixed"],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    empty = x.new_empty(0)
    result = tuple((torch.zeros_like(value) if index == 0 else torch.empty_like(value)) if need else empty
                   for index, (value, need) in enumerate(zip((x, sh, gates, r, i), needs[:5])))
    slots = len(_groups(p))*(p["c"]//32)
    partial = tuple(value.new_empty(value.shape[0], slots, *value.shape[1:]) if need else empty
                    for value, need in zip((sh, r, i), (needs[1], needs[3], needs[4])))
    physical = _physical_gradients(x, p, grads)
    tangents = tuple(empty if value is None else value.contiguous() for value in incoming)
    dy = tuple(torch.empty_like(value) if any(needs[5:]) else empty for value in physical) if fuse_jvp else ()
    if src.numel():
        _EXTENSIONS[key].coherent_cached_mixed(x, sh, w, gates, r, i, src, mask, *physical,
            *(dy if fuse_jvp else ()),
            *topology(dst, x.shape[0]), *tangents, *result, *partial, *dependencies)
    elif fuse_jvp:
        for value in dy:
            value.zero_()
    if not fuse_jvp:
        dy = tuple(torch.empty_like(value) if any(needs[5:]) else empty for value in physical)
    if any(needs[5:]) and not fuse_jvp:
        destination_jvp(x, sh, w, gates, r, i, src, dst, mask, p, projection, tangents, active, dy)
    return (*(value if need else None for value, need in zip(result, needs[:5])),
            *(value.transpose(-1, -2) if need else None for value, need in zip(dy, needs[5:])))
