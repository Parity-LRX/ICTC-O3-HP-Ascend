# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Joint three-stream first/mixed adjoints for energy/force training.

The mixed kernel differentiates the literal Cartesian polynomial in X, H, G,
R and I. It includes harmonic-to-harmonic targets and live output cotangents;
only third/higher derivatives rematerialize the differentiable expression.
No E-by-three-stream message or modulated gate banks are materialized.
"""
import torch

from mace_ictc.models.generated_cartesian_message import _literal, _sum


_EXTENSIONS = {}


def _compact_path(p, columns, ni, bb, bs, degree, offset, width, m, phase):
    """Consume one output component before constructing the next CG column.

    The original schedule keeps both complete ni-by-m basis/tangent matrices
    live through all target contractions. At high l this causes long register
    lifetimes. Factor the two shared adjoint scalars and consume each component
    immediately, preserving every live X/H/G/phase/cotangent target.
    """
    lines = ["float dg_path=0.f,dr_path=0.f,di_path=0.f;"]
    for mag in range(m):
        lines.append("{")
        active = [j for j in range(ni) if columns[bb+j*bs+mag]]
        for j in active:
            terms = columns[bb+j*bs+mag]
            lines += [f"const float b{j}="+_sum(f"{_literal(v)}*h{h}" for h, v in terms)+";",
                      f"const float tb{j}="+_sum(f"{_literal(v)}*th{h}" for h, v in terms)+";"]
        target = mag*(width//m)+offset
        for stream, prefix in enumerate(("v", "vr", "vi")):
            lines.append(f"const float {prefix}=V{degree}[(dst*3+{stream})*{width}+{target}+channel];")
        lines += ["const float vef=v+cr*vr+ci*vi,tv=tr*vr+ti*vi;",
                  "const float u="+_sum(f"x{j}*b{j}" for j in active)+";",
                  "const float tu="+_sum(f"tx{j}*b{j}+x{j}*tb{j}" for j in active)+";",
                  "const float a=gate*tu+tg*u;",
                  "if(needy){"]
        for stream, value in enumerate(("a", "cr*a+tr*gate*u", "ci*a+ti*gate*u")):
            lines.append(f"atomicAdd(DY{degree}+(dst*3+{stream})*{width}+{target}+channel,scale*({value}));")
        lines += ["}", "dg_path+=tu*vef+u*tv;", "dr_path+=a*vr;di_path+=a*vi;",
                  "const float adjoint=gate*vef,mixed=tg*vef+gate*tv;"]
        for j in active:
            lines.append(f"if(needx)dx{j}+=scale*(tb{j}*adjoint+b{j}*mixed);")
            lines.append(f"if(needh){{const float db=scale*(tx{j}*adjoint+x{j}*mixed);")
            for h, value in columns[bb+j*bs+mag]:
                lines.append(f"dh{h}+=({_literal(value)})*db;")
            lines.append("}")
        lines.append("}")
    lines.append(f"if(needg)DG[e*{p['p']*p['c']}+{p['_compact_path']*p['c']}+channel]=scale*dg_path;")
    for head in range(p["phase_heads"]):
        lines.append(f"if(head=={head}){{dr{phase+head}+=scale*dr_path;di{phase+head}+=scale*di_path;}}")
    return lines


def _source(p, projection, dependencies):
    from .coherent import _HEADERS, _columns, _outputs
    c, hd, phases = p["c"], projection["hdim"], p["phase_count"]
    outputs = _outputs(p)
    columns = _columns(p, projection)
    grouped = {}
    for row in p["path_specs"]:
        grouped.setdefault((row[2], row[1]), []).append(row)
    groups = len(grouped) if p.get("mixed_grouped", False) else 1
    threads = p.get("mixed_threads", 0) or (32 if c <= 32 else 128)
    warps = threads//32
    vargs = ",".join(f"const float* V{l}" for l, _, _ in outputs)
    yargs = ",".join(f"float* DY{l}" for l, _, _ in outputs)
    lines = [_HEADERS, f"""
__global__ void coherent_mixed_kernel(const float* X,const float* H,const float* W,
 const float* G,const float* R,const float* I,const int64_t* SRC,const int64_t* DST,
 const float* MASK,{vargs},const float* TX,const float* TH,const float* TG,
 const float* TR,const float* TI,float* DX,float* DH,float* DG,float* DR,float* DI,
 {yargs},int64_t E,bool masked,bool hx,bool hh,bool hg,bool hr,bool hi,
 bool needx,bool needh,bool needg,bool needr,bool needi,bool needy) {{
 const int64_t e=blockIdx.x,src=SRC[e],dst=DST[e];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const float mask=masked?MASK[e]:1.f;
"""]
    for h in range(hd):
        lines += [f"const float h{h}=H[e*{hd}+{h}];",
                  f"const float th{h}=hh?TH[e*{hd}+{h}]:0.f;", f"float dh{h}=0.f;"]
    for phase in range(phases):
        lines.append(f"float dr{phase}=0.f,di{phase}=0.f;")
    lines += [f"for(int channel=threadIdx.x;channel<{c};channel+={threads}){{",
              f"const int head=channel/{c//p['phase_heads']};"]
    for group, ((xb, ni), paths) in enumerate(grouped.items()):
        if groups > 1:
            lines.append(f"if(blockIdx.y=={group}){{")
        lines.append("{")
        for j in range(ni):
            lines += [f"const float x{j}=X[src*{p['xdim']}+{xb+j*c}+channel];",
                      f"const float tx{j}=hx?TX[src*{p['xdim']}+{xb+j*c}+channel]:0.f;",
                      f"float dx{j}=0.f;"]
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
            if p.get("mixed_compact", False):
                lines += _compact_path({**p, "_compact_path": path}, columns, ni, bb, bs,
                    degree, (ob-base)//m, width, m, phase)
                lines.append("}")
                continue
            for j in range(ni):
                for mag in range(m):
                    terms = columns[bb+j*bs+mag]
                    if terms:
                        lines += [f"const float b{j}_{mag}="+_sum(f"{_literal(v)}*h{h}" for h, v in terms)+";",
                                  f"const float tb{j}_{mag}="+_sum(f"{_literal(v)}*th{h}" for h, v in terms)+";"]
            for mag in range(m):
                offset = mag*(width//m)+(ob-base)//m
                for stream, prefix in enumerate(("v", "vr", "vi")):
                    lines.append(f"const float {prefix}{mag}=V{degree}[(dst*3+{stream})*{width}+{offset}+channel];")
                lines += [f"const float vef{mag}=v{mag}+cr*vr{mag}+ci*vi{mag};",
                          f"const float tv{mag}=tr*vr{mag}+ti*vi{mag};",
                          f"const float u{mag}="+_sum(f"x{j}*b{j}_{mag}" for j in range(ni) if columns[bb+j*bs+mag])+";",
                          f"const float tu{mag}="+_sum(f"tx{j}*b{j}_{mag}+x{j}*tb{j}_{mag}" for j in range(ni) if columns[bb+j*bs+mag])+";",
                          f"const float a{mag}=gate*tu{mag}+tg*u{mag};"]
                lines.append("if(needy){")
                for stream, value in enumerate((f"a{mag}", f"cr*a{mag}+tr*gate*u{mag}",
                                                  f"ci*a{mag}+ti*gate*u{mag}")):
                    lines.append(f"atomicAdd(DY{degree}+(dst*3+{stream})*{width}+{offset}+channel,scale*({value}));")
                lines.append("}")
            dg = _sum(f"tu{mag}*vef{mag}+u{mag}*tv{mag}" for mag in range(m))
            lines.append(f"if(needg)DG[e*{p['p']*c}+{path*c}+channel]=scale*({dg});")
            for head in range(p["phase_heads"]):
                dr = _sum(f"a{mag}*vr{mag}" for mag in range(m))
                di = _sum(f"a{mag}*vi{mag}" for mag in range(m))
                lines += [f"if(head=={head}){{dr{phase+head}+=scale*({dr});",
                          f"di{phase+head}+=scale*({di});}}"]
            for j in range(ni):
                dx = _sum(f"(gate*tb{j}_{mag}+tg*b{j}_{mag})*vef{mag}+gate*b{j}_{mag}*tv{mag}"
                          for mag in range(m) if columns[bb+j*bs+mag])
                lines.append(f"if(needx)dx{j}+=scale*({dx});")
                for mag in range(m):
                    terms = columns[bb+j*bs+mag]
                    if terms:
                        lines.append(f"if(needh){{const float db=scale*((gate*tx{j}+tg*x{j})*vef{mag}+gate*x{j}*tv{mag});")
                        for h, value in terms:
                            lines.append(f"dh{h}+=({_literal(value)})*db;")
                        lines.append("}")
            lines.append("}")
        for j in range(ni):
            lines.append(f"if(needx)atomicAdd(DX+src*{p['xdim']}+{xb+j*c}+channel,dx{j});")
        lines.append("}")
        if groups > 1:
            lines.append("}")
    lines.append("}")
    reductions = [(f"dh{h}", "DH", h, hd, "needh") for h in range(hd)]
    reductions += [(f"dr{i}", "DR", i, phases, "needr") for i in range(phases)]
    reductions += [(f"di{i}", "DI", i, phases, "needi") for i in range(phases)]
    lines.append(f"__shared__ float partial[{len(reductions)*warps}];")
    for slot, (value, _, _, _, need) in enumerate(reductions):
        lines += [f"if({need}){{",
                  f"for(int d=16;d>0;d/=2){value}+=__shfl_down_sync(0xffffffff,{value},d);",
                  f"if(lane==0)partial[{slot*warps}+warp]={value};", "}"]
    lines.append("__syncthreads();")
    for slot, (_, target, offset, stride, need) in enumerate(reductions):
        value = _sum(f"partial[{slot*warps+warp}]" for warp in range(warps))
        lines.append(f"if(threadIdx.x=={slot%threads}&&{need}){target}[(e*{groups}+blockIdx.y)*{stride}+{offset}]={value};")
    lines.append("}")
    vwrapper = ",".join(f"torch::Tensor v{l}" for l, _, _ in outputs)
    ywrapper = ",".join(f"torch::Tensor dy{l}" for l, _, _ in outputs)
    vptrs = ",".join(f"v{l}.data_ptr<float>()" for l, _, _ in outputs)
    yptrs = ",".join(f"dy{l}.data_ptr<float>()" for l, _, _ in outputs)
    lines.append(f"""
void coherent_mixed(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor dst,torch::Tensor mask,
 {vwrapper},torch::Tensor tx,torch::Tensor th,torch::Tensor tg,torch::Tensor tr,torch::Tensor ti,
 torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,torch::Tensor dr,torch::Tensor di,
 {ywrapper},bool hx,bool hh,bool hg,bool hr,bool hi,
 bool needx,bool needh,bool needg,bool needr,bool needi,bool needy) {{
 const c10::cuda::CUDAGuard guard(x.device());
 if(!src.numel())return;
 auto stream=at::cuda::getCurrentCUDAStream();
 coherent_mixed_kernel<<<dim3(src.numel(),{groups}),{threads},0,stream>>>(x.data_ptr<float>(),h.data_ptr<float>(),
 w.data_ptr<float>(),g.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),
 src.data_ptr<int64_t>(),dst.data_ptr<int64_t>(),mask.data_ptr<float>(),{vptrs},
 tx.data_ptr<float>(),th.data_ptr<float>(),tg.data_ptr<float>(),tr.data_ptr<float>(),ti.data_ptr<float>(),
 dx.data_ptr<float>(),dh.data_ptr<float>(),dg.data_ptr<float>(),dr.data_ptr<float>(),di.data_ptr<float>(),
 {yptrs},src.numel(),mask.numel()!=0,hx,hh,hg,hr,hi,needx,needh,needg,needr,needi,needy);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
""")
    code, wrapper = "\n".join(lines).split("\nvoid coherent_mixed(", 1)
    # This key contains only autograd dependency metadata, never tensor values.
    # Compile out absent derivatives so their register lifetimes and branches do
    # not penalize the force-loss kernel. Every live target remains represented.
    flags = ("hx", "hh", "hg", "hr", "hi", "needx", "needh", "needg", "needr", "needi", "needy")
    for flag in flags:
        code = code.replace("bool "+flag, "bool unused_"+flag)
    constants = ";".join(f"constexpr bool {flag}={'true' if needed else 'false'}"
                         for flag, needed in zip(flags, dependencies))+";\n"
    return (code.replace(" const int64_t e=blockIdx.x", constants+" const int64_t e=blockIdx.x", 1)
            +"\nvoid coherent_mixed("+wrapper)


def _extension(p, projection, dependencies):
    key = (p["path_specs"], p["xdim"], p["odim"], p["c"], p["phase_heads"], p["phase_angular"],
           tuple(tuple(row) for row in projection["static_rows"]), p.get("mixed_threads", 0),
           p.get("mixed_grouped", False), p.get("mixed_compact", False), dependencies)
    if key not in _EXTENSIONS:
        from .coherent import _outputs
        from torch.utils.cpp_extension import load_inline
        tensors = 19+2*len(_outputs(p))
        declaration = "#include <torch/extension.h>\nvoid coherent_mixed("+",".join(
            ["torch::Tensor"]*tensors+["bool"]*11)+");\n"
        name = f"ictc_coherent_mixed_v1_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}_a{int(p['phase_angular'])}"
        if p.get("mixed_threads", 0):
            name += f"_t{p['mixed_threads']}"
        if p.get("mixed_grouped", False):
            name += "_grouped"
        if p.get("mixed_compact", False):
            name += "_compact"
        name += "_dep"+"".join("1" if need else "0" for need in dependencies)
        _EXTENSIONS[key] = load_inline(
            name=name,
            cpp_sources=declaration, cuda_sources=_source(p, projection, dependencies), functions=["coherent_mixed"],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False)
    return _EXTENSIONS[key]


def _physical_gradients(x, p, grads):
    from .coherent import _outputs
    return tuple(grad.transpose(-1, -2).contiguous() if grad is not None else x.new_zeros(
        x.shape[0], 3, 2*l+1, width//(2*l+1))
        for (l, _, width), grad in zip(_outputs(p), grads))


def _first(x, sh, w, g, r, i, src, dst, mask, p, projection, needed, grads):
    from .coherent import _extension as first_extension
    from mace_ictc.models.message_topology import topology
    if p.get('first_schedule', 'legacy') != 'legacy':
        from ._coherent_destination import adjoint
        return adjoint(x, sh, w, g, r, i, src, dst, mask, p, projection, needed, grads,
            splits=p.get('training_destination_first', 0) or 1, atomic_x=True,
            factored=True, coalesced=True, schedule=p['first_schedule'])
    if p.get("first_cache", False):
        from ._coherent_mixed_cache import mixed
        cached = {**p, "mixed_cache": p.get("mixed_cache", 0) or 16, "cache_first": True}
        return mixed(x, sh, w, g, r, i, src, dst, mask, cached, projection,
            (*needed, *(False for _ in grads)), grads, (None, None, g, None, None))[:5]
    if needed[0] and all(needed[2:]) and p.get("training_destination_first", 0):
        from ._coherent_destination import adjoint
        # Parameter-target backward can prune the geometric H target. Reuse
        # the same cached-cotangent first VJP rather than falling back to a
        # slower source schedule; its small unused H result is discarded.
        values = adjoint(x, sh, w, g, r, i, src, dst, mask, p, projection,
                         (needed[0], True, *needed[2:]), grads,
                         splits=p["training_destination_first"], atomic_x=True,
                         factored=True, coalesced=True)
        return tuple(value if need else None for value, need in zip(values, needed))
    if all(needed) and p.get("training_csr_splits", 0) > 1:
        from ._coherent_csr_split import adjoint
        return adjoint(x, sh, w, g, r, i, src, dst, mask, p, projection, needed,
                       grads, splits=p["training_csr_splits"])
    # X retains zeros for unused degrees/empty topology. Gate gradients have a
    # unique full writer; split edge-owned H/R/I gradients do not need clearing.
    split = p.get('training_split_first', False)
    result = tuple((torch.empty_like(v) if index == 2 or (split and index > 0) or
                    (not needed[0] and index > 0) else torch.zeros_like(v))
                   if need else x.new_empty(0)
                   for index, (v, need) in enumerate(zip((x, sh, g, r, i), needed)))
    if src.numel():
        physical = _physical_gradients(x, p, grads)
        extension = first_extension(p, projection)
        if needed[0] and not split:
            extension.coherent_csr_adjoint(x, sh, w, g, r, i, dst, mask, *physical,
                *topology(src, x.shape[0]), *result, *needed)
        else:
            extension.coherent_adjoint(x, sh, w, g, r, i, src, dst, mask, *physical,
                *(topology(src, x.shape[0]) if needed[0] else (src.new_empty(0), src.new_empty(0))),
                *result, *needed)
    return tuple(v if need else None for v, need in zip(result, needed))


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, w, g, r, i, src, dst, mask, p, projection, needed, *grads):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, sh, w, g, r, i, src, dst, mask, *grads)
        ctx.p, ctx.projection, ctx.needed = p, projection, needed
        return _first(x, sh, w, g, r, i, src, dst, mask, p, projection, needed, grads)

    @staticmethod
    def backward(ctx, *incoming):
        from .coherent import _outputs, _reference
        from mace_ictc.models.adjoint_requirements import required_gradients
        x, sh, w, g, r, i, src, dst, mask, *grads = ctx.saved_tensors
        p, projection = ctx.p, ctx.projection
        inputs = (x, sh, g, r, i, *grads)
        declared = (*ctx.needs_input_grad[:2], *ctx.needs_input_grad[3:6],
                    *ctx.needs_input_grad[12:])
        needs = required_gradients(inputs, declared)
        if not any(h is not None for h in incoming):
            return (None,)*(12+len(grads))
        higher = torch.is_grad_enabled()
        if not higher and p.get("mixed_cache", 0):
            from ._coherent_mixed_cache import mixed
            dx, dh, dg, dr, di, *dv = mixed(x, sh, w, g, r, i, src, dst, mask,
                p, projection, needs, grads, incoming)
            return dx, dh, None, dg, dr, di, None, None, None, None, None, None, *dv
        if not higher:
            empty = x.new_empty(0)
            groups = (len({(row[2], row[1]) for row in p["path_specs"]})
                      if p.get("mixed_grouped", False) else 1)
            # Only DX and output-node cotangents use floating atomics. The
            # edge-owned H/G/R/I arrays are completely overwritten by the kernel.
            result = tuple((torch.zeros_like(v) if index == 0 else
                            v.new_empty(v.shape[0]*groups, *v.shape[1:])
                            if index in (1, 3, 4) and groups > 1 else torch.empty_like(v)) if need else empty
                           for index, (v, need) in enumerate(zip((x, sh, g, r, i), needs[:5])))
            physical = _physical_gradients(x, p, grads)
            destination_owned = p.get("mixed_destination", False) and any(needs[5:])
            dy = tuple((torch.empty_like(v) if destination_owned else torch.zeros_like(v))
                       if any(needs[5:]) else empty for v in physical)
            tangents = tuple(empty if value is None else value.contiguous() for value in incoming)
            active = tuple(h is not None for h in incoming)
            dependencies = (*active, *needs[:5], any(needs[5:]) and not destination_owned)
            _extension(p, projection, dependencies).coherent_mixed(
                x, sh, w, g, r, i, src, dst, mask, *physical, *tangents, *result, *dy,
                *dependencies)
            if destination_owned:
                from ._coherent_training_jvp import destination_jvp
                destination_jvp(x, sh, w, g, r, i, src, dst, mask, p, projection,
                                tangents, active, dy)
            derivatives = tuple((v.reshape(v.shape[0]//groups, groups, *v.shape[1:]).sum(1)
                                 if groups > 1 and index in (1, 3, 4) else v) if need else None
                                for index, (v, need) in enumerate(zip(result, needs[:5])))
            derivatives += tuple(v.transpose(-1, -2) if need else None
                                 for v, need in zip(dy, needs[5:]))
        else:
            # Third/higher derivatives retain every live input and cotangent.
            with torch.enable_grad():
                local = tuple(v.clone() for v in inputs)
                output = _reference(local[0], local[1], w, *local[2:5], src, dst, mask, p, projection)
                parts = []
                for l, offset, width in _outputs(p):
                    value = output.reshape(x.shape[0], 3, p["odim"])[..., offset:offset+width]
                    value = value.reshape(x.shape[0], 3, -1, 2*l+1, p["c"]).transpose(-1, -2)
                    parts.append(value.reshape(x.shape[0], 3, width//(2*l+1), 2*l+1))
                first = torch.autograd.grad(parts, [v for v, need in zip(local[:5], ctx.needed) if need],
                    local[5:], create_graph=True, allow_unused=True)
                pairs = [(v, h) for v, h in zip(first, [h for h, need in zip(incoming, ctx.needed) if need])
                         if v is not None and h is not None and v.requires_grad]
                active = [v for v, need in zip(local, needs) if need]
                values = iter(torch.autograd.grad(tuple(v for v, _ in pairs), active,
                    tuple(h for _, h in pairs), create_graph=True, allow_unused=True))
                derivatives = tuple(next(values) if need else None for need in needs)
        dx, dh, dg, dr, di, *dv = derivatives
        return dx, dh, None, dg, dr, di, None, None, None, None, None, None, *dv


def coherent_adjoint(x, sh, w, g, r, i, src, dst, mask, p, projection, needed, grads):
    from .coherent import _outputs
    grads = tuple(grad if grad is not None else x.new_zeros(
        x.shape[0], 3, width//(2*l+1), 2*l+1)
        for (l, _, width), grad in zip(_outputs(p), grads))
    return _Adjoint.apply(x, sh, w, g, r, i, src, dst, mask, p, projection, needed, *grads)
