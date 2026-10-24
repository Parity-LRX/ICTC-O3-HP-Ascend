# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Live radial-output/TP adjoint fusion, including force-loss mixed VJPs.

The forward gate bank is reused only within this invocation. A force-loss
backward neither saves the wide first gate cotangent nor materializes its
wide tangent: the latter is evaluated inside the original ICTC CG kernels.
Bounded mixed gate-cotangent tiles are consumed immediately by IEEE FP32 GEMM.
Both radial banks and every original parameter remain live and trainable.
"""
import re

import torch

from mace_ictc.models.adjoint_requirements import required_gradients
from .coherent import _outputs, _reference
from ._coherent_training import _first, _physical_gradients

_EXTENSIONS = {}


def _replace(code, old, new):
    if code.count(old) != 1:
        raise RuntimeError(f"Radial training generator marker changed: {old!r}")
    return code.replace(old, new, 1)


def _tangent_source(hidden, width, active):
    has_z, has_b = active
    return f"""
__device__ __forceinline__ float ictc_live_radial_tangent(const float* Z,const float* TZ,
 const float* BT,const float* TBT,int64_t e,int col) {{
 float value=0.f;
 #pragma unroll 4
 for(int k=0;k<{hidden};++k){{
  {f'value=fmaf(TZ[e*{hidden}+k],BT[k*{width}+col],value);' if has_z else ''}
  {f'value=fmaf(Z[e*{hidden}+k],TBT[k*{width}+col],value);' if has_b else ''}
 }}
 return value;
}}
"""


def _inline_tangent(code, p):
    width = p["p"]*p["c"]
    pattern = rf"const float tg=hg\?TG\[e\*{width}\+(\d+)\+channel\]:0.f;"
    code, count = re.subn(pattern, lambda match:
        f"const float tg=ictc_live_radial_tangent(Z,TZ,BT,TBT,e,{match[1]}+channel);", code)
    if count != p["p"]:
        raise RuntimeError("Live radial tangent did not replace every CG path")
    return code


def _extension(p, projection, dependencies, hidden, active, first_gate, jvp=False):
    key = (p["path_specs"], p["c"], p["xdim"], p["odim"], p["phase_heads"],
           p["phase_angular"], tuple(tuple(row) for row in projection["static_rows"]),
           p.get("mixed_threads", 0), p.get("mixed_grouped", False),
           dependencies, hidden, active, first_gate, jvp)
    if key in _EXTENSIONS:
        return _EXTENSIONS[key]
    from torch.utils.cpp_extension import load_inline
    width = p["p"]*p["c"]
    if jvp:
        from ._coherent_training_jvp import _source
        code = _inline_tangent(_source(p, projection, dependencies), p)
        code = _replace(code, "const float* TI,", "const float* TI,const float* Z,const float* TZ,const float* BT,const float* TBT,")
        code = _replace(code, "torch::Tensor ti,", "torch::Tensor ti,torch::Tensor z,torch::Tensor tz,torch::Tensor bt,torch::Tensor tbt,")
        code = _replace(code, "ti.data_ptr<float>(),", "ti.data_ptr<float>(),z.data_ptr<float>(),tz.data_ptr<float>(),bt.data_ptr<float>(),tbt.data_ptr<float>(),")
        function = "coherent_destination_jvp"
        types = ["torch::Tensor"]*(19+len(_outputs(p)))
    else:
        from ._coherent_training import _source
        p = {**p, "mixed_compact": True}
        code = _inline_tangent(_source(p, projection, dependencies), p)
        code = _replace(code, "float* DG,", "float* DG,float* FG,const float* Z,const float* TZ,const float* BT,const float* TBT,")
        code = _replace(code, "int64_t E,bool masked", "int64_t E,int64_t BEGIN,int64_t END,bool masked")
        code = _replace(code, "e=blockIdx.x,src=SRC[e]", "e=BEGIN+blockIdx.x,src=SRC[e]")
        code = code.replace("DG[e*", "DG[(e-BEGIN)*")
        if first_gate:
            code = code.replace("float dg_path=0.f,dr_path=0.f,di_path=0.f;",
                                "float fg_path=0.f,dg_path=0.f,dr_path=0.f,di_path=0.f;")
            code = code.replace("dg_path+=tu*vef+u*tv;", "dg_path+=tu*vef+u*tv;fg_path+=u*vef;")
            pattern = rf"(if\(needg\)DG\[\(e-BEGIN\)\*{width}\+(\d+)\+channel\]=scale\*dg_path;)"
            code, count = re.subn(pattern, lambda match: match[1]+
                f"FG[(e-BEGIN)*{width}+{match[2]}+channel]=scale*fg_path;", code)
            if count != p["p"]:
                raise RuntimeError("Missing bounded first gate-cotangent writer")
        code = _replace(code, "bool hx,bool hh,bool hg,bool hr,bool hi,\n bool needx", "torch::Tensor fg,torch::Tensor z,torch::Tensor tz,torch::Tensor bt,torch::Tensor tbt,int64_t BEGIN,int64_t END,\n bool hx,bool hh,bool hg,bool hr,bool hi,\n bool needx")
        code = _replace(code, "if(!src.numel())return;", "if(END<=BEGIN)return;")
        code = _replace(code, "dim3(src.numel(),", "dim3(END-BEGIN,")
        code = _replace(code, "dg.data_ptr<float>(),", "dg.data_ptr<float>(),fg.data_ptr<float>(),z.data_ptr<float>(),tz.data_ptr<float>(),bt.data_ptr<float>(),tbt.data_ptr<float>(),")
        code = _replace(code, "src.numel(),mask.numel()!=0,hx", "src.numel(),BEGIN,END,mask.numel()!=0,hx")
        function = "coherent_mixed"
        types = ["torch::Tensor"]*(24+2*len(_outputs(p)))+["int64_t"]*2+["bool"]*11
    declaration = "#include <torch/extension.h>\nvoid "+function+"("+",".join(types)+");\n"
    name = (f"ictc_radial_training_v1_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}"
            f"_a{int(p['phase_angular'])}_k{hidden}_t{p.get('mixed_threads', 0)}"
            f"_g{int(p.get('mixed_grouped', False))}_f{int(first_gate)}_j{int(jvp)}_"
            +"".join(str(int(flag)) for flag in (*dependencies, *active)))
    code = _tangent_source(hidden, width, active)+code
    # The device helper needs CUDA declarations before float intrinsics.
    from .coherent import _HEADERS
    _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration,
        cuda_sources=_HEADERS+code, functions=[function], extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    return _EXTENSIONS[key]


def _first_radial(x, sh, z, bank, r, i, g, w, src, dst, mask, p, projection, needed, grads, tile):
    if p.get('radial_first_bounded', False):
        from ._radial_first_training import first
        return first(x, sh, z, bank, r, i, g, w, src, dst, mask, p, projection, needed, grads, tile)
    targets = (needed[0], needed[1], needed[2] or needed[3], needed[4], needed[5])
    dx, dh, dg, dr, di = _first(x, sh, w, g, r, i, src, dst, mask, p, projection, targets, grads)
    dz = dg @ bank if needed[2] else None
    db = dg.T @ z if needed[3] else None
    return dx, dh, dz, db, dr, di


class _RadialAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, z, bank, r, i, g, w, src, dst, mask, p, projection, needed, tile, *grads):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, sh, z, bank, r, i, g, w, src, dst, mask, *grads)
        ctx.p, ctx.projection, ctx.needed, ctx.tile = p, projection, needed, tile
        return _first_radial(x, sh, z, bank, r, i, g, w, src, dst, mask, p, projection, needed, grads, tile)

    @staticmethod
    def backward(ctx, *incoming):
        x, sh, z, bank, r, i, gates, w, src, dst, mask, *grads = ctx.saved_tensors
        p, projection = ctx.p, ctx.projection
        inputs = (x, sh, z, bank, r, i, *grads)
        needs = required_gradients(inputs, (*ctx.needs_input_grad[:6], *ctx.needs_input_grad[15:]))
        if not any(value is not None for value in incoming):
            return (None,)*(15+len(grads))
        if torch.is_grad_enabled():
            with torch.enable_grad():
                local = tuple(value.clone() for value in inputs)
                output = _reference(local[0], local[1], w,
                    torch.nn.functional.linear(local[2], local[3]), *local[4:6],
                    src, dst, mask, p, projection)
                parts = []
                for l, offset, width in _outputs(p):
                    value = output.reshape(x.shape[0], 3, p["odim"])[..., offset:offset+width]
                    value = value.reshape(x.shape[0], 3, -1, 2*l+1, p["c"]).transpose(-1, -2)
                    parts.append(value.reshape(x.shape[0], 3, width//(2*l+1), 2*l+1))
                first = torch.autograd.grad(parts, [value for value, need in zip(local[:6], ctx.needed) if need],
                    local[6:], create_graph=True, allow_unused=True)
                pairs = [(value, tangent) for value, tangent in zip(first,
                    [value for value, need in zip(incoming, ctx.needed) if need])
                    if value is not None and tangent is not None and value.requires_grad]
                values = iter(torch.autograd.grad(tuple(value for value, _ in pairs),
                    [value for value, need in zip(local, needs) if need],
                    tuple(tangent for _, tangent in pairs), create_graph=True, allow_unused=True))
                result = tuple(next(values) if need else None for need in needs)
        else:
            if p.get("radial_cached", False):
                from ._radial_cached_training import mixed
                result = mixed(x, sh, z, bank, r, i, gates, w, src, dst, mask, p, projection,
                    needs, grads, incoming, ctx.tile)
                return (*result[:6], None, None, None, None, None, None, None, None, None, *result[6:])
            empty = x.new_empty(0)
            tx, th, tz, tb, tr, ti = tuple(empty if value is None else value.contiguous() for value in incoming)
            active = tuple(value is not None for value in incoming)
            dependency = (active[0], active[1], active[2] or active[3], active[4], active[5])
            need_gate = needs[2] or needs[3]
            first_gate = (needs[3] and active[2]) or (needs[2] and active[3])
            flags = (*dependency, needs[0], needs[1], need_gate, needs[4], needs[5], False)
            groups = (len({(row[2], row[1]) for row in p["path_specs"]}) if p.get("mixed_grouped", False) else 1)
            dx = torch.zeros_like(x) if needs[0] else empty
            edge = tuple(value.new_empty(value.shape[0]*groups, *value.shape[1:]) if need else empty
                         for value, need in zip((sh, r, i), (needs[1], needs[4], needs[5])))
            dz = torch.empty_like(z) if needs[2] else None
            db = torch.zeros_like(bank) if needs[3] else None
            physical = _physical_gradients(x, p, grads)
            dy = tuple(torch.empty_like(value) if any(needs[6:]) else empty for value in physical)
            tile = min(ctx.tile, max(1, src.numel()))
            dg = z.new_empty(tile, bank.shape[0]) if need_gate else empty
            fg = z.new_empty(tile, bank.shape[0]) if first_gate else empty
            bt = bank.T.contiguous()
            tbt = tb.T.contiguous() if active[3] else empty
            mixed = _extension(p, projection, flags, z.shape[1], active[2:4], first_gate)
            for begin in range(0, src.numel(), tile):
                end = min(begin+tile, src.numel())
                count = end-begin
                mixed.coherent_mixed(x, sh, w, gates, r, i, src, dst, mask, *physical,
                    tx, th, empty, tr, ti, dx, edge[0], dg, edge[1], edge[2],
                    *(empty for _ in physical), fg, z, tz, bt, tbt, begin, end, *flags)
                if needs[2]:
                    torch.mm(dg[:count], bank, out=dz[begin:end])
                    if active[3]:
                        dz[begin:end].addmm_(fg[:count], tb)
                if needs[3]:
                    db.addmm_(dg[:count].T, z[begin:end])
                    if active[2]:
                        db.addmm_(fg[:count].T, tz[begin:end])
            if any(needs[6:]):
                from mace_ictc.models.message_topology import topology
                _extension(p, projection, dependency, z.shape[1], active[2:4], False, True).coherent_destination_jvp(
                    x, sh, w, gates, r, i, src, mask, *topology(dst, x.shape[0]),
                    tx, th, empty, tr, ti, z, tz, bt, tbt, *dy)
            dh, dr, di = tuple((value.reshape(-1, groups, *original.shape[1:]).sum(1)
                if groups > 1 else value) if need else None
                for value, original, need in zip(edge, (sh, r, i), (needs[1], needs[4], needs[5])))
            result = (dx if needs[0] else None, dh, dz, db, dr, di,
                      *(value.transpose(-1, -2) if need else None for value, need in zip(dy, needs[6:])))
        return (*result[:6], None, None, None, None, None, None, None, None, None, *result[6:])


class _RadialTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, z, bank, real, imag, w, src, dst, mask, p, projection, tile):
        from .coherent import _extension
        from mace_ictc.models.message_topology import topology
        ctx.set_materialize_grads(False)
        gates = torch.nn.functional.linear(z, bank)
        ctx.save_for_backward(x, sh, z, bank, real, imag, gates, w, src, dst, mask)
        ctx.p, ctx.projection, ctx.tile = p, projection, tile
        out = tuple((x.new_empty if src.numel() else x.new_zeros)(
            x.shape[0], 3, 2*l+1, width//(2*l+1)).transpose(-1, -2)
            for l, _, width in _outputs(p))
        if src.numel():
            _extension(p, projection).coherent_forward(x, sh, w, gates, real, imag, src, mask,
                *topology(dst, x.shape[0]), *out)
        return out

    @staticmethod
    def backward(ctx, *grads):
        x, sh, z, bank, real, imag, gates, w, src, dst, mask = ctx.saved_tensors
        needed = required_gradients((x, sh, z, bank, real, imag), ctx.needs_input_grad[:6])
        grads = tuple(grad if grad is not None else x.new_zeros(
            x.shape[0], 3, width//(2*l+1), 2*l+1)
            for (l, _, width), grad in zip(_outputs(ctx.p), grads))
        result = _RadialAdjoint.apply(x, sh, z, bank, real, imag, gates, w, src, dst, mask,
            ctx.p, ctx.projection, needed, ctx.tile, *grads)
        return (*result, None, None, None, None, None, None, None)


def radial_training(owner, blocks, harmonics, hidden, bank, src, dst, mask, real, imag, tile_edges=8192):
    from .operators import _eager, _operand
    _eager(blocks[0])
    if not isinstance(tile_edges, int) or isinstance(tile_edges, bool) or tile_edges < 1:
        raise ValueError("Live radial training tile_edges must be positive")
    sample = owner.edge.tp.weight
    if bank.ndim != 2 or not 1 <= bank.shape[1] <= 128:
        raise ValueError("Live radial bank requires 1..128 input channels")
    e = src.numel()
    _operand(bank, (owner.edge.gate_width, bank.shape[1]), sample, "live radial bank")
    _operand(hidden, (e, bank.shape[1]), sample, "live radial hidden")
    for value in (src, dst):
        if value.shape != (e,) or value.dtype != torch.int64 or value.device != sample.device:
            raise ValueError("Live radial topology requires int64 [E] on the execution device")
    phases = owner.heads*(owner.output_lmax+1 if owner.angular else 1)
    for value in (real, imag):
        _operand(value, (e, phases, 1), sample, "phase coefficient")
    if mask is not None:
        if mask.requires_grad or mask.shape not in ((e,), (e, 1)):
            raise ValueError("Live radial mask must be fixed [E] or [E,1]")
        _operand(mask, mask.shape, sample, "mask")
    p0, projection = owner.edge._prepare()
    p = {**p0, "native_packed": True, "phase_heads": owner.heads,
         "phase_angular": owner.angular, "phase_count": phases,
         "mixed_threads": owner.mixed_threads, "mixed_grouped": owner.mixed_grouped,
         "mixed_compact": True, "training_destination_first": owner.training_destination_first,
         "training_csr_splits": owner.training_csr_splits,
         "training_split_first": owner.training_split_first}
    p["first_cache"] = owner.first_cache
    p["first_schedule"] = owner.first_schedule
    p["radial_cached"] = getattr(owner, "radial_cached", False)
    p["radial_first_bounded"] = getattr(owner, "radial_first_bounded", False)
    p["mixed_fuse_jvp"] = owner.mixed_fuse_jvp
    if p["radial_cached"]:
        if p["c"] % 32:
            raise ValueError("Cached live radial training requires channel multiples of 32")
        p["mixed_cache"] = owner.mixed_cache or 16
    x = owner.edge.pack_input(blocks)
    sh = torch.cat([harmonics[l].squeeze(-2) for l in projection["keys"]], -1).contiguous()
    _operand(sh, (e, projection["hdim"]), sample, "harmonics")
    m = x.new_empty(0) if mask is None else mask.reshape(e).contiguous()
    out = _RadialTP.apply(x, sh, hidden.contiguous(), bank.contiguous(),
        real.squeeze(-1).contiguous(), imag.squeeze(-1).contiguous(), sample,
        src.contiguous(), dst.contiguous(), m, p, projection, tile_edges)
    return {l: value for (l, _, _), value in zip(_outputs(p), out)}
