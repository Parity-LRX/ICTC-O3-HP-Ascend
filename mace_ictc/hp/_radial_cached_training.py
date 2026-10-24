# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded GEMM radial tangents consumed by cached live TP mixed adjoints.

The chunk order is destination CSR order, so each node's cotangent stays cached
while gate tangents and gate VJPs have a bounded edge footprint. Every tile is
projected back into the live hidden/weight gradients before its storage is reused.
"""
import re
import torch

_EXTENSIONS = {}


def _once(code, old, new):
    if code.count(old) != 1:
        raise RuntimeError(f"Bounded radial generator marker changed: {old!r}")
    return code.replace(old, new, 1)


def _extension(p, projection, dependencies, first_gate=False, jvp=False):
    from .coherent import _outputs
    from torch.utils.cpp_extension import load_inline
    key = (p["path_specs"], p["c"], p["xdim"], p["odim"], p["phase_heads"], p["phase_angular"],
           p["mixed_cache"], tuple(tuple(row) for row in projection["static_rows"]),
           dependencies, first_gate, jvp, p.get('mixed_fuse_jvp', False), p.get('mixed_grouped', False))
    if key in _EXTENSIONS:
        return _EXTENSIONS[key]
    width = p["p"]*p["c"]
    if jvp:
        from ._coherent_training_jvp import _source
        code = _source(p, projection, dependencies)
        # The owner task's divisor is structural and independent of graph size.
        code, count = re.subn(r"(const int64_t dst=task/\d+;)", r"\1\n"
            "const int64_t edge_lo=PTR[dst]>BEGIN?PTR[dst]:BEGIN;"
            "const int64_t edge_hi=PTR[dst+1]<END?PTR[dst+1]:END;if(edge_lo>=edge_hi)return;", code)
        if count != 1:
            raise RuntimeError("Missing bounded JVP destination owner")
        code = _once(code, "int64_t N,bool masked)", "int64_t N,int64_t BEGIN,int64_t END,bool masked)")
        last = f"torch::Tensor out{_outputs(p)[-1][0]})"
        code = _once(code, last, last[:-1]+",int64_t BEGIN,int64_t END)")
        code = _once(code, "x.size(0),mask.numel()!=0);", "x.size(0),BEGIN,END,mask.numel()!=0);")
        code, count = re.subn(r"(O\d+\[[^\n]+?\])=([ari]\d+_\d+);", r"\1+=\2;", code)
        if not count:
            raise RuntimeError("Missing bounded output-cotangent accumulation")
        function = "coherent_destination_jvp"
        types = ["torch::Tensor"]*(15+len(_outputs(p)))+["int64_t"]*2
    else:
        from ._coherent_mixed_cache import _source, _groups, _specialize_groups
        code = _source(p, projection, dependencies)
        slots = len(_groups(p))*(p["c"]//32)
        marker = f"const int64_t dst=task/{slots};"
        code = _once(code, marker, marker+"\nconst int64_t edge_lo=PTR[dst]>BEGIN?PTR[dst]:BEGIN;"
            "const int64_t edge_hi=PTR[dst+1]<END?PTR[dst+1]:END;if(edge_lo>=edge_hi)return;")
        code = _once(code, "int64_t N,bool masked)", "int64_t N,int64_t BEGIN,int64_t END,bool masked)")
        code = _once(code, "float* __restrict__ DG,", "float* __restrict__ DG,float* __restrict__ FG,")
        code = _once(code, "torch::Tensor dg,", "torch::Tensor dg,torch::Tensor fg,")
        code = _once(code, "bool hx,bool hh,bool hg,bool hr,bool hi,bool needx", "int64_t BEGIN,int64_t END,\n bool hx,bool hh,bool hg,bool hr,bool hi,bool needx")
        code = _once(code, "dg.data_ptr<float>(),", "dg.data_ptr<float>(),fg.data_ptr<float>(),")
        code = _once(code, "if(!src.numel())return;", "if(END<=BEGIN)return;")
        code = _once(code, "x.size(0),mask.numel()!=0);", "x.size(0),BEGIN,END,mask.numel()!=0);")
        code = code.replace("DG[e*", "DG[(index-BEGIN)*")
        # Kernel writers use global CSR indices; the reduction's own e is
        # already tile-local. Shift only writers, never reduction reads.
        marker = "__global__ void coherent_cached_mixed_reduce"
        writer, reduction = code.split(marker, 1)
        for target in ("PH", "PR", "PI"):
            writer = writer.replace(target+"[(e*", target+"[((index-BEGIN)*")
        code = writer+marker+reduction
        if first_gate:
            code = code.replace("float dg_path=0.f,dr_path=0.f,di_path=0.f;", "float fg_path=0.f,dg_path=0.f,dr_path=0.f,di_path=0.f;")
            code = code.replace("dg_path+=tu*vef+u*tv;", "dg_path+=tu*vef+u*tv;fg_path+=u*vef;")
            pattern = rf"(if\(needg\)DG\[\(index-BEGIN\)\*{width}\+(\d+)\+channel\]=scale\*dg_path;)"
            code, count = re.subn(pattern, lambda match: match[1]+
                f"FG[(index-BEGIN)*{width}+{match[2]}+channel]=scale*fg_path;", code)
            if count != p["p"]:
                raise RuntimeError("Missing bounded first gate VJP")
        code = _once(code, "int64_t E)", "const int64_t* __restrict__ ORDER,int64_t BEGIN,int64_t E)")
        for target, stride in (("DH", projection["hdim"]), ("DR", p["phase_count"]), ("DI", p["phase_count"])):
            code = _once(code, target+"[index]=", f"{target}[ORDER[BEGIN+index/{stride}]*{stride}+index%{stride}]=")
        code = code.replace("(src.numel()*", "((END-BEGIN)*")
        code = _once(code, ",src.numel());", ",order.data_ptr<int64_t>(),BEGIN,END-BEGIN);")
        function = "coherent_cached_mixed"
        fused_jvp = bool(p.get('mixed_fuse_jvp', False))
        if fused_jvp:
            code, count = re.subn(r'(DY\d+\[[^\n]+?\])=(y\d+_\d+_\d+);', r'\1+=\2;', code)
            if not count:
                raise RuntimeError('Missing bounded fused output-cotangent accumulation')
        if p.get('mixed_grouped', False):
            code = _specialize_groups(code, p)
        types = ["torch::Tensor"]*(24+len(_outputs(p))*(2 if fused_jvp else 1))+["int64_t"]*2+["bool"]*(11 if fused_jvp else 10)
    code = code.replace("index=PTR[dst];index<PTR[dst+1];++index", "index=edge_lo;index<edge_hi;++index")
    code = code.replace("TG[e*", "TG[(index-BEGIN)*")
    name = (f"ictc_radial_cached_v4_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}"
            f"_a{int(p['phase_angular'])}_g{p['mixed_cache']}_f{int(first_gate)}_j{int(jvp)}_"
            +"".join(str(int(value)) for value in dependencies))
    if p.get('mixed_fuse_jvp', False):name += '_fusedjvp'
    if p.get('mixed_grouped', False):name += '_splitgroups'
    declaration = "#include <torch/extension.h>\nvoid "+function+"("+",".join(types)+");\n"
    _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration, cuda_sources=code,
        functions=[function], extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    return _EXTENSIONS[key]


def mixed(x, sh, z, bank, r, i, gates, w, src, dst, mask, p, projection, needs, grads, incoming, tile_edges):
    from ._coherent_training import _physical_gradients
    from ._coherent_mixed_cache import _groups
    from mace_ictc.models.message_topology import topology
    active = tuple(value is not None for value in incoming)
    empty = x.new_empty(0)
    tx, th, tz, tb, tr, ti = tuple(empty if value is None else value.contiguous() for value in incoming)
    dependency = (active[0], active[1], active[2] or active[3], active[4], active[5])
    need_gate = needs[2] or needs[3]
    first_gate = (needs[3] and active[2]) or (needs[2] and active[3])
    flags = (*dependency, needs[0], needs[1], need_gate, needs[4], needs[5])
    fuse_jvp = bool(p.get('mixed_fuse_jvp', False))
    if fuse_jvp:
        flags += (any(needs[6:]),)
    dx = torch.zeros_like(x) if needs[0] else empty
    edge = tuple(torch.empty_like(value) if need else empty
        for value, need in zip((sh, r, i), (needs[1], needs[4], needs[5])))
    dz = torch.empty_like(z) if needs[2] else None
    db = torch.zeros_like(bank) if needs[3] else None
    physical = _physical_gradients(x, p, grads)
    dy = tuple(torch.zeros_like(value) if any(needs[6:]) else empty for value in physical)
    ptr, order = topology(dst, x.shape[0])
    tile = min(tile_edges, max(1, src.numel()))
    dg = z.new_empty(tile, bank.shape[0]) if need_gate else empty
    fg = z.new_empty(tile, bank.shape[0]) if first_gate else empty
    slots = len(_groups(p))*(p["c"]//32)
    partial = tuple(value.new_empty(tile, slots, *value.shape[1:]) if need else empty
        for value, need in zip((sh, r, i), (needs[1], needs[4], needs[5])))
    mixed_extension = _extension(p, projection, flags, first_gate)
    jvp_extension = _extension(p, projection, dependency, False, True) if any(needs[6:]) and not fuse_jvp else None
    for begin in range(0, src.numel(), tile):
        end = min(begin+tile, src.numel())
        count, index = end-begin, order[begin:end]
        local_z = z.index_select(0, index) if needs[3] or active[3] else None
        local_tz = tz.index_select(0, index) if active[2] else None
        tg = torch.nn.functional.linear(local_tz, bank) if active[2] else None
        if active[3]:
            term = torch.nn.functional.linear(local_z, tb)
            tg = term if tg is None else tg+term
        mixed_extension.coherent_cached_mixed(x, sh, w, gates, r, i, src, mask,
            *physical, *(dy if fuse_jvp else ()), ptr, order, tx, th, empty if tg is None else tg, tr, ti,
            dx, edge[0], dg, fg, edge[1], edge[2], *partial, begin, end, *flags)
        if needs[2]:
            local = dg[:count] @ bank
            if active[3]:
                local.addmm_(fg[:count], tb)
            dz.index_copy_(0, index, local)
        if needs[3]:
            db.addmm_(dg[:count].T, local_z)
            if active[2]:
                db.addmm_(fg[:count].T, local_tz)
        if jvp_extension is not None:
            jvp_extension.coherent_destination_jvp(x, sh, w, gates, r, i, src, mask,
                ptr, order, tx, th, empty if tg is None else tg, tr, ti, *dy, begin, end)
    return (dx if needs[0] else None, edge[0] if needs[1] else None, dz, db,
            edge[1] if needs[4] else None, edge[2] if needs[5] else None,
            *(value.transpose(-1, -2) if need else None for value, need in zip(dy, needs[6:])))
