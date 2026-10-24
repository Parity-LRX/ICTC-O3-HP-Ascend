# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Native coherent TP whose radial cotangent is already one scalar per edge."""
import re
import torch
from .coherent import _outputs, _extension, _reference, _columns, _HEADERS, _csr_adjoint_source

_EXTENSIONS = {}


def _extension_distance(p, projection, gate_pipeline=False):
    key = (p["path_specs"], p["xdim"], p["c"], p["phase_heads"], p["phase_angular"],
           tuple(tuple(row) for row in projection["static_rows"]), gate_pipeline)
    if key in _EXTENSIONS:
        return _EXTENSIONS[key]
    from torch.utils.cpp_extension import load_inline
    from ._coherent_adjoint import source as edge_source
    width = p["p"]*p["c"]
    csr = _csr_adjoint_source(p, projection)
    csr = csr.replace("coherent_adjoint", "coherent_distance_csr")
    csr = csr.replace("const float* G,", "const float* G,const float* GP,")
    csr = csr.replace("torch::Tensor g,", "torch::Tensor g,torch::Tensor gp,")
    csr = re.sub(r"\bg\.data_ptr<float>\(\),", "g.data_ptr<float>(),gp.data_ptr<float>(),", csr)
    csr = csr.replace("bool needi)", "bool needi,int64_t BEGIN,int64_t END)")
    csr = csr.replace("needx,needh,needg,needr,needi);", "needx,needh,needg,needr,needi,BEGIN,END);")
    csr = csr.replace("for(int64_t index=PTR[src];index<PTR[src+1];++index)",
        "for(int64_t index=(PTR[src]>BEGIN?PTR[src]:BEGIN);index<PTR[src+1]&&index<END;++index)")
    csr = csr.replace("const float mask=masked?MASK[e]:1.f;", "const float mask=masked?MASK[e]:1.f;float radial=0.f;")
    csr = re.sub(r"if\(needg&&valid\)DG\[e\*\d+\+(\d+)\+channel\]=([^;\n]+);",
        lambda m: f"if(needg&&valid)radial+=({m[2]})*GP[(index-BEGIN)*{width}+{m[1]}+channel];", csr)
    csr = csr.replace("if(needh){\nfor(int d=16;d>0;d/=2)dh0",
        "if(needg){for(int d=16;d>0;d/=2)radial+=__shfl_down_sync(0xffffffff,radial,d);"
        "if(lane==0)atomicAdd(DG+e,radial);}\nif(needh){\nfor(int d=16;d>0;d/=2)dh0")
    csr = re.sub(r"(DX\[src\*\d+\+\d+\+channel\])=dx(\d+);", r"\1+=dx\2;", csr)
    if gate_pipeline:
        csr = csr.replace("G[e*", "G[(index-BEGIN)*")
    edge = edge_source(p, projection, _columns(p, projection), _outputs(p))
    edge = edge[:edge.index("__global__ void coherent_source_adjoint_kernel")]
    edge = edge.replace("coherent_edge_adjoint_kernel", "coherent_distance_edge_kernel")
    edge = edge.replace("const float* G,", "const float* G,const float* GP,")
    edge = edge.replace("bool needi)", "bool needi,int64_t BEGIN)")
    edge = edge.replace("const int64_t e=blockIdx.x,", "const int64_t e=blockIdx.x+BEGIN,")
    edge = edge.replace("const float mask=masked?MASK[e]:1.f;", "const float mask=masked?MASK[e]:1.f;float radial=0.f;")
    edge = re.sub(r"if\(needg\)DG\[e\*\d+\+(\d+)\+channel\]=([^;\n]+);",
        lambda m: f"if(needg)radial+=({m[2]})*GP[(e-BEGIN)*{width}+{m[1]}+channel];", edge)
    if gate_pipeline:
        edge = edge.replace("G[e*", "G[(e-BEGIN)*")
    threads = 32 if p["c"] <= 32 else 128
    warps = threads//32
    marker = "__shared__ float partial"
    edge = edge.replace(marker, f"""
__shared__ float radial_part[{warps}];
if(needg){{
 for(int d=16;d>0;d/=2)radial+=__shfl_down_sync(0xffffffff,radial,d);
 if(lane==0)radial_part[warp]=radial;
}}
__syncthreads();
if(needg&&threadIdx.x==0)DG[e]="""+"+".join(f"radial_part[{i}]" for i in range(warps))+";\n"+marker)
    outputs = _outputs(p)
    args = ",".join(f"torch::Tensor v{l}" for l, _, _ in outputs)
    pointers = ",".join(f"v{l}.data_ptr<float>()" for l, _, _ in outputs)
    edge += f"""
void coherent_distance_edge(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,torch::Tensor gp,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor dst,torch::Tensor mask,{args},
 torch::Tensor dh,torch::Tensor dg,torch::Tensor dr,torch::Tensor di,
 bool needh,bool needg,bool needr,bool needi,int64_t BEGIN,int64_t END) {{
 const c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
 if(END<=BEGIN)return;
 coherent_distance_edge_kernel<<<END-BEGIN,{threads},0,stream>>>(x.data_ptr<float>(),h.data_ptr<float>(),
 w.data_ptr<float>(),g.data_ptr<float>(),gp.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),
 src.data_ptr<int64_t>(),dst.data_ptr<int64_t>(),mask.data_ptr<float>(),{pointers},
 dh.data_ptr<float>(),dg.data_ptr<float>(),dr.data_ptr<float>(),di.data_ptr<float>(),src.numel(),
 mask.numel()!=0,needh,needg,needr,needi,BEGIN);C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""
    declarations = "#include <torch/extension.h>\nvoid coherent_distance_csr("+",".join(
        ["torch::Tensor"]*(16+len(outputs))+["bool"]*5+["int64_t"]*2)+");\n"
    declarations += "void coherent_distance_edge("+",".join(
        ["torch::Tensor"]*(14+len(outputs))+["bool"]*4+["int64_t"]*2)+");\n"
    name = f"ictc_radial_distance_v2_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}_a{int(p['phase_angular'])}"
    if gate_pipeline:name = name.replace('_v2_', '_v3_')+'_g1'
    _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declarations,
        cuda_sources=_HEADERS+csr+edge, functions=["coherent_distance_csr", "coherent_distance_edge"],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    return _EXTENSIONS[key]


class _Distance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, w, distance, real, imag, src, dst, mask, p, projection, radial, tile_edges,
                gate_pipeline):
        ctx.set_materialize_grads(False)
        hidden, tangent = radial.evaluate(distance)
        cached_gates = p.get('distance_cached_gates', False)
        gates = x.new_empty(0) if gate_pipeline and not cached_gates else torch.nn.functional.linear(hidden, radial.weight)
        saved_hidden = hidden if gate_pipeline and not cached_gates else x.new_empty(0)
        inline = gate_pipeline and p.get('distance_projection', 'gemm') in ('forward', 'both')
        projected_weight = radial.weight.T.contiguous() if inline else x.new_empty(0)
        ctx.save_for_backward(x, sh, w, distance, saved_hidden, tangent, gates, real, imag, src, dst, mask,
                              projected_weight, *tuple(radial.fc.parameters()))
        ctx.p, ctx.projection, ctx.radial, ctx.tile_edges = p, projection, radial, tile_edges
        ctx.gate_pipeline = gate_pipeline
        ctx.cached_gates = cached_gates
        out = tuple((x.new_empty if src.numel() and (not gate_pipeline or cached_gates) else x.new_zeros)(
            x.shape[0], 3, 2*l+1, width//(2*l+1)).transpose(-1, -2)
            for l, _, width in _outputs(p))
        if src.numel():
            from mace_ictc.models.message_topology import topology
            if inline:
                from ._radial_projection import forward_extension
                extension = forward_extension(p, projection, hidden.shape[1])
                extension.coherent_inline_radial_forward(x, sh, w, hidden, projected_weight,
                    real, imag, src, mask, *topology(dst, x.shape[0]), *out)
            elif gate_pipeline and not cached_gates:
                from ._radial_coherent import _chunk_extension
                ptr, order = topology(dst, x.shape[0])
                tile = min(tile_edges, src.numel())
                tile_hidden = hidden.new_empty(tile, hidden.shape[1])
                tile_gates = hidden.new_empty(tile, radial.weight.shape[0])
                extension = _chunk_extension(p, projection, True)
                for begin in range(0, src.numel(), tile):
                    end = min(begin+tile, src.numel())
                    torch.index_select(hidden, 0, order[begin:end], out=tile_hidden[:end-begin])
                    torch.mm(tile_hidden[:end-begin], radial.weight.T, out=tile_gates[:end-begin])
                    extension.coherent_radial_chunk_forward(x, sh, w, tile_gates, real, imag, src,
                        mask, ptr, order, *out, begin, end)
            else:
                _extension(p, projection).coherent_forward(x, sh, w, gates, real, imag, src, mask,
                                                           *topology(dst, x.shape[0]), *out)
        return out

    @staticmethod
    def backward(ctx, *grads):
        x, sh, w, distance, hidden, tangent, gates, real, imag, src, dst, mask = ctx.saved_tensors[:12]
        projected_weight = ctx.saved_tensors[12]
        p, projection = ctx.p, ctx.projection
        needed = tuple(ctx.needs_input_grad[i] for i in (0, 1, 3, 4, 5))
        if torch.is_grad_enabled():
            local = [v.clone() for v in (x, sh, distance, real, imag)]
            output = _reference(local[0], local[1], w, ctx.radial.reference(local[2]),
                *local[3:], src, dst, mask, p, projection)
            values, cotangents = [], []
            for (l, offset, width), grad in zip(_outputs(p), grads):
                if grad is not None:
                    value = output.reshape(x.shape[0], 3, p["odim"])[..., offset:offset+width]
                    value = value.reshape(x.shape[0], 3, -1, 2*l+1, p["c"]).transpose(-1, -2)
                    values.append(value.reshape(x.shape[0], 3, width//(2*l+1), 2*l+1))
                    cotangents.append(grad)
            active = [v for v, need in zip(local, needed) if need]
            derivatives = iter(torch.autograd.grad(values, active, cotangents,
                create_graph=True, allow_unused=True))
            result = [next(derivatives) if need else None for need in needed]
        else:
            empty = x.new_empty(0)
            result = [torch.zeros_like(v) if need else empty
                for v, need in zip((x, sh, distance, real, imag), needed)]
            dx, dh, dradial, dr, di = result
            if src.numel():
                from mace_ictc.models.message_topology import topology
                cotangents = [grad.transpose(-1, -2).contiguous() if grad is not None else x.new_zeros(
                    x.shape[0], 3, 2*l+1, width//(2*l+1))
                    for (l, _, width), grad in zip(_outputs(p), grads)]
                tile = min(src.numel(), ctx.tile_edges)
                destination = ctx.gate_pipeline and p.get('distance_destination', False) and all(needed[1:])
                inline = destination and p.get('distance_projection') == 'both'
                if destination:
                    from ._radial_destination import extension as destination_extension, workspace
                    extension = destination_extension(p, projection, bool(needed[0]),
                        hidden.shape[1] if inline else 0, cached_gates=ctx.cached_gates)
                    partials = workspace(x, p, projection, tile)
                else:
                    extension = _extension_distance(p, projection, ctx.gate_pipeline and not ctx.cached_gates)
                if ctx.cached_gates:
                    gp = x.new_empty(tile, ctx.radial.weight.shape[0])
                    direction = tangent.new_empty(tile, tangent.shape[1])
                elif ctx.gate_pipeline and not inline:
                    # One bounded GEMM generates both live gate values and
                    # their distance directional derivatives. No E*P*C gate
                    # or gate-cotangent tensor is retained by this path.
                    pair = x.new_empty(2*tile, hidden.shape[1])
                    projected = x.new_empty(2*tile, ctx.radial.weight.shape[0])
                elif not inline:
                    gp = x.new_empty(tile, gates.shape[1])
                if destination or needed[0]:
                    ptr, order = topology(dst if destination else src, x.shape[0])
                for begin in range(0, src.numel(), tile):
                    end = min(begin+tile, src.numel())
                    count = end-begin
                    if inline:
                        gates, gp = hidden, tangent
                    elif ctx.cached_gates:
                        if destination or needed[0]:
                            torch.index_select(tangent, 0, order[begin:end], out=direction[:count])
                        else:direction[:count].copy_(tangent[begin:end])
                        torch.mm(direction[:count], ctx.radial.weight.T, out=gp[:count])
                    elif ctx.gate_pipeline:
                        if destination or needed[0]:
                            torch.index_select(hidden, 0, order[begin:end], out=pair[:count])
                            torch.index_select(tangent, 0, order[begin:end], out=pair[count:2*count])
                        else:
                            pair[:count].copy_(hidden[begin:end])
                            pair[count:2*count].copy_(tangent[begin:end])
                        torch.mm(pair[:2*count], ctx.radial.weight.T, out=projected[:2*count])
                        gates, gp = projected[:count], projected[count:2*count]
                    else:
                        direction = tangent.index_select(0, order[begin:end]) if needed[0] else tangent[begin:end]
                        torch.mm(direction, ctx.radial.weight.T, out=gp[:count])
                    if destination:
                        extension.coherent_distance_destination(x, sh, w, gates, gp,
                            *([projected_weight] if inline else []), real, imag, src,
                            mask, *cotangents, ptr, order, dx, dh, dradial, dr, di, *partials, begin, end)
                    elif needed[0]:
                        extension.coherent_distance_csr(x, sh, w, gates, gp, real, imag, dst, mask,
                            *cotangents, ptr, order, dx, dh, dradial, dr, di, *needed, begin, end)
                    else:
                        extension.coherent_distance_edge(x, sh, w, gates, gp, real, imag, src, dst, mask,
                            *cotangents, dh, dradial, dr, di, *needed[1:], begin, end)
            result = [v if need else None for v, need in zip(result, needed)]
        dx, dh, dradial, dr, di = result
        return dx, dh, None, dradial, dr, di, None, None, None, None, None, None, None, None


def distance_coherent(owner, blocks, harmonics, distance, radial, src, dst, mask, real, imag, *,
                      tile_edges=65536, gate_pipeline=False):
    from .operators import _eager, _operand
    _eager(blocks[0])
    e = src.numel()
    p0, projection = owner.edge._prepare()
    phases = owner.heads*(owner.output_lmax+1 if owner.angular else 1)
    sample = owner.edge.tp.weight
    _operand(distance, (e,), sample, "distance")
    if not isinstance(tile_edges, int) or isinstance(tile_edges, bool) or tile_edges < 1:
        raise ValueError("Distance TP tile_edges must be a positive integer")
    if not isinstance(gate_pipeline, bool):
        raise TypeError('Distance TP gate_pipeline must be a boolean')
    if owner.distance_projection not in ('gemm', 'forward', 'both'):
        raise ValueError('Distance TP projection must be gemm, forward or both')
    if owner.distance_cached_gates and not (gate_pipeline and owner.distance_destination
                                            and owner.distance_projection == 'gemm'):
        raise ValueError('Cached radial gates require destination pipeline and GEMM projection')
    for name, value in (("src", src), ("dst", dst)):
        if value.shape != (e,) or value.dtype != torch.int64 or value.device != sample.device:
            raise ValueError(f"{name} must be int64 [E] on the execution device")
    if mask is not None:
        if mask.requires_grad or mask.shape not in ((e,), (e, 1)):
            raise ValueError("Coherent mask must be fixed [E] or [E,1]")
        _operand(mask, mask.shape, sample, "mask")
    if not radial.eligible():
        raise ValueError("Radial directional plan no longer frozen/eligible")
    for value in (real, imag):
        _operand(value, (e, phases, 1), sample, "phase coefficient")
    p = {**p0, "native_packed": True, "phase_heads": owner.heads,
         "phase_angular": owner.angular, "phase_count": phases,
         "distance_destination": owner.distance_destination,
         "distance_cached_gates": owner.distance_cached_gates,
         "distance_projection": owner.distance_projection}
    x = owner.edge.pack_input(blocks)
    sh = torch.cat([harmonics[l].squeeze(-2) for l in projection["keys"]], -1).contiguous()
    m = x.new_empty(0) if mask is None else mask.reshape(e).contiguous()
    output = _Distance.apply(x, sh, sample, distance, real.squeeze(-1).contiguous(),
        imag.squeeze(-1).contiguous(), src.contiguous(), dst.contiguous(), m, p, projection, radial,
        tile_edges, gate_pipeline)
    return {l: value for (l, _, _), value in zip(_outputs(p), output)}
