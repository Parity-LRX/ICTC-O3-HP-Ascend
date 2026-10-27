"""Bound inference message scratch by recomputing one edge chunk per adjoint."""
import torch
from ictc_o3.ascendc.graph import graph_message as original_native_graph
from ictc_o3.ascendc.graph import _validate
from ictc_o3.backends.ascend.channel_message import graph_message as fast_graph

class _Message(torch.autograd.Function):
    @staticmethod
    def forward(ctx,owner,fkeys,hkeys,chunk,projection,*tensors):
        nf,nh=len(fkeys),len(hkeys);features=dict(zip(fkeys,tensors[:nf]));harmonics=dict(zip(hkeys,tensors[nf:nf+nh]));gates,src,dst,mask=tensors[nf+nh:]
        ctx.weight_reference=owner.weight
        ctx.projection_reference=None if projection is None else (projection.weight,projection.bias)
        constants=() if projection is None else (projection.weight,projection.bias)
        ctx.meta=(owner,fkeys,hkeys,chunk,projection)
        ctx.save_for_backward(owner.weight,*constants,*tensors);ctx.set_materialize_grads(False)
        result=None
        for begin in range(0,src.numel(),chunk):
            end=min(src.numel(),begin+chunk)
            part=gates[begin:end] if projection is None else projection(gates[begin:end])
            parts=fast_graph(owner,features,{k:v[begin:end] for k,v in harmonics.items()},part,src[begin:end],dst[begin:end],None if mask is None else mask[begin:end],edge_chunk_size=chunk)
            if result is None:result=parts
            else:
                for key,value in parts.items():result[key].add_(value)
            torch.npu.current_stream(gates.device).synchronize()
            del parts,part
        ctx.keys=tuple(result)
        return tuple(result[k] for k in ctx.keys)
    @staticmethod
    def backward(ctx,*outputs):
        owner,fkeys,hkeys,chunk,projection=ctx.meta;saved_weight,*tensors=ctx.saved_tensors
        if projection is not None:
            saved_projection,saved_bias,*tensors=tensors
            if any(a is not b for a,b in zip(ctx.projection_reference,(projection.weight,projection.bias))):
                raise RuntimeError('Radial projection coefficients were replaced between forward and backward')
        nf,nh=len(fkeys),len(hkeys);ng=nf+nh;gates,src,dst,mask=tensors[ng:];needed=ctx.needs_input_grad[5:];result=[None]*len(tensors)
        if owner.weight is not ctx.weight_reference:
            raise RuntimeError('Tensor-product coefficients were replaced between forward and backward')
        if torch.is_grad_enabled():
            # Preserve native recursive higher derivatives; capacity inference changes only
            # the ordinary first-order inference adjoint's storage schedule.
            with torch.enable_grad():
                aliases=[v.view_as(v) for v in tensors[:ng+1]]
                part=aliases[ng] if projection is None else projection(aliases[ng])
                values=original_native_graph(owner,dict(zip(fkeys,aliases[:nf])),dict(zip(hkeys,aliases[nf:ng])),part,src,dst,mask,edge_chunk_size=None)
                selected=[i for i in range(ng+1) if needed[i]]
                pairs=[(values[k],g) for k,g in zip(ctx.keys,outputs) if g is not None and values[k].requires_grad]
                grads=torch.autograd.grad(tuple(v for v,g in pairs),tuple(aliases[i] for i in selected),tuple(g for v,g in pairs),create_graph=True,allow_unused=True)
                for i,g in zip(selected,grads):result[i]=g
            return (None,None,None,None,None,*result)
        for i in range(ng+1):
            if needed[i]:result[i]=torch.zeros_like(tensors[i]) if i<nf else torch.empty_like(tensors[i])
        for begin in range(0,src.numel(),chunk):
            end=min(src.numel(),begin+chunk)
            with torch.enable_grad():
                values=[(v if i<nf else v[begin:end]).detach().requires_grad_(needed[i]) for i,v in enumerate(tensors[:ng+1])]
                part=values[ng] if projection is None else projection(values[ng])
                pieces=fast_graph(owner,dict(zip(fkeys,values[:nf])),dict(zip(hkeys,values[nf:ng])),part,src[begin:end],dst[begin:end],None if mask is None else mask[begin:end],edge_chunk_size=chunk)
                selected=[i for i in range(ng+1) if needed[i]];pairs=[(pieces[k],g) for k,g in zip(ctx.keys,outputs) if g is not None and pieces[k].requires_grad]
                grads=torch.autograd.grad(tuple(v for v,g in pairs),tuple(values[i] for i in selected),tuple(g for v,g in pairs),allow_unused=True)
            for i,g in zip(selected,grads):
                if g is None:
                    if i>=nf:result[i][begin:end].zero_()
                elif i<nf:result[i].add_(g)
                else:result[i][begin:end].copy_(g)
            torch.npu.current_stream(gates.device).synchronize()
            del pieces,values,grads,pairs,part
        return (None,None,None,None,None,*result)

def _automatic_chunk(owner, sample, features, harmonics, gates, edges):
    from .tensor_product import _edge_working_bytes
    from .runtime import _device_memory
    total=_device_memory(sample.device)
    index=torch.npu.current_device() if sample.device.index is None else sample.device.index
    allocated=int(torch.ops.ictc_ascendc.allocated_bytes(index))
    gradient=gates.numel()*gates.element_size() if gates.requires_grad else 0
    from .tensor_product import _destination_nodes
    width=sum(count*(2*degree+1) for degree,count in owner.path_counts_by_l.items())
    node_bytes=_destination_nodes(owner,sample.shape[0])*sample.shape[1]*width*sample.element_size()
    # Compact outputs, layout restoration and their adjoints can coexist.
    budget=min(total//8,max(0,total-allocated-gradient-6*node_bytes)//2)
    per_edge=_edge_working_bytes(owner,sample,features,harmonics)
    if budget<max(per_edge,2*node_bytes):return None
    chunk=min(edges,1048576,budget//per_edge)
    return chunk//128*128 if chunk>=128 else chunk


def graph_message(owner,features,harmonics,gates,src,dst,mask=None,*,edge_chunk_size=None):
    sample=next(iter(features.values()))
    from .runtime import require_npu_fp32
    require_npu_fp32(owner.weight,gates,*features.values(),*harmonics.values())
    _validate(src,dst,sample.shape[0],sample.device)
    edges,channels=src.numel(),sample.shape[1]
    if gates.shape not in ((edges,owner.num_paths*channels),(edges,owner.num_paths,channels)):
        raise ValueError('External graph gates must have shape [E,P*C] or [E,P,C]')
    if any(value.shape[0]!=edges for value in harmonics.values()):
        raise ValueError('Harmonic row counts must match graph edges')
    if mask is not None and (mask.device!=sample.device or mask.shape not in ((edges,),(edges,1))):
        raise ValueError('Expected an edge mask [E] or [E,1] on the feature device')
    # Model training retains the established native higher-order schedule.
    if (getattr(owner,'_ascend_tp_schedule',None)=='gemm' or owner.weight.requires_grad
                    or mask is not None and mask.requires_grad or not src.numel()):
        return original_native_graph(owner,features,harmonics,gates,src,dst,mask,edge_chunk_size=edge_chunk_size)
    chunk=(_automatic_chunk(owner,sample,features,harmonics,gates,src.numel())
                                if edge_chunk_size is None else edge_chunk_size)
    if chunk is None:return original_native_graph(owner,features,harmonics,gates,src,dst,mask,edge_chunk_size=edge_chunk_size)
    if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<1:raise ValueError('Edge chunk size must be positive')
    if mask is not None:mask=mask.to(dtype=sample.dtype)
    fkeys,hkeys=tuple(sorted(features)),tuple(sorted(harmonics))
    values=_Message.apply(owner,fkeys,hkeys,chunk,None,*(features[k] for k in fkeys),*(harmonics[k] for k in hkeys),gates,src,dst,mask)
    return dict(zip(range(owner.lmax+1),values))


def radial_graph_message(owner,features,harmonics,radial,edge_features,src,dst,mask=None,*,edge_chunk_size=None):
    """Stream a frozen final radial Linear when full gates exceed the message budget.

    The original prefix is evaluated once. Each chunk evaluates the unchanged
    final Linear; its adjoint returns a hidden-feature gradient directly.
    Other workloads retain their established external-gate dispatch.
    """
    from ictc_o3.ascendc.runtime import is_enabled,mode
    if (not is_enabled() or mode()!='capacity' or not src.numel()
            or getattr(owner,'_ascend_tp_schedule',None)=='gemm' or owner.weight.requires_grad
            or not getattr(owner,'_ascend_tp_preproject',True)
            or mask is not None and mask.requires_grad
            or type(radial) is not torch.nn.Sequential or not len(radial) or 'forward' in radial.__dict__
            or not isinstance(radial[-1],torch.nn.Linear)):
        return None
    projection=radial[-1]
    from ictc_o3.ascendc.linear import NativeLinear
    if type(projection) not in (torch.nn.Linear,NativeLinear):return None
    if (projection.weight.requires_grad or projection.bias is not None and projection.bias.requires_grad
            or radial._forward_hooks or radial._forward_pre_hooks
            or radial._backward_hooks or radial._backward_pre_hooks
            or projection._forward_hooks or projection._forward_pre_hooks
            or projection._backward_hooks or projection._backward_pre_hooks):
        return None
    sample=next(iter(features.values()))
    if projection.out_features!=owner.num_paths*sample.shape[1]:
        return None
    from .runtime import require_npu_fp32,_device_memory
    require_npu_fp32(owner.weight,projection.weight,edge_features,*features.values(),*harmonics.values())
    if projection.bias is not None:require_npu_fp32(projection.bias,sample)
    _validate(src,dst,sample.shape[0],sample.device)
    if any(value.shape[0]!=src.numel() for value in harmonics.values()):
        raise ValueError('Harmonic row counts must match graph edges')
    if mask is not None and (mask.device!=sample.device or mask.shape not in ((src.numel(),),(src.numel(),1))):
        raise ValueError('Expected an edge mask [E] or [E,1] on the feature device')
    if edge_chunk_size is not None and (not isinstance(edge_chunk_size,int)
            or isinstance(edge_chunk_size,bool) or edge_chunk_size<1):
        raise ValueError('Edge chunk size must be positive')
    hidden=edge_features
    for layer in list(radial.children())[:-1]:hidden=layer(hidden)
    if hidden.ndim!=2 or hidden.shape!=(src.numel(),projection.in_features):
        raise ValueError('Radial prefix must return one hidden-feature row per edge')
    from .tensor_product import _edge_working_bytes,_destination_nodes
    total=_device_memory(sample.device)
    index=torch.npu.current_device() if sample.device.index is None else sample.device.index
    allocated=int(torch.ops.ictc_ascendc.allocated_bytes(index))
    width=sum(count*(2*degree+1) for degree,count in owner.path_counts_by_l.items())
    node_bytes=_destination_nodes(owner,sample.shape[0])*sample.shape[1]*width*sample.element_size()
    per_edge=_edge_working_bytes(owner,sample,features,harmonics)
    gate_bytes=src.numel()*projection.out_features*sample.element_size()
    grad_enabled=torch.is_grad_enabled()
    gate_gradient=gate_bytes if grad_enabled and hidden.requires_grad else 0
    # Predict the existing selector after materializing the full projection.
    ordinary_budget=min(total//8,max(0,total-allocated-gate_bytes-gate_gradient-6*node_bytes)//2)
    if ordinary_budget>=max(per_edge,2*node_bytes):
        gates=projection(hidden)
        del hidden
        return graph_message(owner,features,harmonics,gates,src,dst,mask,edge_chunk_size=edge_chunk_size)
    hidden_gradient=hidden.numel()*hidden.element_size() if grad_enabled and hidden.requires_grad else 0
    budget=min(total//3,max(0,total-allocated-hidden_gradient-6*node_bytes)//2)
    if budget<max(per_edge,2*node_bytes):
        gates=projection(hidden)
        del hidden
        return graph_message(owner,features,harmonics,gates,src,dst,mask,edge_chunk_size=edge_chunk_size)
    chunk=(min(src.numel(),1048576,budget//per_edge) if edge_chunk_size is None else edge_chunk_size)
    if edge_chunk_size is None and chunk>=128:chunk=chunk//128*128
    if mask is not None:mask=mask.to(dtype=sample.dtype)
    fkeys,hkeys=tuple(sorted(features)),tuple(sorted(harmonics))
    values=_Message.apply(owner,fkeys,hkeys,chunk,projection,*(features[k] for k in fkeys),
                          *(harmonics[k] for k in hkeys),hidden,src,dst,mask)
    return dict(zip(range(owner.lmax+1),values))
