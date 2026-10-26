"""Bound inference message scratch by recomputing one edge chunk per adjoint."""
import torch
from ictc_o3.ascendc.graph import graph_message as original_native_graph
from ictc_o3.ascendc.graph import _validate
from ictc_o3.backends.ascend.channel_message import graph_message as fast_graph

class _Message(torch.autograd.Function):
    @staticmethod
    def forward(ctx,owner,fkeys,hkeys,chunk,*tensors):
        nf,nh=len(fkeys),len(hkeys);features=dict(zip(fkeys,tensors[:nf]));harmonics=dict(zip(hkeys,tensors[nf:nf+nh]));gates,src,dst,mask=tensors[nf+nh:]
        ctx.weight_reference=owner.weight
        ctx.meta=(owner,fkeys,hkeys,chunk);ctx.save_for_backward(owner.weight,*tensors);ctx.set_materialize_grads(False)
        result=None
        for begin in range(0,src.numel(),chunk):
            end=min(src.numel(),begin+chunk)
            parts=fast_graph(owner,features,{k:v[begin:end] for k,v in harmonics.items()},gates[begin:end],src[begin:end],dst[begin:end],None if mask is None else mask[begin:end],edge_chunk_size=chunk)
            if result is None:result=parts
            else:
                for key,value in parts.items():result[key].add_(value)
            torch.npu.current_stream(gates.device).synchronize()
            del parts
        ctx.keys=tuple(result)
        return tuple(result[k] for k in ctx.keys)
    @staticmethod
    def backward(ctx,*outputs):
        owner,fkeys,hkeys,chunk=ctx.meta;saved_weight,*tensors=ctx.saved_tensors;nf,nh=len(fkeys),len(hkeys);ng=nf+nh;gates,src,dst,mask=tensors[ng:];needed=ctx.needs_input_grad[4:];result=[None]*len(tensors)
        if owner.weight is not ctx.weight_reference:
            raise RuntimeError('Tensor-product coefficients were replaced between forward and backward')
        if torch.is_grad_enabled():
            # Preserve native recursive higher derivatives; capacity inference changes only
            # the ordinary first-order inference adjoint's storage schedule.
            with torch.enable_grad():
                aliases=[v.view_as(v) for v in tensors[:ng+1]]
                values=original_native_graph(owner,dict(zip(fkeys,aliases[:nf])),dict(zip(hkeys,aliases[nf:ng])),aliases[ng],src,dst,mask,edge_chunk_size=None)
                selected=[i for i in range(ng+1) if needed[i]]
                pairs=[(values[k],g) for k,g in zip(ctx.keys,outputs) if g is not None and values[k].requires_grad]
                grads=torch.autograd.grad(tuple(v for v,g in pairs),tuple(aliases[i] for i in selected),tuple(g for v,g in pairs),create_graph=True,allow_unused=True)
                for i,g in zip(selected,grads):result[i]=g
            return (None,None,None,None,*result)
        for i in range(ng+1):
            if needed[i]:result[i]=torch.zeros_like(tensors[i]) if i<nf else torch.empty_like(tensors[i])
        for begin in range(0,src.numel(),chunk):
            end=min(src.numel(),begin+chunk)
            with torch.enable_grad():
                values=[(v if i<nf else v[begin:end]).detach().requires_grad_(needed[i]) for i,v in enumerate(tensors[:ng+1])]
                pieces=fast_graph(owner,dict(zip(fkeys,values[:nf])),dict(zip(hkeys,values[nf:ng])),values[ng],src[begin:end],dst[begin:end],None if mask is None else mask[begin:end],edge_chunk_size=chunk)
                selected=[i for i in range(ng+1) if needed[i]];pairs=[(pieces[k],g) for k,g in zip(ctx.keys,outputs) if g is not None and pieces[k].requires_grad]
                grads=torch.autograd.grad(tuple(v for v,g in pairs),tuple(values[i] for i in selected),tuple(g for v,g in pairs),allow_unused=True)
            for i,g in zip(selected,grads):
                if g is None:
                    if i>=nf:result[i][begin:end].zero_()
                elif i<nf:result[i].add_(g)
                else:result[i][begin:end].copy_(g)
            torch.npu.current_stream(gates.device).synchronize()
            del pieces,values,grads,pairs
        return (None,None,None,None,*result)

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
    values=_Message.apply(owner,fkeys,hkeys,chunk,*(features[k] for k in fkeys),*(harmonics[k] for k in hkeys),gates,src,dst,mask)
    return dict(zip(range(owner.lmax+1),values))
