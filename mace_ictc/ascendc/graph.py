"""Fused graph gather, multilinear map and FP32 destination accumulation."""
import weakref
import torch


def _validate(src,dst,nodes,device):
    if (src.device!=device or dst.device!=device or src.dtype!=torch.int64 or dst.dtype!=torch.int64
            or src.ndim!=1 or dst.shape!=src.shape):
        raise ValueError('Expected NPU int64 edge vectors on the feature device')
    key=(src._version,dst._version,nodes)
    cached=getattr(src,'_ictc_ascendc_graph_bounds',None)
    if cached is not None and cached[0]() is dst and cached[1]==key:
        return
    if src.numel():
        bounds=torch.stack((src.amin(),src.amax(),dst.amin(),dst.amax())).cpu().tolist()
        if bounds[0]<0 or bounds[2]<0 or bounds[1]>=nodes or bounds[3]>=nodes:
            raise ValueError('Graph index is outside the node array')
    src._ictc_ascendc_graph_bounds=(weakref.ref(dst),key)


def graph_message(owner,features,harmonics,gates,src,dst,mask=None,*,edge_chunk_size=None):
    from mace_ictc.ascend.tensor_product import _plan,_automatic_edge_chunk
    from mace_ictc.ascend.structured_product import _FixedLinear
    from mace_ictc.ascend.sparse_product import _Product
    from mace_ictc.ascend.runtime import require_npu_fp32
    sample=next(iter(features.values()))
    nodes,channels=sample.shape[:2]
    _validate(src,dst,nodes,sample.device)
    require_npu_fp32(owner.weight,gates,*features.values(),*harmonics.values())
    edges=src.numel()
    if gates.shape not in ((edges,owner.num_paths*channels),(edges,owner.num_paths,channels)):
        raise ValueError('External graph gates must have shape [E,P*C] or [E,P,C]')
    if mask is not None and (mask.shape not in ((edges,),(edges,1)) or mask.device!=sample.device):
        raise ValueError('Expected a fixed edge mask [E] or [E,1]')
    if edge_chunk_size is None:
        edge_chunk_size=_automatic_edge_chunk(owner,sample,features,harmonics)
    if not isinstance(edge_chunk_size,int) or isinstance(edge_chunk_size,bool) or edge_chunk_size<1:
        raise ValueError('Edge chunk size must be a positive integer')
    left=torch.cat([features[l] for l in sorted(features)],-1).contiguous()
    sh=torch.cat([harmonics[l] for l in sorted(harmonics)],-1).squeeze(1)
    base=_plan(owner,left,sorted(features),sorted(harmonics))
    # Per-call graph metadata stays local; immutable schedules are shared by
    # every chunk and call using this mathematical coupling plan.
    base.setdefault('_ascendc_metadata',{})
    base.setdefault('_ascendc_tiling',{})
    if 'bond_projection' not in base:
        raise ValueError('Native fused graph messages require shared geometry projection')
    result=None
    for begin in range(0,max(1,edges),edge_chunk_size):
        end=min(begin+edge_chunk_size,edges)
        right=_FixedLinear.apply(sh[begin:end],base['bond_projection']).unsqueeze(1)
        weight=(gates[begin:end].reshape(end-begin,owner.num_paths,channels).transpose(1,2)*
                owner.weight.T.unsqueeze(0)).contiguous()
        graph=dict(src=src[begin:end].contiguous(),dst=dst[begin:end].contiguous(),
                   mask=None if mask is None else mask[begin:end].reshape(-1).to(dtype=sample.dtype).contiguous())
        plan={**base,'_ascendc_graph':graph}
        value=_Product.apply(left,right,weight,None,plan,False)
        result=value if result is None else result+value
    blocks,offset={},0
    for degree in range(owner.lmax+1):
        count=owner.path_counts_by_l.get(degree,0)
        width=2*degree+1
        span=count*width
        blocks[degree]=result[:,:,offset:offset+span].reshape(nodes,channels,count,width).transpose(1,2).reshape(nodes,count*channels,width)
        offset+=span
    return blocks
