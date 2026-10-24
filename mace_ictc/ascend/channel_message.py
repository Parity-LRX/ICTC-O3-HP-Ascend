"""Channel-contiguous messages with one GEMM per input angular degree."""
import torch
from mace_ictc.models.adjoint_requirements import required_gradients
from .structured_product import _FixedLinear


class _BatchedMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,ta,tb):
        ctx.save_for_backward(a if b.requires_grad else a.new_empty(0),
                              b if a.requires_grad else b.new_empty(0))
        ctx.spec=ta,tb
        return torch.bmm(a.transpose(-2,-1) if ta else a,b.transpose(-2,-1) if tb else b)

    @staticmethod
    def backward(ctx,g):
        a,b=ctx.saved_tensors
        ta,tb=ctx.spec
        needed=required_gradients((a,b),ctx.needs_input_grad[:2])
        da=db=None
        if needed[0]:
            da=(_BatchedMatrix.apply(b,g,tb,True) if ta else
                _BatchedMatrix.apply(g,b,False,not tb))
        if needed[1]:
            db=(_BatchedMatrix.apply(g,a,True,ta) if tb else
                _BatchedMatrix.apply(a,g,not ta,False))
        return da,db,None,None


def graph_message(owner,features,harmonics,gates,src,dst,mask=None,*,edge_chunk_size=None):
    from .tensor_product import _plan,_automatic_edge_chunk
    from .runtime import require_npu_fp32
    sample=next(iter(features.values()))
    nodes,channels=sample.shape[:2]
    require_npu_fp32(owner.weight,gates,*features.values(),*harmonics.values())
    edges=src.numel()
    if gates.shape not in ((edges,owner.num_paths*channels),(edges,owner.num_paths,channels)):
        raise ValueError('External graph gates must have shape [E,P*C] or [E,P,C]')
    if mask is not None and (mask.shape not in ((edges,),(edges,1)) or mask.device!=sample.device):
        raise ValueError('Expected an edge mask [E] or [E,1] on the feature device')
    if edge_chunk_size is None:
        edge_chunk_size=_automatic_edge_chunk(owner,sample,features,harmonics)
    if not isinstance(edge_chunk_size,int) or isinstance(edge_chunk_size,bool) or edge_chunk_size<1:
        raise ValueError('Edge chunk size must be a positive integer')
    output={l:sample.new_zeros(nodes,owner.path_counts_by_l.get(l,0)*(2*l+1),channels)
            for l in range(owner.lmax+1)}
    if not edges:
        zero=owner.weight.sum()*0+gates.sum()*0
        zero=zero+sum(value.sum()*0 for value in (*features.values(),*harmonics.values()))
        if mask is not None:zero=zero+mask.sum()*0
        return {l:value.reshape(nodes,owner.path_counts_by_l.get(l,0)*channels,2*l+1)+zero
                for l,value in output.items()}
    plan=_plan(owner,sample,sorted(features),sorted(harmonics))
    specs=plan.get('_ascend_channel_specs')
    if specs is None:
        specs=tuple((p,i,j,k,2*i+1,2*k+1) for p,(i,j,k) in enumerate(owner.paths)
                    if i in features and j in harmonics)
        plan['_ascend_channel_specs']=specs
    group_plan=plan.get('_ascend_channel_groups')
    if group_plan is None:
        by_input={}
        offset=0
        for spec in specs:
            p,i,j,k,ni,no=spec
            by_input.setdefault(i,[]).append((*spec,offset))
            offset+=ni*no
        order=[]
        groups=[]
        for i,items in sorted(by_input.items()):
            ni=items[0][4]
            widths=tuple(item[5] for item in items)
            for p,li,j,k,ni,no,offset in items:
                order.extend(offset+a*no+q for q in range(no) for a in range(ni))
            groups.append((i,ni,sum(widths),widths,tuple(item[:6] for item in items)))
        index=torch.tensor(order,device=sample.device,dtype=torch.int64)
        projection=plan['bond_projection'].index_select(1,index).contiguous()
        group_plan=(projection,tuple(groups))
        plan['_ascend_channel_groups']=group_plan
    # The node-sized transpose makes channels contiguous before source gather.
    # Edge intermediates retain this layout through both matrix adjoints.
    angular={l:value.transpose(1,2).contiguous() for l,value in features.items()}
    sh=torch.cat([harmonics[l] for l in sorted(harmonics)],-1).squeeze(1)
    for begin in range(0,edges,edge_chunk_size):
        end=min(begin+edge_chunk_size,edges)
        rows=end-begin
        edge={l:value.index_select(0,src[begin:end]) for l,value in angular.items()}
        weights=(gates[begin:end].reshape(rows,owner.num_paths,channels)*owner.weight[None]).unbind(1)
        blocks={l:[None]*owner.path_counts_by_l.get(l,0) for l in output}
        projection,groups=group_plan
        bond=_FixedLinear.apply(sh[begin:end],projection)
        matrices=bond.split([ni*width for i,ni,width,widths,items in groups],1)
        for (i,ni,width,widths,items),matrix in zip(groups,matrices):
            value=_BatchedMatrix.apply(matrix.reshape(rows,width,ni),edge[i],False,False)
            for spec,part in zip(items,value.split(widths,1)):
                p,li,j,k,ni,no=spec
                blocks[k][owner.path_offset[p]]=part*weights[p][:,None,:]
        for l,items in blocks.items():
            values=[sample.new_zeros(rows,2*l+1,channels) if value is None else value for value in items]
            block=torch.cat(values,1) if values else sample.new_empty(rows,0,channels)
            if mask is not None:block=block*mask[begin:end].reshape(-1,1,1)
            output[l]=output[l].index_add(0,dst[begin:end],block)
    # Restore the public path/channel/angular convention at node count only.
    return {l:value.reshape(nodes,owner.path_counts_by_l.get(l,0),2*l+1,channels)
                   .transpose(2,3).reshape(nodes,owner.path_counts_by_l.get(l,0)*channels,2*l+1)
            for l,value in output.items()}
