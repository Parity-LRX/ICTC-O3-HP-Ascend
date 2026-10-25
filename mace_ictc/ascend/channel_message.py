"""Channel-contiguous messages with one GEMM per input angular degree."""
import os
import torch
from mace_ictc.models.adjoint_requirements import required_gradients
from .structured_product import _FixedLinear


class _BatchedMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,ta,tb):
        ctx.save_for_backward(a if b.requires_grad else a.new_empty(0),
                              b if a.requires_grad else b.new_empty(0))
        ctx.spec=ta,tb
        from mace_ictc.ascendc.batched_vector import try_apply
        from .runtime import drain_graph_queue
        drain_graph_queue(a.device)
        value=try_apply(a,b,ta,tb)
        return value if value is not None else torch.bmm(a.transpose(-2,-1) if ta else a,b.transpose(-2,-1) if tb else b)

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
    from .tensor_product import _plan,_automatic_edge_chunk,_edge_working_bytes
    from .runtime import require_npu_fp32,bound_graph_queue
    sample=next(iter(features.values()))
    nodes,channels=sample.shape[:2]
    require_npu_fp32(owner.weight,gates,*features.values(),*harmonics.values())
    edges=src.numel()
    bound_graph_queue(sample.device,edges*_edge_working_bytes(owner,sample,features,harmonics))
    if gates.shape not in ((edges,owner.num_paths*channels),(edges,owner.num_paths,channels)):
        raise ValueError('External graph gates must have shape [E,P*C] or [E,P,C]')
    if mask is not None and (mask.shape not in ((edges,),(edges,1)) or mask.device!=sample.device):
        raise ValueError('Expected an edge mask [E] or [E,1] on the feature device')
    if edge_chunk_size is None:
        edge_chunk_size=_automatic_edge_chunk(owner,sample,features,harmonics)
    if not isinstance(edge_chunk_size,int) or isinstance(edge_chunk_size,bool) or edge_chunk_size<1:
        raise ValueError('Edge chunk size must be a positive integer')
    output={l:None for l in range(owner.lmax+1)}
    if not edges:
        output={l:sample.new_zeros(nodes,owner.path_counts_by_l.get(l,0)*(2*l+1),channels) for l in output}
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
    from mace_ictc.ascendc.epilogue import enabled,supported,regroup,apply as epilogue
    fused=(enabled(mask,group_plan[1],channels,min(edges,edge_chunk_size)) and bool(group_plan[1])
           and supported(group_plan[1],mask is not None))
    projection,groups=group_plan
    weight_groups=groups
    compact=False
    if fused:
        widths=[owner.path_counts_by_l.get(l,0)*(2*l+1) for l in output]
        offsets=[]
        offset=0
        for width in widths:
            offsets.append(offset)
            offset+=width
        plan['_ascendc_output_widths']=tuple(widths)
        plan['_ascendc_owner_path_offset']=tuple(owner.path_offset)
        projection,groups=regroup(plan,projection,groups,owner.path_offset)
        policy=os.environ.get('ICTC_ASCEND_COMPACT_OUTPUT','1')
        if policy not in ('0','1'):
            raise ValueError('The compact graph-output policy must be zero or one')
        compact=policy=='1'
        if compact and '_ascendc_compact_order' not in plan:
            width=sum(group[2] for group in groups)
            restore=[width]*sum(widths)
            start=0
            for group in groups:
                for p,_,_,degree,_,span in group[4]:
                    target=offsets[degree]+owner.path_offset[p]*span
                    restore[target:target+span]=range(start,start+span)
                    start+=span
            inverse=None if restore==list(range(len(restore))) else torch.tensor(restore,dtype=torch.int64,device=sample.device)
            plan['_ascendc_compact_order']=(inverse,any(item==width for item in restore))
    policy=os.environ.get('ICTC_ASCEND_CHUNK_NODE_RESTORE','1')
    if policy not in ('0','1'):
        raise ValueError('The chunk node-restore policy must be zero or one')
    late_restore=fused and compact and policy=='1'
    if not late_restore:
        output={l:sample.new_zeros(nodes,owner.path_counts_by_l.get(l,0)*(2*l+1),channels) for l in output}
    compact_sums=None
    policy=os.environ.get('ICTC_ASCEND_NODE_ACCUMULATE','1')
    if policy not in ('0','1'):
        raise ValueError('The native node-accumulation policy must be zero or one')
    node_accumulate=late_restore and policy=='1'
    policy=os.environ.get('ICTC_ASCEND_CHUNK_SPLIT','1')
    if policy not in ('0','1'):
        raise ValueError('The shared chunk-adjoint policy must be zero or one')
    split=policy=='1'
    # One split joins chunk cotangents without a full-edge gradient per slice.
    sh_parts=((sh,) if edges<=edge_chunk_size else sh.split(edge_chunk_size,0)) if split else ()
    gate_parts=((gates,) if edges<=edge_chunk_size else gates.split(edge_chunk_size,0)) if split else ()
    for chunk,begin in enumerate(range(0,edges,edge_chunk_size)):
        end=min(begin+edge_chunk_size,edges)
        rows=end-begin
        edge={l:value.index_select(0,src[begin:end]) for l,value in angular.items()}
        part=gate_parts[chunk] if split else gates[begin:end]
        weighted=part.reshape(rows,owner.num_paths,channels)*owner.weight[None]
        blocks={l:[None]*owner.path_counts_by_l.get(l,0) for l in output}
        bond=_FixedLinear.apply(sh_parts[chunk] if split else sh[begin:end],projection)
        matrices=bond.split([ni*width for i,ni,width,widths,items in groups],1)
        if fused:
            order=tuple(item[0] for group in weight_groups for item in group[4])
            if order==tuple(range(owner.num_paths)):
                grouped_weights=weighted.split([len(group[4]) for group in weight_groups],1)
            else:
                weights=weighted.unbind(1)
                grouped_weights=tuple(torch.stack([weights[item[0]] for item in group[4]],1) for group in weight_groups)
            accumulated=None
            node_parts=[]
            for group_id,(group,old_group,matrix,weight) in enumerate(zip(groups,weight_groups,matrices,grouped_weights)):
                i,ni,width,_,_=group
                base=_BatchedMatrix.apply(matrix.reshape(rows,width,ni),edge[i],False,False)
                value=epilogue(base,weight,plan,group,tuple(item[0] for item in old_group[4]),offsets,dst[begin:end],nodes,
                               None if mask is None else mask[begin:end].reshape(-1).contiguous(),compact=compact,
                               accumulator=compact_sums[group_id] if node_accumulate and compact_sums is not None else None)
                if compact:node_parts.append(value)
                else:accumulated=value if accumulated is None else accumulated+value
            if late_restore:
                # Keep each input-degree block in its compact order until all
                # disjoint edge chunks have contributed to the node tensor.
                compact_sums=(node_parts if compact_sums is None or node_accumulate else
                              [old+value for old,value in zip(compact_sums,node_parts)])
                continue
            if compact:
                restore,missing=plan['_ascendc_compact_order']
                if missing:node_parts.append(sample.new_zeros(nodes,1,channels))
                packed=node_parts[0] if len(node_parts)==1 else torch.cat(node_parts,1)
                if restore is None:accumulated=packed
                else:
                    from .node_layout import restore_output
                    accumulated=restore_output(packed,restore,plan)
            for (l,old),block in zip(output.items(),accumulated.split(plan['_ascendc_output_widths'],1)):
                output[l]=old+block
            continue
        weights=weighted.unbind(1)
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
    if late_restore:
        restore,missing=plan['_ascendc_compact_order']
        if missing:compact_sums.append(sample.new_zeros(nodes,1,channels))
        packed=compact_sums[0] if len(compact_sums)==1 else torch.cat(compact_sums,1)
        if restore is None:accumulated=packed
        else:
            from .node_layout import restore_output
            accumulated=restore_output(packed,restore,plan)
        output=dict(zip(output,accumulated.split(plan['_ascendc_output_widths'],1)))
    # Restore the public path/channel/angular convention at node count only.
    return {l:value.reshape(nodes,owner.path_counts_by_l.get(l,0),2*l+1,channels)
                   .transpose(2,3).reshape(nodes,owner.path_counts_by_l.get(l,0)*channels,2*l+1)
            for l,value in output.items()}
