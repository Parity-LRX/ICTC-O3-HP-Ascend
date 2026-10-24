"""Channel-contiguous graph epilogues on the Ascend Vector cores."""
import os
import torch
from .runtime import load_library


def enabled(mask,groups,channels,rows):
    if mask is not None and (mask.requires_grad or mask.dtype != torch.float32):
        return False
    policy=os.environ.get('ICTC_ASCEND_EPILOGUE', 'auto')
    if policy not in ('auto','0','1'):
        raise ValueError('The epilogue policy must be auto, zero or one')
    # Amortize irregular transfers and launches using the actual message volume.
    width=sum(group[2] for group in groups)
    return policy=='1' or (policy=='auto' and width>=8 and rows*width*channels>=64*1024**2)


def _align(n):
    return (n+7)//8*8


def supported(groups, masked=False):
    ub,_ = load_library().hardware()
    reserve = max(4096,int(os.environ.get('ICTC_ASCEND_EPILOGUE_RESERVE', '32768')))
    for _,_,q,_,items in groups:
        p,bc = len(items),8
        if q > 1024 or p > 4095:
            return False
        for mode in range(3):
            vec,rc = q*bc,bc
            inp = (0 if mode == 1 else vec)+(0 if mode == 2 else p*bc)+(
                0 if mode == 0 else vec)+8+(8 if masked else 0)
            out = p*bc if mode == 2 else vec
            if 4*(2*inp+2*out+3*vec+3*rc+_align(4*p))+reserve > ub:
                return False
    return True


def _schedule(plan, sample, mode, masked):
    q,p,_,m = plan['sizes']
    c = sample.shape[1]
    ub,cores = load_library().hardware()
    cap = int(os.environ.get('ICTC_ASCEND_EPILOGUE_CHANNELS', '64'))
    reserve = int(os.environ.get('ICTC_ASCEND_EPILOGUE_RESERVE', '32768'))
    if cap < 8 or cap % 8 or reserve < 4096 or reserve >= ub:
        raise ValueError('Invalid epilogue channel tile or UB reserve')
    requested_dense = int(os.environ.get('ICTC_ASCEND_EPILOGUE_DENSE_G', '0'))
    if requested_dense not in (0,1):
        raise ValueError('The epilogue node-gather override must be zero or one')
    key = str(sample.device),c,mode,masked,cap,reserve,ub,requested_dense
    cache = plan.setdefault('_ascendc_epilogue_schedule', {})
    if key in cache:
        return cache[key]
    # Balance channel tails; all UB rows remain aligned to a 32-byte block.
    for dense in ((1,0) if requested_dense and mode and m <= 4095 else (0,)):
        blocks = (c+cap-1)//cap
        bc = _align((c+blocks-1)//blocks)
        while bc >= 8:
            rows = min(4095,8192//(q*bc))
            while rows:
                vec,rc = rows*q*bc,rows*bc
                inp = (0 if mode == 1 else vec)+(0 if mode == 2 else rows*p*bc)+(
                    0 if mode == 0 else rows*(m if dense else q)*bc)+_align(2*rows)+(_align(rows) if masked else 0)
                out = rows*p*bc if mode == 2 else vec
                # Two input/output slots, work vectors, byte indices and metadata.
                used = 4*(2*inp+2*out+(3+dense)*vec+3*rc+_align(4*p))
                if used+reserve <= ub:
                    break
                rows -= 1
            if rows:
                break
            bc -= 8
        if rows:
            break
    if not rows or q > 4095 or p > 4095:
        raise ValueError('The epilogue geometry exceeds the native UB or DMA limits')
    specs = plan['_ascendc_epilogue']
    path_for_q = [0]*q
    output_for_q = [0]*q
    for start,path,output,width in specs:
        path_for_q[start:start+width] = [path]*width
        output_for_q[start:start+width] = range(output,output+width)
    packet = []
    packet.extend((r*p*bc+path_for_q[a]*bc+ch)*4
                  for r in range(rows) for a in range(q) for ch in range(bc))
    packet.extend(r*4 for r in range(rows) for _ in range(q*bc))
    packet.extend((r*q*bc+ch)*4 for r in range(rows) for ch in range(bc))
    packet.extend(r*4 for r in range(rows) for _ in range(bc))
    if dense:
        packet.extend((r*m*bc+output_for_q[a]*bc+ch)*4
                      for r in range(rows) for a in range(q) for ch in range(bc))
    metadata = torch.tensor([v for spec in specs for v in spec],dtype=torch.int32,device=sample.device)
    indices = torch.tensor(packet,dtype=torch.int32,device=sample.device)
    result = rows,bc,metadata,indices,dense,used,ub,cores
    cache[key] = result
    return result


def _physical(value):
    value = value.transpose(1,2)
    return value if (value.stride(2) == 1 and value.stride(1) == value.shape[2]
                     and value.stride(0) >= value.shape[1]*value.shape[2]) else value.contiguous()


def raw(left,right,plan,mode=0,grad=None):
    graph = plan['_ascendc_graph']
    mask = graph['mask']
    rows,bc,metadata,indices,dense,*_ = _schedule(plan,left,mode,mask is not None)
    n,c,q = left.shape
    p,_,m = plan['sizes'][1:]
    dummy = left.new_empty(0)
    shape = [n,c,q,p,m,p,mode,rows,bc,int(mask is not None),graph['destination_nodes'],dense]
    out = load_library().epilogue(_physical(left),_physical(right),
        dummy if grad is None else grad.transpose(1,2).contiguous(),graph['dst'].contiguous(),
        dummy if mask is None else mask.contiguous(),metadata,indices,shape)
    return out.transpose(1,2)


def _native(left,right,plan,mode=0,grad=None):
    return raw(left.transpose(1,2),right.transpose(1,2),plan,mode,
               None if grad is None else grad.transpose(1,2)).transpose(1,2)


def _reference(left,right,plan):
    from mace_ictc.ascend.sparse_product import _reference as polynomial
    return polynomial(left.transpose(1,2),right.transpose(1,2),
                      plan['_ascendc_ones'],None,plan,True).transpose(1,2)


class _PhysicalAdjoint(torch.autograd.Function):
    """Bilinear adjoints with the same contiguous channel axis as the GEMM."""
    @staticmethod
    def forward(ctx,left,right,grad,plan,needed):
        from mace_ictc.ascend.sparse_product import _graph_tensors
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(left,right,grad,*_graph_tensors(plan))
        ctx.spec=plan,needed
        return tuple(_native(left,right,plan,i+1,grad) if need else None
                     for i,need in enumerate(needed))

    @staticmethod
    def backward(ctx,hleft,hright):
        from mace_ictc.models.adjoint_requirements import required_gradients
        left,right,grad,*_=ctx.saved_tensors
        plan,first_needed=ctx.spec
        needed=required_gradients((left,right,grad),ctx.needs_input_grad[:3])
        if torch.is_grad_enabled():
            with torch.enable_grad():
                local=[v.clone() for v in (left,right,grad)]
                active=[i for i,v in enumerate(local) if v.requires_grad]
                y=_reference(*local[:2],plan)
                first_ids=[i for i in range(2) if first_needed[i]]
                first=torch.autograd.grad(y,[local[i] for i in first_ids],local[2],
                                          create_graph=True,allow_unused=True)
                hs=(hleft,hright)
                terms=[(value*hs[i]).sum() for i,value in zip(first_ids,first)
                       if value is not None and hs[i] is not None and value.requires_grad]
                result=[None]*3
                if terms:
                    values=torch.autograd.grad(sum(terms),[local[i] for i in active],
                                               create_graph=True,allow_unused=True)
                    for i,value in zip(active,values):
                        result[i]=value if needed[i] else None
            return *result,None,None
        dl=dr=dg=None
        if hleft is not None:
            if needed[1]:dr=_native(hleft,right,plan,2,grad)
            if needed[2]:dg=_native(hleft,right,plan)
        if hright is not None:
            if needed[0]:dl=_native(left,hright,plan,1,grad)
            if needed[2]:
                value=_native(left,hright,plan)
                dg=value if dg is None else dg+value
        return dl,dr,dg,None,None


class _PhysicalProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx,left,right,plan):
        from mace_ictc.ascend.sparse_product import _graph_tensors
        ctx.save_for_backward(left,right,*_graph_tensors(plan))
        ctx.plan=plan
        return _native(left,right,plan)

    @staticmethod
    def backward(ctx,grad):
        from mace_ictc.models.adjoint_requirements import required_gradients
        left,right,*_=ctx.saved_tensors
        needed=required_gradients((left,right),ctx.needs_input_grad[:2])
        if torch.is_grad_enabled():
            values=_PhysicalAdjoint.apply(left,right,grad,ctx.plan,needed)
        else:
            values=tuple(_native(left,right,ctx.plan,i+1,grad) if need else None
                         for i,need in enumerate(needed))
        return *values,None


def regroup(plan, projection, groups, path_offsets):
    """Place adjacent destination paths next to each other in the Cube output."""
    key='_ascendc_epilogue_groups'
    if key not in plan:
        columns,new_groups,start=[],[],0
        for i,ni,width,widths,items in groups:
            positions={}
            q=0
            for item in items:
                positions[item[0]]=q
                q+=item[5]
            ordered=tuple(sorted(items,key=lambda item:(item[3],path_offsets[item[0]])))
            for item in ordered:
                columns.extend(start+(positions[item[0]]+a)*ni+t
                               for a in range(item[5]) for t in range(ni))
            new_groups.append((i,ni,width,tuple(item[5] for item in ordered),ordered))
            start+=ni*width
        index=torch.tensor(columns,dtype=torch.int64,device=projection.device)
        plan[key]=projection.index_select(1,index).contiguous(),tuple(new_groups)
    return plan[key]


def apply(base,weights,plan,group,weight_ids,offsets,destination,nodes,mask=None):
    """Fuse path gates and destination accumulation without edge-sized copies."""
    _,_,q,_,items = group
    c = base.shape[2]
    key = tuple(item[0] for item in items),c,str(base.device)
    cache = plan.setdefault('_ascendc_epilogue_plans', {})
    if key not in cache:
        specs,entries,start = [],[],0
        for item in items:
            p,_,_,l,_,width = item
            local=weight_ids.index(p)
            output = offsets[l]+plan['_ascendc_owner_path_offset'][p]*width
            specs.append((start,local,output,width))
            entries.extend((start+a,local,0,output+a,1.) for a in range(width))
            start += width
        cache[key] = dict(sizes=(q,len(items),1,sum(plan['_ascendc_output_widths'])),
                          entries=tuple(entries),_ascendc_epilogue=tuple(specs),
                          _ascendc_identity={},_ascendc_epilogue_schedule={},
                          _ascendc_ones=base.new_ones(1,c,1))
    core = cache[key]
    identities = core['_ascendc_identity']
    edges = base.shape[0]
    if edges not in identities:
        if len(identities) >= 8:
            identities.clear()
        identities[edges] = torch.arange(edges,device=base.device,dtype=torch.int64)
    proxy = dict(core)
    proxy['_ascendc_graph'] = dict(src=identities[edges],dst=destination,
                                 mask=mask,destination_nodes=nodes)
    return _PhysicalProduct.apply(base,weights,proxy)
