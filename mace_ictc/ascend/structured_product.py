"""Path GEMMs with layout-preserving first and mixed-second adjoints."""
import torch

from mace_ictc.models.adjoint_requirements import required_gradients


class _FixedLinear(torch.autograd.Function):
    """Constant matrix map with a recursively differentiable linear adjoint."""
    @staticmethod
    def forward(ctx, value, matrix):
        ctx.save_for_backward(matrix)
        from mace_ictc.ascendc.runtime import is_enabled
        if is_enabled():
            from mace_ictc.ascendc.linear import matmul
            shape=value.shape
            return matmul(value.reshape(-1,shape[-1]),matrix).reshape(*shape[:-1],matrix.shape[-1])
        return value @ matrix

    @staticmethod
    def backward(ctx, grad):
        matrix, = ctx.saved_tensors
        return _FixedLinear.apply(grad, matrix.T.contiguous()), None


def _forward(meta, right, weights, left, bases=None, *, higher=False):
    rows, paths, channels = weights.shape
    output = [[None]*count for count,width in meta['outputs']]
    saved = [right.new_empty(0) for _ in range(paths)]
    for li, lo, offset, bo, ni, no, path in meta['specs']:
        if left[li] is None:
            continue
        if bases is None:
            matrix = right[:,bo:bo+ni*no].reshape(rows,ni,no)
            # CANN matmul_double_backward has no third-order autograd in the
            # qualified stack. Higher derivatives use elementary polynomials.
            base = ((left[li].unsqueeze(-1)*matrix.unsqueeze(1)).sum(-2) if higher
                    else torch.bmm(left[li],matrix))
        else:
            base = bases[path]
        saved[path] = base
        output[lo][offset//channels] = base*weights[:,path,:,None]
    result=[]
    for blocks,(count,width) in zip(output,meta['outputs']):
        result.append(torch.cat([right.new_zeros(rows,channels,width) if block is None else block
                                for block in blocks],1) if count else right.new_empty(rows,0,width))
    return tuple(result),tuple(saved)


def _left_grad(meta,right,weights,go,needed):
    rows,paths,channels=weights.shape
    result=[None]*meta['left_count']
    for li,lo,offset,bo,ni,no,path in meta['specs']:
        if not needed[li] or go[lo] is None:
            continue
        cotangent=go[lo][:,offset:offset+channels,:]*weights[:,path,:,None]
        value=torch.bmm(cotangent,right[:,bo:bo+ni*no].reshape(rows,ni,no).transpose(1,2))
        result[li]=value if result[li] is None else result[li]+value
    return tuple(result)


def _right_grad(meta,right,weights,left,go):
    rows,paths,channels=weights.shape
    parts=[]
    any_active=False
    for li,lo,offset,bo,ni,no,path in meta['specs']:
        if left[li] is None or go[lo] is None:
            parts.append(right.new_zeros(rows,ni*no))
            continue
        any_active=True
        cotangent=go[lo][:,offset:offset+channels,:]*weights[:,path,:,None]
        parts.append(torch.bmm(left[li].transpose(1,2),cotangent).reshape(rows,ni*no))
    return torch.cat(parts,1) if any_active else None


def _weight_grad(meta,right,weights,left,go,bases=None):
    rows,paths,channels=weights.shape
    parts=[None]*paths
    any_active=False
    for li,lo,offset,bo,ni,no,path in meta['specs']:
        if left[li] is None or go[lo] is None:
            continue
        any_active=True
        base=(torch.bmm(left[li],right[:,bo:bo+ni*no].reshape(rows,ni,no))
              if bases is None else bases[path])
        parts[path]=(base*go[lo][:,offset:offset+channels,:]).sum(-1)
    return torch.stack([right.new_zeros(rows,channels) if value is None else value
                        for value in parts],1) if any_active else None


def _first(meta,right,weights,left,go,needed,bases=None):
    gr=_right_grad(meta,right,weights,left,go) if needed[0] else None
    gw=_weight_grad(meta,right,weights,left,go,bases) if needed[1] else None
    gl=_left_grad(meta,right,weights,go,needed[2:]) if any(needed[2:]) else (None,)*len(left)
    return gr,gw,*gl


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,meta,bases,needed,present,right,weights,*rest):
        n=meta['left_count']
        left,go=rest[:n],rest[n:]
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(right,weights,*left,*go,*bases)
        ctx.spec=meta,needed,present
        return _first(meta,right,weights,left,tuple(g if p else None for g,p in zip(go,present)),needed,bases)

    @staticmethod
    def backward(ctx,hr,hw,*hl):
        meta,first_needed,present=ctx.spec
        n=meta['left_count']
        right,weights,*saved=ctx.saved_tensors
        left=tuple(saved[:n])
        gout_tokens=tuple(saved[n:n+len(present)])
        bases=tuple(saved[n+len(present):])
        go=tuple(g if p else None for g,p in zip(gout_tokens,present))
        values=(right,weights,*left,*gout_tokens)
        needed=required_gradients(values,ctx.needs_input_grad[4:])
        if torch.is_grad_enabled():
            with torch.enable_grad():
                local=[value.clone() for value in values]
                outputs,_=_forward(meta,local[0],local[1],tuple(local[2:2+n]),higher=True)
                ids=[i for i,need in enumerate(first_needed) if need]
                live=[(y,g) for y,g,p in zip(outputs,local[2+n:],present) if p and y.requires_grad]
                first=torch.autograd.grad([y for y,g in live],[local[i] for i in ids],
                                          [g for y,g in live],create_graph=True,allow_unused=True)
                hs=(hr,hw,*hl)
                terms=[(value*hs[i]).sum() for i,value in zip(ids,first)
                       if value is not None and hs[i] is not None and value.requires_grad]
                result=[None]*len(values)
                active=[i for i,value in enumerate(local) if needed[i] and value.requires_grad]
                if terms and active:
                    grads=torch.autograd.grad(sum(terms),[local[i] for i in active],
                                              create_graph=True,allow_unused=True)
                    for i,value in zip(active,grads):result[i]=value
            return None,None,None,None,*result
        result=[None]*len(values)
        def add(index,value):
            if value is not None:
                result[index]=value if result[index] is None else result[index]+value
        def add_left(items):
            for i,value in enumerate(items):
                if needed[2+i]:add(2+i,value)
        def add_go(items):
            for i,value in enumerate(items):
                if needed[2+n+i]:add(2+n+i,value)
        if any(h is not None for h in hl):
            hleft=tuple(None if h is None else h.contiguous() for h in hl)
            if needed[0]:add(0,_right_grad(meta,right,weights,hleft,go))
            if needed[1]:add(1,_weight_grad(meta,right,weights,hleft,go))
            if any(needed[2+n:]):add_go(_forward(meta,right,weights,hleft)[0])
        if hr is not None:
            if any(needed[2:2+n]):add_left(_left_grad(meta,hr,weights,go,needed[2:2+n]))
            if needed[1]:add(1,_weight_grad(meta,hr,weights,left,go))
            if any(needed[2+n:]):add_go(_forward(meta,hr,weights,left)[0])
        if hw is not None:
            if any(needed[2:2+n]):add_left(_left_grad(meta,right,hw,go,needed[2:2+n]))
            if needed[0]:add(0,_right_grad(meta,right,hw,left,go))
            if any(needed[2+n:]):add_go(_forward(meta,right,hw,left,bases)[0])
        return None,None,None,None,*result


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,meta,right,weights,*left):
        outputs,bases=_forward(meta,right,weights,left)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(right,weights,*left,*bases)
        ctx.spec=meta,len(left)
        return outputs

    @staticmethod
    def backward(ctx,*go):
        meta,n=ctx.spec
        right,weights,*saved=ctx.saved_tensors
        left,bases=tuple(saved[:n]),tuple(saved[n:])
        needed=required_gradients((right,weights,*left),ctx.needs_input_grad[1:])
        if torch.is_grad_enabled():
            present=tuple(g is not None for g in go)
            tokens=tuple(right.new_empty(0) if g is None else g for g in go)
            result=_Adjoint.apply(meta,bases,needed,present,right,weights,*left,*tokens)
        else:
            result=_first(meta,right,weights,left,go,needed,bases)
        return None,*result


def tensor_product(owner,features,harmonics,gates,plan):
    meta=plan.get('_ascend_structured')
    if meta is None:
        degrees=sorted(features)
        index={degree:i for i,degree in enumerate(degrees)}
        specs=[]
        offset=0
        for path,(i,j,k) in enumerate(owner.paths):
            if i not in index or j not in harmonics:continue
            ni,no=2*i+1,2*k+1
            specs.append((index[i],k,owner.path_offset[path]*owner.channels,offset,ni,no,path))
            offset+=ni*no
        meta=dict(left_count=len(degrees),input_degrees=tuple(degrees),specs=tuple(specs),
                  outputs=tuple((owner.path_counts_by_l.get(l,0),2*l+1) for l in range(owner.lmax+1)))
        plan['_ascend_structured']=meta
    rows=next(iter(features.values())).shape[0]
    sh=torch.cat([harmonics[l] for l in sorted(harmonics)],-1).squeeze(1)
    right=_FixedLinear.apply(sh,plan['bond_projection'])
    weights=(owner.weight[None].expand(rows,-1,-1) if gates is None else
             gates.reshape(rows,owner.num_paths,owner.channels)*owner.weight[None])
    outputs=_Product.apply(meta,right,weights,*[features[l].contiguous() for l in meta['input_degrees']])
    return dict(enumerate(outputs))
