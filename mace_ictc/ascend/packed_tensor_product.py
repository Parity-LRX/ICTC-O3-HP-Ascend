"""Standalone path-preserving TP with channels contiguous inside each GEMM."""
import os

import torch

from mace_ictc.models.adjoint_requirements import required_gradients
from .runtime import require_npu_fp32
from .structured_product import _FixedLinear
from .tensor_product import _plan


class _Matrix(torch.autograd.Function):
    """Recursively differentiable BMM with a standalone angular dispatch."""
    @staticmethod
    def forward(ctx, a, b, ta, tb):
        ctx.save_for_backward(a if b.requires_grad else a.new_empty(0),
                              b if a.requires_grad else b.new_empty(0))
        ctx.spec = ta, tb
        from mace_ictc.ascendc.batched_vector import try_apply
        from .runtime import drain_graph_queue
        drain_graph_queue(a.device)
        q = a.shape[2] if ta else a.shape[1]
        inner = a.shape[1] if ta else a.shape[2]
        angular = b.shape[1] if tb else inner
        channels = inner if tb else b.shape[2]
        cann = (os.environ.get('ICTC_ASCEND_VECTOR_BMM', 'auto') == 'auto'
                and a.shape[0] >= 16384 and channels >= 64
                and min(q, angular) >= 4 and q*angular >= 64)
        value = None if cann else try_apply(a, b, ta, tb)
        return value if value is not None else torch.bmm(
            a.transpose(-2, -1) if ta else a,
            b.transpose(-2, -1) if tb else b)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        ta, tb = ctx.spec
        needed = required_gradients((a, b), ctx.needs_input_grad[:2])
        da = db = None
        if needed[0]:
            da = (_Matrix.apply(b, grad, tb, True) if ta else
                  _Matrix.apply(grad, b, False, not tb))
        if needed[1]:
            db = (_Matrix.apply(grad, a, True, ta) if tb else
                  _Matrix.apply(a, grad, not ta, False))
        return da, db, None, None


def _output_parts(meta, value):
    rows, channels = value.shape[0], meta['channels']
    parts = value.split(meta['output_spans'], 1)
    result = {}
    for path, span, block in zip(meta['output_order'], meta['output_spans'], parts):
        width = span//channels
        result[path] = (block.reshape(rows,width,channels) if meta['layout']=='ir_mul'
                        else block.reshape(rows,channels,width).transpose(1,2))
    return result


class _Product(torch.autograd.Function):
    """Write each path directly and recompute bases only for gate adjoints."""
    @staticmethod
    def forward(ctx, meta, right, weights, *features):
        ctx.meta = meta
        ctx.save_for_backward(right, weights, *features)
        rows, channels = weights.shape[0], meta['channels']
        result = right.new_empty(rows, sum(meta['output_spans']))
        outputs = result.split(meta['output_spans'],1)
        targets = dict(zip(meta['output_order'],outputs))
        matrices = right.split(meta['matrix_spans'],1)
        path_weights = weights.unbind(1)
        for (i, items, width), matrix, left in zip(meta['groups'],matrices,features):
            base = _Matrix.apply(matrix.reshape(rows,width,items[0][1]),left,False,False)
            for (path, ni, no, _), block in zip(items,base.split([item[2] for item in items],1)):
                value = block*path_weights[path][:,None,:]
                if meta['layout']=='mul_ir':
                    value = value.transpose(1,2)
                targets[path].copy_(value.reshape(rows,-1))
            del base, block, value
        return result

    @staticmethod
    def backward(ctx, grad):
        meta = ctx.meta
        right, weights, *features = ctx.saved_tensors
        needed = required_gradients((right,weights,*features),ctx.needs_input_grad[1:])
        rows, channels = weights.shape[0], meta['channels']
        outputs = _output_parts(meta,grad)
        matrices = right.split(meta['matrix_spans'],1)
        path_weights = weights.unbind(1)
        dr, dw, dx = [], {}, []
        for index, ((i,items,width),matrix,left) in enumerate(zip(meta['groups'],matrices,features)):
            matrix = matrix.reshape(rows,width,items[0][1])
            if needed[0] or needed[2+index]:
                gated = torch.cat([outputs[path]*path_weights[path][:,None,:]
                                   for path,ni,no,offset in items],1)
                if needed[0]:
                    dr.append(_Matrix.apply(gated,left,False,True).reshape(rows,-1))
                dx.append(_Matrix.apply(matrix,gated,True,False) if needed[2+index] else None)
                del gated
            else:
                dx.append(None)
            if needed[1]:
                base = _Matrix.apply(matrix,left,False,False)
                for (path,ni,no,offset),block in zip(items,base.split([item[2] for item in items],1)):
                    dw[path] = (block*outputs[path]).sum(1)
                del base, block
        return (None, torch.cat(dr,1) if needed[0] else None,
                torch.stack([dw[path] for path in range(len(dw))],1) if needed[1] else None,
                *dx)


def tensor_product(op, features, harmonics, gates):
    """Preserve the public packed layout without intermediate output repacks."""
    owner = op.tp
    require_npu_fp32(features, harmonics, owner.weight,
                     *(() if gates is None else (gates,)))
    channels = owner.channels
    if features.ndim != 2 or features.shape[1] != channels*(op.input_lmax+1)**2:
        raise ValueError('Packed features must have shape [E,C*(L_in+1)^2]')
    rows = features.shape[0]
    if harmonics.shape != (rows, (op.edge_lmax+1)**2):
        raise ValueError('Harmonics must have shape [E,(L_edge+1)^2]')
    if gates is not None and gates.shape not in ((rows, owner.num_paths*channels),
                           (rows, owner.num_paths, channels)):
        raise ValueError('External gates must have shape [E,P*C] or [E,P,C]')
    if not rows:
        zero = sum(value.sum()*0 for value in (features, harmonics, owner.weight,
                                               *(() if gates is None else (gates,))))
        return features.new_empty(0, op.output_width)+zero
    blocks = {}
    widths = [channels*(2*l+1) for l in range(op.input_lmax+1)]
    for l, block in enumerate(features.split(widths, 1)):
        blocks[l] = (block.reshape(rows, 2*l+1, channels) if op.layout == 'ir_mul'
                     else block.reshape(rows, channels, 2*l+1).transpose(1, 2).contiguous())
    edge_degrees = tuple(range(op.edge_lmax+1))
    plan = _plan(owner, features, sorted(blocks), edge_degrees)
    cache_key = '_ascend_packed_groups_'+op.layout
    grouped = plan.get(cache_key)
    if grouped is None:
        by_input, offset = {}, 0
        for path, (i, j, k) in enumerate(owner.paths):
            ni, no = 2*i+1, 2*k+1
            by_input.setdefault(i, []).append((path, ni, no, offset))
            offset += ni*no
        groups = tuple((i, tuple(items), sum(item[2] for item in items))
                       for i, items in sorted(by_input.items()))
        order = [offset+a*no+q for i, items, width in groups
                 for path, ni, no, offset in items for q in range(no) for a in range(ni)]
        projection = plan['bond_projection'].index_select(
            1, torch.tensor(order, device=features.device, dtype=torch.int64)).contiguous()
        output_order = tuple(sorted(range(owner.num_paths),
                                   key=lambda p: (owner.paths[p][2], owner.path_offset[p])))
        path_spans = {p:channels*(2*k+1) for p,(i,j,k) in enumerate(owner.paths)}
        meta = dict(groups=groups,output_order=output_order,
                    output_spans=tuple(path_spans[p] for p in output_order),
                    matrix_spans=tuple(items[0][1]*width for i,items,width in groups),
                    channels=channels,layout=op.layout)
        grouped = projection, meta
        plan[cache_key] = grouped
    projection, meta = grouped
    bond = _FixedLinear.apply(harmonics, projection)
    weights = (owner.weight[None].expand(rows, -1, -1) if gates is None else
               gates.reshape(rows, owner.num_paths, channels)*owner.weight[None])
    return _Product.apply(meta,bond,weights,*[blocks[i] for i,items,width in meta['groups']])
