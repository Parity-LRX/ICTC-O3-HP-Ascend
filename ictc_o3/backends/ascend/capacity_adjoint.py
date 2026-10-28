"""Capacity adjoints with one node cotangent layout and group-local edge scratch."""
import os
import torch

def _matrix_product(a,b,transpose_a,transpose_b):
    from ictc_o3.ascendc.batched_vector import try_apply
    from .runtime import drain_graph_queue
    drain_graph_queue(a.device)
    value=try_apply(a,b,transpose_a,transpose_b,capacity=True)
    return value if value is not None else torch.bmm(
        a.transpose(-2,-1) if transpose_a else a,b.transpose(-2,-1) if transpose_b else b)


def _cotangents(owner, features, harmonics, keys, outputs, masked):
    """Convert the public node cotangent to each compact coupling group once."""
    from ictc_o3.backends.ascend.tensor_product import _plan, _destination_nodes
    from ictc_o3.ascendc.epilogue import supported
    sample = next(iter(features.values()))
    plan = _plan(owner, sample, sorted(features), sorted(harmonics))
    if '_ascendc_epilogue_groups' not in plan:
        return None
    projection, groups = plan['_ascendc_epilogue_groups']
    if (not groups or not supported(groups,masked)
            or os.environ.get('ICTC_ASCEND_EPILOGUE','auto')=='0'
            or os.environ.get('ICTC_ASCEND_COMPACT_OUTPUT', '1') != '1'):
        return None
    nodes, channels = _destination_nodes(owner, sample.shape[0]), sample.shape[1]
    cores = plan.get('_ascendc_epilogue_plans', {})
    if any((tuple(item[0] for item in group[4]), channels, str(sample.device), True) not in cores
           for group in groups):return None
    supplied = dict(zip(keys, outputs))
    blocks = []
    for degree in range(owner.lmax+1):
        count = owner.path_counts_by_l.get(degree, 0)
        value = supplied.get(degree)
        blocks.append(sample.new_zeros(nodes, count*(2*degree+1), channels) if value is None else
            value.reshape(nodes, count, channels, 2*degree+1).transpose(2, 3).reshape(nodes, count*(2*degree+1), channels))
    packed = torch.cat(blocks, 1)
    offsets = plan['_ascendc_output_widths']
    start, starts = 0, []
    for width in offsets:
        starts.append(start)
        start += width
    cache = plan.setdefault('_capacity_node_cotangent_indices', {})
    results = []
    for group in groups:
        pids = tuple(item[0] for item in group[4])
        if pids not in cache:
            indices = [starts[degree]+owner.path_offset[p]*width+a
                       for p, _, _, degree, _, width in group[4] for a in range(width)]
            cache[pids] = torch.tensor(indices, device=sample.device, dtype=torch.long)
        results.append(packed.index_select(1, cache[pids]).contiguous())
    return plan, projection, groups, tuple(results), nodes


def first_order_backward(ctx, outputs):
    """Accumulate node gradients while keeping recomputation scratch edge-local.

    Frozen radial projections and their hidden-feature adjoints are evaluated
    per coupling group. Higher derivatives retain the recursive native path.
    """
    from ictc_o3.ascendc.epilogue import _adjoints, _native
    from ictc_o3.ascendc.linear import matmul
    owner, fkeys, hkeys, chunk, linear = ctx.meta
    if linear is None:return None
    saved_weight, *tensors = ctx.saved_tensors
    saved_projection, saved_bias, *tensors = tensors
    if any(a is not b for a, b in zip(ctx.projection_reference, (linear.weight, linear.bias))):
        raise RuntimeError('Radial projection coefficients were replaced between forward and backward')
    if owner.weight is not ctx.weight_reference:
        raise RuntimeError('Tensor-product coefficients were replaced between forward and backward')
    nf, nh = len(fkeys), len(hkeys)
    ng = nf+nh
    gates, src, dst, mask = tensors[ng:]
    needed = ctx.needs_input_grad[5:]
    features, harmonics = dict(zip(fkeys, tensors[:nf])), dict(zip(hkeys, tensors[nf:ng]))
    prepared = _cotangents(owner, features, harmonics, ctx.keys, outputs, mask is not None)
    if prepared is None:return None
    plan, projection, groups, node_gradients, nodes = prepared
    sample = tensors[0]
    channels = sample.shape[1]
    angular = {k: value.transpose(1, 2).contiguous() for k, value in features.items()}
    result = [None]*len(tensors)
    feature_grad = {k: angular[k].new_zeros(angular[k].shape) for i, k in enumerate(fkeys) if needed[i]}
    for i in range(nf, ng+1):
        if needed[i]:result[i] = torch.zeros_like(tensors[i])
    gradient_sh = any(needed[nf:ng])
    old_groups = plan['_ascend_channel_groups'][1]
    metadata, start = [], 0
    for group, old_group, node_gradient in zip(groups, old_groups, node_gradients):
        degree, ni, width, _, items = group
        ids = torch.tensor([item[0] for item in old_group[4]], device=sample.device, dtype=torch.long)
        tp_weight = owner.weight.index_select(0, ids)
        bond_projection = projection[:, start:start+ni*width].contiguous()
        start += ni*width
        linear_weight = linear.weight.reshape(owner.num_paths, channels, linear.in_features).index_select(0, ids).reshape(-1, linear.in_features)
        linear_bias = (None if linear.bias is None else
            linear.bias.reshape(owner.num_paths, channels).index_select(0, ids).reshape(-1))
        key = tuple(item[0] for item in items), channels, str(sample.device), True
        core = plan['_ascendc_epilogue_plans'][key]
        metadata.append((group, ids, tp_weight, bond_projection, linear_weight, linear_bias, core, node_gradient))
    for begin in range(0, src.numel(), chunk):
        end = min(src.numel(), begin+chunk)
        rows = end-begin
        sh = torch.cat([value[begin:end] for value in harmonics.values()], -1).squeeze(1)
        dsh = torch.zeros_like(sh) if gradient_sh else None
        for group, ids, tp_weight, p, w, bias, core, node_gradient in metadata:
            degree, ni, width, _, items = group
            matrix = matmul(sh, p).reshape(rows, width, ni)
            edge = angular[degree].index_select(0, src[begin:end])
            base = _matrix_product(matrix, edge, False, False)
            weights = matmul(gates[begin:end], w.T.contiguous())
            if bias is not None:weights.add_(bias)
            weights = weights.reshape(rows, ids.numel(), channels)
            weights.mul_(tp_weight[None])
            proxy = dict(core)
            proxy['_ascendc_graph'] = dict(src=src[begin:end], dst=dst[begin:end], destination_nodes=nodes,
                mask=None if mask is None else mask[begin:end].reshape(-1).contiguous())
            needs = degree in feature_grad or gradient_sh, bool(needed[ng])
            grads = _adjoints(base, weights, node_gradient, proxy, needs)
            if grads is None:
                grads = tuple(_native(base, weights, proxy, i+1, node_gradient) if need else None
                              for i, need in enumerate(needs))
            db, dw = grads
            if dw is not None:
                dw.mul_(tp_weight[None])
                result[ng][begin:end].add_(matmul(dw.reshape(rows, -1), w))
            if db is not None:
                if degree in feature_grad:
                    dx = _matrix_product(matrix, db, True, False)
                    feature_grad[degree].index_add_(0, src[begin:end], dx)
                    del dx
                if gradient_sh:
                    dm = _matrix_product(db, edge, False, True)
                    dsh.add_(matmul(dm.reshape(rows, width*ni), p.T.contiguous()))
                    del dm
            del matrix, edge, base, weights, proxy, grads, db, dw
        if gradient_sh:
            offset = 0
            for i, key in enumerate(hkeys, nf):
                span = harmonics[key].shape[-1]
                if needed[i]:result[i][begin:end].copy_(dsh[:, offset:offset+span].unsqueeze(1))
                offset += span
        torch.npu.current_stream(gates.device).synchronize()
        del sh, dsh
    for i, key in enumerate(fkeys):
        if needed[i]:result[i] = feature_grad[key].transpose(1, 2)
    return (None, None, None, None, None, *result)
