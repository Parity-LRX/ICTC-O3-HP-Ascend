"""Path-preserving ICTC tensor products using shared-geometry NPU GEMMs."""
import torch
import os

from mace_ictc.models.sparse_weighted_product import sparse_product_plan
from .sparse_product import sparse_product


def _plan(owner, sample, input_degrees, edge_degrees):
    # Only immutable mathematical constants are cached. Live gates and learned
    # weights are always inputs to the sparse map and its adjoints.
    constants = owner._get_cg_list(torch.device("cpu"), torch.float64)
    preproject = getattr(owner, "_ascend_tp_preproject", True)
    key = (str(sample.device), tuple(input_degrees), tuple(edge_degrees),
           tuple((id(c), c._version) for c in constants), tuple(owner.paths), preproject)
    cached = getattr(owner, "_ascend_tp_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    a_offsets, b_offsets, o_offsets = {}, {}, {}
    a = b = out = 0
    for degree in input_degrees:
        a_offsets[degree] = a
        a += 2*degree+1
    for degree in edge_degrees:
        b_offsets[degree] = b
        b += 2*degree+1
    for degree in range(owner.lmax+1):
        o_offsets[degree] = out
        out += owner.path_counts_by_l.get(degree, 0)*(2*degree+1)
    entries, projection_entries, bond_width = [], [], 0
    for p, ((l1, l2, l3), coefficients) in enumerate(zip(owner.paths, constants)):
        if l1 not in a_offsets or l2 not in b_offsets:
            continue
        start = o_offsets[l3]+owner.path_offset[p]*(2*l3+1)
        if preproject:
            for i in range(2*l1+1):
                for k in range(2*l3+1):
                    column = bond_width+i*(2*l3+1)+k
                    entries.append((a_offsets[l1]+i, column, p, start+k, 1.))
                    for j in range(2*l2+1):
                        value = float(coefficients[i, j, k])
                        if value != 0.:
                            projection_entries.append((b_offsets[l2]+j, column, value))
            bond_width += (2*l1+1)*(2*l3+1)
        else:
            for i, j, k in torch.nonzero(coefficients, as_tuple=False).tolist():
                entries.append((a_offsets[l1]+i, b_offsets[l2]+j, p, start+k,
                                float(coefficients[i, j, k])))
    result = sparse_product_plan(entries, (a, bond_width if preproject else b, owner.num_paths, out),
                                 sample.device, sample.dtype)
    if preproject:
        projection = torch.zeros(b, bond_width, device="cpu", dtype=sample.dtype)
        for row, column, value in projection_entries:
            projection[row, column] = value
        result["bond_projection"] = projection.to(sample.device)
    owner._ascend_tp_cache = key, result
    return result


def _sparse_tensor_product(owner, features, harmonics, gates=None):
    input_degrees, edge_degrees = sorted(features), sorted(harmonics)
    left = torch.cat([features[l] for l in input_degrees], -1)
    right = torch.cat([harmonics[l] for l in edge_degrees], -1)
    plan = _plan(owner, left, input_degrees, edge_degrees)
    rows, channels = left.shape[:2]
    if "bond_projection" in plan:
        projection = plan["bond_projection"]
        from .structured_product import _FixedLinear
        right = (_FixedLinear.apply(right.squeeze(1), projection).unsqueeze(1) if rows
                 else right.reshape(0, 1, projection.shape[1]))
    weight = owner.weight.T.unsqueeze(0)
    if gates is None:
        output = sparse_product(left, right, weight, plan)
    else:
        if gates.shape not in ((rows, owner.num_paths*channels), (rows, owner.num_paths, channels)):
            raise ValueError("External gates must have shape [E,P*C] or [E,P,C]")
        live_weight = gates.reshape(rows, owner.num_paths, channels).transpose(1, 2)*weight
        output = sparse_product(left, right, live_weight, plan, bank=False)
    result, offset = {}, 0
    for degree in range(owner.lmax+1):
        count = owner.path_counts_by_l.get(degree, 0)
        width = 2*degree+1
        span = count*width
        result[degree] = output[:, :, offset:offset+span].reshape(rows, channels, count, width).transpose(1, 2).reshape(rows, count*channels, width)
        offset += span
    return result


def tensor_product(owner, features, harmonics, gates=None):
    """Apply the original map with shared geometry and per-path FP32 GEMMs."""
    from mace_ictc.ascendc.runtime import native_graph_enabled
    if native_graph_enabled():
        return _sparse_tensor_product(owner,features,harmonics,gates)
    sample=next(iter(features.values()))
    rows,channels=sample.shape[:2]
    if (not rows or not getattr(owner,"_ascend_tp_preproject",True)
            or getattr(owner,"_ascend_tp_schedule","structured")=="sparse"):
        return _sparse_tensor_product(owner,features,harmonics,gates)
    from .runtime import require_npu_fp32
    require_npu_fp32(owner.weight,*features.values(),*harmonics.values(),
                     *(() if gates is None else (gates,)))
    if gates is not None and gates.shape not in (
            (rows,owner.num_paths*channels),(rows,owner.num_paths,channels)):
        raise ValueError("External gates must have shape [E,P*C] or [E,P,C]")
    plan=_plan(owner,sample,sorted(features),sorted(harmonics))
    if getattr(owner,"_ascend_tp_schedule","structured")=="structured":
        from .structured_product import tensor_product as structured_product
        return structured_product(owner,features,harmonics,gates,plan)
    sh=torch.cat([harmonics[l] for l in sorted(harmonics)],-1).squeeze(1)
    bond=sh@plan["bond_projection"]
    weights=(owner.weight[None] if gates is None else
             gates.reshape(rows,owner.num_paths,channels)*owner.weight[None])
    output={l:[None]*owner.path_counts_by_l.get(l,0) for l in range(owner.lmax+1)}
    offset=0
    for path,(l1,l2,l3) in enumerate(owner.paths):
        if l1 not in features or l2 not in harmonics:
            continue
        width_in,width_out=2*l1+1,2*l3+1
        matrix=bond[:,offset:offset+width_in*width_out].reshape(rows,width_in,width_out)
        block=torch.bmm(features[l1],matrix)*weights[:,path,:,None]
        output[l3][owner.path_offset[path]]=block
        offset+=width_in*width_out
    return {l:torch.cat([sample.new_zeros(rows,channels,2*l+1) if block is None else block
                         for block in blocks],1) if blocks else sample.new_empty(rows,0,2*l+1)
            for l,blocks in output.items()}


def _edge_working_bytes(owner, sample, features, harmonics):
    """Estimate message storage from the actual channel and angular widths."""
    channels = sample.shape[1]
    angular = sum(2*l+1 for l in features)
    output = sum(count*(2*l+1) for l,count in owner.path_counts_by_l.items())
    bond = sum((2*i+1)*(2*k+1) for i,j,k in owner.paths
               if i in features and j in harmonics)
    return sample.element_size()*(channels*(angular+output+2*owner.num_paths)+bond)


def _automatic_edge_chunk(owner, sample, features, harmonics):
    """Estimate forward scratch from the actual channel and angular widths."""
    bytes_per_edge = _edge_working_bytes(owner, sample, features, harmonics)
    limit = max(1, min(131072, (3*1024**3)//max(1, bytes_per_edge)))
    return (limit//128)*128 if limit >= 128 else limit


def graph_message(owner, features, harmonics, gates, src, dst, mask=None,
                  *, edge_chunk_size=16384):
    """Bound edge intermediates while retaining ordinary differentiable gathers."""
    from mace_ictc.ascendc.runtime import native_graph_enabled,is_enabled,mode
    schedule=getattr(owner,'_ascend_message_schedule','auto')
    if (is_enabled() and mode()=='performance' and schedule!='legacy'
            and getattr(owner,'_ascend_tp_preproject',True)):
        sample=next(iter(features.values()))
        chosen=(_automatic_edge_chunk(owner,sample,features,harmonics)
                if edge_chunk_size is None else edge_chunk_size)
        policy=os.environ.get('ICTC_ASCEND_VOLUME_DISPATCH','1')
        if policy not in ('0','1'):
            raise ValueError('The message-volume dispatch policy must be zero or one')
        # Wide/angular messages amortize grouping even with a small edge tile.
        width=sum(2*k+1 for i,j,k in owner.paths if i in features and j in harmonics)
        volume=min(src.numel(),chosen)*sample.shape[1]*width
        minimum=int(os.environ.get('ICTC_ASCEND_EPILOGUE_MIN_ELEMENTS',str(64*1024**2)))
        if minimum<1:
            raise ValueError('The message minimum working set must be positive')
        if (schedule=='grouped' or (src.numel()>=65536 and chosen>=32768)
                or (policy=='1' and volume>=minimum)):
            from .channel_message import graph_message as channel_message
            return channel_message(owner,features,harmonics,gates,src,dst,mask,edge_chunk_size=chosen)
    if (native_graph_enabled() and getattr(owner,'_ascend_tp_preproject',True)
            and (mask is None or not mask.requires_grad)):
        from mace_ictc.ascendc.graph import graph_message as native_graph
        return native_graph(owner,features,harmonics,gates,src,dst,mask,edge_chunk_size=edge_chunk_size)
    sample = next(iter(features.values()))
    if edge_chunk_size is None:
        edge_chunk_size = _automatic_edge_chunk(owner, sample, features, harmonics)
    nodes, channels = sample.shape[:2]
    output = {l: sample.new_zeros(nodes, owner.path_counts_by_l.get(l, 0)*channels, 2*l+1)
              for l in range(owner.lmax+1)}
    if src.numel() == 0:
        zero = owner.weight.sum()*0+gates.sum()*0
        zero = zero+sum(value.sum()*0 for value in (*features.values(), *harmonics.values()))
        return {l: value+zero for l, value in output.items()}
    policy=os.environ.get('ICTC_ASCEND_CHUNK_SPLIT','1')
    if policy not in ('0','1'):
        raise ValueError('The shared chunk-adjoint policy must be zero or one')
    split=policy=='1'
    # One split joins chunk cotangents without a full-edge gradient per slice.
    edges=src.numel()
    harmonic_parts={l:((value,) if edges<=edge_chunk_size else value.split(edge_chunk_size,0))
                    for l,value in harmonics.items()} if split else {}
    gate_parts=((gates,) if edges<=edge_chunk_size else gates.split(edge_chunk_size,0)) if split else ()
    for chunk,begin in enumerate(range(0, src.numel(), edge_chunk_size)):
        end = min(begin+edge_chunk_size, src.numel())
        edge_features = {l: value.index_select(0, src[begin:end]) for l, value in features.items()}
        edge_harmonics = {l: harmonic_parts[l][chunk] if split else value[begin:end] for l, value in harmonics.items()}
        value = tensor_product(owner, edge_features, edge_harmonics, gate_parts[chunk] if split else gates[begin:end])
        for l, block in value.items():
            if mask is not None:
                block = block*mask[begin:end].reshape(-1, 1, 1)
            output[l] = output[l].index_add(0, dst[begin:end], block)
    return output
