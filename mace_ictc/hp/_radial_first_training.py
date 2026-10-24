# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Consume bounded first-TP gate VJPs in the live radial projection.

The wide gate cotangent is tile-local in destination CSR order. Hidden and
bank gradients are projected before that storage is reused. The custom radial
adjoint owns the corresponding complete mixed derivative, including both banks.
"""
import torch

from ._radial_cached_training import _once

_EXTENSIONS = {}


def _extension(p, projection, needx, needh, schedule):
    from .coherent import _HEADERS, _outputs, _adjoint_groups
    from ._coherent_destination import _source
    from torch.utils.cpp_extension import load_inline
    p = {**p, 'factor_edge_scalars': True, 'coalesced_edge_partials': True,
         'componentwise_first': schedule in ('compact', 'compact_grouped')}
    key = (p['path_specs'], p['c'], p['xdim'], p['odim'], p['phase_heads'],
           p['phase_angular'], tuple(tuple(row) for row in projection['static_rows']),
           needx, needh, schedule)
    if key in _EXTENSIONS:
        return _EXTENSIONS[key]
    code = _source(p, projection, 1, atomic_x=True, needx=needx, needh=needh,
                   specialized=schedule in ('grouped', 'compact_grouped'))
    divisor = (p['c']//32 if schedule in ('grouped', 'compact_grouped') else
               (p['c']//32)*len(_adjoint_groups(p)))
    marker = f'const int64_t dst=task/{divisor};'
    code = _once(code, marker, marker+'\nconst int64_t edge_lo=PTR[dst]>BEGIN?PTR[dst]:BEGIN;'
        'const int64_t edge_hi=PTR[dst+1]<END?PTR[dst+1]:END;if(edge_lo>=edge_hi)return;')
    code = _once(code, 'int64_t N,int64_t E,bool masked,',
        'int64_t N,int64_t E,int64_t BEGIN,int64_t END,bool masked,')
    code = _once(code, 'torch::Tensor pi) {', 'torch::Tensor pi,int64_t BEGIN,int64_t END) {')
    code = code.replace('x.size(0),src.numel(),mask.numel()!=0,true,true,true,true,true);',
        'x.size(0),src.numel(),BEGIN,END,mask.numel()!=0,true,true,true,true,true);')
    loop = 'index=PTR[dst]+split;index<PTR[dst+1];index+=1'
    if code.count(loop) != len(_adjoint_groups(p)):
        raise RuntimeError('Missing bounded first-VJP neighbor loops')
    code = code.replace(loop, 'index=edge_lo+split;index<edge_hi;index+=1')
    writer, reduction = code.split('__global__ void coherent_destination_dx_reduce', 1)
    writer = writer.replace('DG[e*', 'DG[(index-BEGIN)*')
    for target in ('DH', 'DR', 'DI'):
        writer = writer.replace(target+'[(e*', target+'[((index-BEGIN)*')
    code = writer+'__global__ void coherent_destination_dx_reduce'+reduction
    code = _once(code, 'float* DH,float* DR,float* DI,int64_t edges)',
        'float* DH,float* DR,float* DI,const int64_t* ORDER,int64_t BEGIN,int64_t edges)')
    for target, width in (('DH', projection['hdim']), ('DR', p['phase_count']), ('DI', p['phase_count'])):
        code = _once(code, target+'[i]=', f'{target}[ORDER[BEGIN+i/{width}]*{width}+i%{width}]=')
    code = _once(code, '(src.numel()*', '((END-BEGIN)*')
    code = _once(code, 'di.data_ptr<float>(),src.numel());',
        'di.data_ptr<float>(),order.data_ptr<int64_t>(),BEGIN,END-BEGIN);')
    function = 'coherent_destination'
    declaration = '#include <torch/extension.h>\nvoid '+function+'('+','.join(
        ['torch::Tensor']*(21+len(_outputs(p)))+['int64_t']*2)+');\n'
    name = (f"ictc_radial_first_v2_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}"
            f"_a{int(p['phase_angular'])}_x{int(needx)}_h{int(needh)}_{schedule}")
    _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration,
        cuda_sources=_HEADERS+code, functions=[function], extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--ptxas-options=-v'], verbose=False)
    return _EXTENSIONS[key]


def first(x, sh, z, bank, r, i, gates, w, src, dst, mask, p, projection, needed, grads, tile_edges):
    from ._coherent_training import _physical_gradients
    from .coherent import _adjoint_groups
    from mace_ictc.models.message_topology import topology
    empty = x.new_empty(0)
    dx = torch.zeros_like(x) if needed[0] else empty
    dh = torch.empty_like(sh) if needed[1] else empty
    dr, di = torch.empty_like(r), torch.empty_like(i)
    dz = torch.empty_like(z) if needed[2] else None
    db = torch.zeros_like(bank) if needed[3] else None
    if not src.numel():
        return (dx if needed[0] else None, dh if needed[1] else None,
                dz, db, dr if needed[4] else None, di if needed[5] else None)
    if not (needed[2] or needed[3]):
        from ._coherent_training import _first
        values = _first(x, sh, w, gates, r, i, src, dst, mask, p, projection,
            (needed[0], needed[1], False, needed[4], needed[5]), grads)
        return values[0], values[1], None, None, values[3], values[4]
    tile = min(tile_edges, src.numel())
    dg = z.new_empty(tile, bank.shape[0])
    slots = (p['c']//32)*len(_adjoint_groups(p))
    ph = sh.new_empty(tile, slots, projection['hdim']) if needed[1] else empty
    pr, pi = r.new_empty(tile, slots, p['phase_count']), i.new_empty(tile, slots, p['phase_count'])
    physical = _physical_gradients(x, p, grads)
    ptr, order = topology(dst, x.shape[0])
    schedule = p.get('first_schedule', 'pruned')
    if schedule == 'legacy':
        schedule = 'pruned'
    extension = _extension(p, projection, needed[0], needed[1], schedule)
    for begin in range(0, src.numel(), tile):
        end = min(begin+tile, src.numel())
        count, index = end-begin, order[begin:end]
        extension.coherent_destination(x, sh, w, gates, r, i, src, mask,
            *physical, ptr, order, src.new_empty(0), src.new_empty(0),
            dx, dh, dg, dr, di, empty, ph, pr, pi, begin, end)
        if needed[2]:
            dz.index_copy_(0, index, dg[:count] @ bank)
        if needed[3]:
            db.addmm_(dg[:count].T, z.index_select(0, index))
    return (dx if needed[0] else None, dh if needed[1] else None, dz, db,
            dr if needed[4] else None, di if needed[5] else None)
