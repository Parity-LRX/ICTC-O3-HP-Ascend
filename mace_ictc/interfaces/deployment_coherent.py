# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Prepared three-stream Cartesian TP for frozen FX/AOTInductor inference.

The neutral, real and imaginary messages share each Cartesian contraction.
Only the original radial gate bank is materialized. All preparation happens
before tracing; the exported first adjoint consists of ordinary Triton calls.
"""
from typing import Tuple

import torch


def _source(p, projection):
    from .deployment_triton import _sum
    c, hd = p['c'], projection['hdim']
    columns = [[] for _ in range(p['bdim'])]
    for h, rows in enumerate(projection['static_rows']):
        for col, value in rows:
            columns[col].append((h, value))
    lines = ['import triton', 'import triton.language as tl', '', '@triton.jit',
        'def kernel(X,H,W,G,R,I,S,D,M,V,O,DX,DH,DG,DR,DI,E,MODE:tl.constexpr,BLOCK:tl.constexpr):',
        '    row=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)',
        '    e=row//32;lane=row%32;channel=tl.program_id(1)*32+lane',
        f'    valid=(e<E)&(channel<{c})',
        '    src=tl.load(S+e,e<E,0);dst=tl.load(D+e,e<E,0)',
        '    mask=tl.load(M+e,e<E,0.)',
        f'    head=channel//{c//p["heads"]}']
    def emit(value, indent=1):
        lines.append('    '*indent+value)
    for h in range(hd):
        emit(f'h{h}=tl.load(H+e*{hd}+{h},e<E,0.)')
        emit(f'dh{h}=tl.full((BLOCK,),0.,tl.float32)')
    reduce_phase=(c//p['heads'])%32==0
    degrees=sorted({(row[-1]-1)//2 if p['angular'] else 0 for row in p['path_specs']})
    if reduce_phase:
        for degree in degrees:
            emit(f'pr{degree}=tl.full((BLOCK,),0.,tl.float32);pi{degree}=tl.full((BLOCK,),0.,tl.float32)')
    groups = {}
    for row in p['path_specs']:
        groups.setdefault((row[2], row[1]), []).append(row)
    for (xb, ni), paths in groups.items():
        for j in range(ni):
            emit(f'x{j}=tl.load(X+src*{p["xdim"]}+{xb+j*c}+channel,valid,0.)')
            emit(f'dx{j}=tl.full((BLOCK,),0.,tl.float32)')
        for path, _, _, bb, bs, ob, m in paths:
            phase = ((m-1)//2)*p['heads'] if p['angular'] else 0
            emit(f'phase={phase}+head')
            emit(f'cr=tl.load(R+e*{p["phase_count"]}+phase,valid,0.)')
            emit(f'ci=tl.load(I+e*{p["phase_count"]}+phase,valid,0.)')
            emit(f'w=tl.load(W+{path*c}+channel,valid,0.)')
            emit(f'g=tl.load(G+e*{p["p"]*c}+{path*c}+channel,valid,0.)')
            for j in range(ni):
                for mag in range(m):
                    terms = columns[bb+j*bs+mag]
                    if terms:
                        emit(f'b{j}_{mag}='+_sum(f'{value!r}*h{h}' for h,value in terms))
            emit('if MODE==0:')
            for mag in range(m):
                value = _sum(f'x{j}*b{j}_{mag}' for j in range(ni)
                             if columns[bb+j*bs+mag])
                emit(f'y=({value})*w*g*mask',2)
                for stream, factor in enumerate(('1.', 'cr', 'ci')):
                    emit(f'tl.atomic_add(O+(dst*3+{stream})*{p["odim"]}+{ob+mag*c}+channel,y*{factor},valid,sem="relaxed")',2)
            emit('else:')
            emit('dg=tl.full((BLOCK,),0.,tl.float32)',2)
            emit('dr=tl.full((BLOCK,),0.,tl.float32);di=tl.full((BLOCK,),0.,tl.float32)',2)
            for mag in range(m):
                for stream, name in enumerate(('v','vr','vi')):
                    emit(f'{name}{mag}=tl.load(V+(dst*3+{stream})*{p["odim"]}+{ob+mag*c}+channel,valid,0.)*mask',2)
                emit(f'z{mag}=v{mag}+cr*vr{mag}+ci*vi{mag}',2)
                value = _sum(f'x{j}*b{j}_{mag}' for j in range(ni)
                             if columns[bb+j*bs+mag])
                emit(f'product={value}',2)
                emit(f'dg+=product*z{mag}*w;dr+=product*vr{mag}*w*g;di+=product*vi{mag}*w*g',2)
                for j in range(ni):
                    if not columns[bb+j*bs+mag]:
                        continue
                    emit(f'dx{j}+=b{j}_{mag}*z{mag}*w*g',2)
                    emit(f'db=x{j}*z{mag}*w*g',2)
                    for h, value in columns[bb+j*bs+mag]:
                        emit(f'dh{h}+={value!r}*db',2)
            emit(f'tl.store(DG+e*{p["p"]*c}+{path*c}+channel,dg,valid)',2)
            # Each head owns a contiguous channel slice. Atomic sums retain
            # every channel/path, including ragged channel tiles.
            if reduce_phase:
                degree=phase//p['heads']
                emit(f'pr{degree}+=dr;pi{degree}+=di',2)
            else:
                emit(f'tl.atomic_add(DR+e*{p["phase_count"]}+phase,dr,valid,sem="relaxed")',2)
                emit(f'tl.atomic_add(DI+e*{p["phase_count"]}+phase,di,valid,sem="relaxed")',2)
        emit('if MODE==1:')
        for j in range(ni):
            emit(f'tl.atomic_add(DX+src*{p["xdim"]}+{xb+j*c}+channel,dx{j},valid,sem="relaxed")',2)
    emit('if MODE==1:')
    emit('edge=tl.program_id(0)*(BLOCK//32)+tl.arange(0,BLOCK//32)',2)
    if reduce_phase:
        for degree in degrees:
            for pointer,value in (('DR',f'pr{degree}'),('DI',f'pi{degree}')):
                emit(f'tl.atomic_add({pointer}+edge*{p["phase_count"]}+{degree*p["heads"]}+tl.program_id(1)*32//{c//p["heads"]},'
                    f'tl.sum(tl.reshape({value},(BLOCK//32,32)),1),edge<E,sem="relaxed")',2)
    for h in range(hd):
        emit(f'tl.atomic_add(DH+edge*{hd}+{h},tl.sum(tl.reshape(dh{h},(BLOCK//32,32)),1),edge<E,sem="relaxed")',2)
    return '\n'.join(lines)+'\n'


def _csr_source(p,projection,paths,streams):
    from .deployment_triton import _sum
    c,hd=p['c'],projection['hdim']
    columns=[[] for _ in range(p['bdim'])]
    for h,rows in enumerate(projection['static_rows']):
        for col,value in rows:columns[col].append((h,value))
    lines=['import triton','import triton.language as tl','','@triton.jit',
        'def kernel(X,H,W,G,R,I,S,ORDER,PTR,M,O,N,BLOCK:tl.constexpr):',
        '    row=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)',
        '    dst=row//32;channel=tl.program_id(1)*32+row%32',
        f'    valid=(dst<N)&(channel<{c})',
        '    start=tl.load(PTR+dst,dst<N,0);stop=tl.load(PTR+dst+1,dst<N,0)',
        '    count=tl.max(stop-start,0)']
    def emit(value,indent=1):lines.append('    '*indent+value)
    for path,_,_,_,_,_,m in paths:
        emit(f'w{path}=tl.load(W+{path*c}+channel,channel<{c},0.)')
        for mag in range(m):
            for stream in range(streams):emit(f'a{path}_{mag}_{stream}=tl.full((BLOCK,),0.,tl.float32)')
    emit('for k in range(count):')
    emit('active=valid&(start+k<stop)',2)
    emit('e=tl.load(ORDER+start+k,active,0);src=tl.load(S+e,active,0)',2)
    emit('mask=tl.load(M+e,active,0.)',2)
    for h in range(hd):emit(f'h{h}=tl.load(H+e*{hd}+{h},active,0.)',2)
    groups={}
    for path in paths:groups.setdefault((path[2],path[1]),[]).append(path)
    for (xb,ni),rows in groups.items():
        for j in range(ni):emit(f'x{j}=tl.load(X+src*{p["xdim"]}+{xb+j*c}+channel,active,0.)',2)
        for path,_,_,bb,bs,ob,m in rows:
            if streams==3:
                phase=((m-1)//2)*p['heads'] if p['angular'] else 0
                emit(f'phase={phase}+channel//{c//p["heads"]}',2)
                emit(f'cr=tl.load(R+e*{p["phase_count"]}+phase,active,0.)',2)
                emit(f'ci=tl.load(I+e*{p["phase_count"]}+phase,active,0.)',2)
            emit(f'gate=tl.load(G+e*{p["p"]*c}+{path*c}+channel,active,0.)*w{path}*mask',2)
            for mag in range(m):
                terms=[]
                for j in range(ni):
                    values=columns[bb+j*bs+mag]
                    if values:terms.append(f'x{j}*('+_sum(f'{value!r}*h{h}' for h,value in values)+')')
                emit('value=('+_sum(terms)+')*gate',2)
                for stream,factor in enumerate(('1.','cr','ci')[:streams]):
                    emit(f'a{path}_{mag}_{stream}+=value*{factor}',2)
    for path,_,_,_,_,ob,m in paths:
        for mag in range(m):
            for stream in range(streams):
                emit(f'tl.store(O+(dst*{streams}+{stream})*{p["odim"]}+{ob+mag*c}+channel,a{path}_{mag}_{stream},valid)')
    return '\n'.join(lines)+'\n'


def prepare_csr_forward(p,projection,streams):
    """Live destination ownership, with no captured/cached input topology."""
    import triton
    from torch.library import wrap_triton
    from .deployment_triton import _module
    groups=[];current=[];width=0
    for row in p['path_specs']:
        if current and width+row[-1]>16:
            groups.append(current);current=[];width=0
        current.append(row);width+=row[-1]
    if current:groups.append(current)
    kernels=[_module(_csr_source(p,projection,rows,streams),'prepared_csr').kernel for rows in groups]
    def evaluate(x,h,w,g,r,i,s,d,m):
        n=x.shape[0];e=s.shape[0]
        order=torch.argsort(d,stable=True)
        counts=torch.zeros(n,device=d.device,dtype=torch.int64).index_add(0,d,torch.ones_like(d))
        ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
        out=x.new_empty(n,streams,p['odim'])
        for kernel in kernels:
            wrap_triton(kernel)[(triton.cdiv(n*32,128),triton.cdiv(p['c'],32))](
                x,h,w,g,r,i,s,order,ptr,m,out,n,128,num_warps=4,enable_fp_fusion=False)
        return out
    return evaluate


def prepare_coherent(layer, device, *, csr_forward=False):
    """Return a tensor-only three-stream callable, preserving checkpoint paths."""
    import triton
    from torch.library import triton_op, wrap_triton
    from mace_ictc.models.fused_message import (
        _plan, _channel_node, _channel_cotangent,
        _normal_message_output, _normal_feature_gradient,
    )
    from mace_ictc.models.bond_projection import projection_plan
    from .deployment_triton import _module, _name
    tp = layer.tp
    p = dict(_plan(tp, list(range(layer.input_lmax+1)), device, True),
             heads=layer.phase_heads, angular=layer.phase_angular_channels)
    p['phase_count'] = p['heads']*((layer.target_lmax+1) if p['angular'] else 1)
    projection = projection_plan(tp, tp._get_bond_projectors(device,torch.float32),device,torch.float32)
    # Partition by input degree to bound compiler/register work at high L/C.
    groups = {}
    for row in p['path_specs']:
        groups.setdefault(row[2], []).append(row)
    kernels = [_module(_source(dict(p,path_specs=rows),projection),'coherent').kernel
               for rows in groups.values()]
    tiles, hd = triton.cdiv(p['c'],32), projection['hdim']
    csr=prepare_csr_forward(p,projection,3) if csr_forward else None

    @triton_op(_name('coherent_backward'),mutates_args={})
    def backward_op(x:torch.Tensor,h:torch.Tensor,w:torch.Tensor,g:torch.Tensor,
                    r:torch.Tensor,i:torch.Tensor,s:torch.Tensor,d:torch.Tensor,
                    m:torch.Tensor,v:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        e=s.shape[0];n=x.shape[0]
        xc=_channel_node(x,p)
        vc=_channel_cotangent(v.reshape(n*3,p['odim']),p)
        dx=torch.zeros_like(xc);dh=torch.zeros_like(h);dg=torch.empty_like(g)
        dr=torch.zeros_like(r);di=torch.zeros_like(i)
        for kernel in kernels:
            wrap_triton(kernel)[(triton.cdiv(e*32,128),tiles)](
                xc,h,w,g,r,i,s,d,m,vc,vc,dx,dh,dg,dr,di,e,1,128,
                num_warps=4,enable_fp_fusion=False)
        return _normal_feature_gradient(dx,p),dh,dg,dr,di

    @triton_op(_name('coherent'),mutates_args={})
    def forward_op(x:torch.Tensor,h:torch.Tensor,w:torch.Tensor,g:torch.Tensor,
                   r:torch.Tensor,i:torch.Tensor,s:torch.Tensor,d:torch.Tensor,
                   m:torch.Tensor)->torch.Tensor:
        e=s.shape[0];n=x.shape[0];xc=_channel_node(x,p)
        if csr is not None:
            out=csr(xc,h,w,g,r,i,s,d,m)
        else:
            out=x.new_zeros(n,3,p['odim'])
            for kernel in kernels:
                wrap_triton(kernel)[(triton.cdiv(e*32,128),tiles)](
                    xc,h,w,g,r,i,s,d,m,out,out,xc,h,g,r,i,e,0,128,
                    num_warps=4,enable_fp_fusion=False)
        return _normal_message_output(out.reshape(n*3,p['odim']),p).reshape(n,3,p['odim'])

    def setup(ctx,inputs,output):
        ctx.save_for_backward(*inputs)
    def backward(ctx,v):
        x,h,w,g,r,i,s,d,m=ctx.saved_tensors
        dx,dh,dg,dr,di=backward_op(x,h,w,g,r,i,s,d,m,v.contiguous())
        return dx,dh,None,dg,dr,di,None,None,None
    forward_op.register_autograd(backward,setup_context=setup)

    def evaluate(blocks,harmonics,gates,s,d,mask,cos,sin):
        n=blocks[0].shape[0];e=s.shape[0]
        x=torch.cat([blocks[l].reshape(n,p['c']*(2*l+1)) for l in sorted(blocks)],-1).contiguous()
        h=torch.cat([harmonics[l].squeeze(-2) for l in projection['keys']],-1).contiguous()
        mask=x.new_ones(e) if mask is None else mask.to(x).reshape(e).contiguous()
        out=forward_op(x,h,tp.weight,gates.reshape(e,p['p'],p['c']).contiguous(),
            cos.reshape(e,p['phase_count']).contiguous(),sin.reshape(e,p['phase_count']).contiguous(),
            s.contiguous(),d.contiguous(),mask)
        return {l:out[:,:,p['ooffset'][l]:p['ooffset'][l]+p['c']*p['counts'].get(l,0)*(2*l+1)].reshape(
                    n,3,p['c']*p['counts'].get(l,0),2*l+1) for l in range(tp.lmax+1)}
    return evaluate
