# SPDX-License-Identifier: MIT
"""Fused Poisson geometry for live cells that do not request cell adjoints."""
import math
import torch
import triton
import triton.language as tl

_ALPHAS={}


@triton.jit
def _geometry(C,P,N,D,KP,KN,SP,SN,V,GRAPHS,K:tl.constexpr,PADDING:tl.constexpr,P0,P1,N0,N1,
              ALPHA,FLOOR:tl.constexpr,GREEN_FLOOR:tl.constexpr,ZERO:tl.constexpr,BLOCK:tl.constexpr):
    for graph in range(GRAPHS):
        cell=graph*9
        a=tl.load(C+cell);b=tl.load(C+cell+1);c=tl.load(C+cell+2)
        d=tl.load(C+cell+3);e=tl.load(C+cell+4);f=tl.load(C+cell+5)
        g=tl.load(C+cell+6)*PADDING
        h=tl.load(C+cell+7)*PADDING
        i=tl.load(C+cell+8)*PADDING
        u0=e*i-f*h;u1=f*g-d*i;u2=d*h-e*g
        v0=h*c-i*b;v1=i*a-g*c;v2=g*b-h*a
        w0=b*f-c*e;w1=c*d-a*f;w2=a*e-b*d
        det=(a*u0+b*u1)+c*u2
        safe=tl.where(tl.abs(det)>=1.e-12,det,tl.where(det>=0.,1.e-12,-1.e-12))
        volume=tl.maximum(tl.abs(det),FLOOR)
        if tl.program_id(0)==0:tl.store(V+graph,volume)
        for tile in range(tl.program_id(0),tl.cdiv(K,BLOCK),tl.num_programs(0)):
            freq=tile*BLOCK+tl.arange(0,BLOCK);valid=freq<K;row=graph*K+freq
            for partner in tl.static_range(2):
                if partner==0:
                    fx=tl.load(P+P0*freq,valid,other=0.);fy=tl.load(P+P0*freq+P1,valid,other=0.);fz=tl.load(P+P0*freq+2*P1,valid,other=0.)
                else:
                    fx=tl.load(N+N0*freq,valid,other=0.);fy=tl.load(N+N0*freq+N1,valid,other=0.);fz=tl.load(N+N0*freq+2*N1,valid,other=0.)
                # Match the reference's inverse, matrix product, then 2*pi scale.
                x=((fx*(u0/safe)+fy*(v0/safe))+fz*(w0/safe))*(2*math.pi)
                y=((fx*(u1/safe)+fy*(v1/safe))+fz*(w1/safe))*(2*math.pi)
                z=((fx*(u2/safe)+fy*(v2/safe))+fz*(w2/safe))*(2*math.pi)
                norm=tl.sqrt(tl.maximum((x*x+y*y)+z*z,FLOOR*FLOOR))
                green_norm=tl.maximum(norm,GREEN_FLOOR)
                value=(4*math.pi/(green_norm*green_norm))/volume*tl.load(D+freq,valid,other=0.)
                value=value*tl.exp(-(norm*norm)/(4*ALPHA*ALPHA))
                if ZERO:value=tl.where(norm>FLOOR,value,0.)
                if partner==0:
                    tl.store(KP+(graph*3)*K+freq,x,valid);tl.store(KP+(graph*3+1)*K+freq,y,valid);tl.store(KP+(graph*3+2)*K+freq,z,valid)
                    tl.store(SP+row,value,valid)
                else:
                    tl.store(KN+(graph*3)*K+freq,x,valid);tl.store(KN+(graph*3+1)*K+freq,y,valid);tl.store(KN+(graph*3+2)*K+freq,z,valid)
                    tl.store(SN+row,value,valid)


def geometry(kernel,cell,positive,negative,interior,deconv,*,exclude_k0):
    graphs=cell.shape[0];frequencies=positive.shape[0]
    kp=cell.new_empty(graphs,3,frequencies);kn=torch.empty_like(kp)
    sp=cell.new_empty(graphs,frequencies);sn=torch.empty_like(sp);volume=cell.new_empty(graphs)
    alpha=1./(math.sqrt(2.)*kernel.smearing_sigma)
    key=str(cell.device),cell.dtype,graphs,alpha
    if key not in _ALPHAS:
        if len(_ALPHAS)>=16:_ALPHAS.pop(next(iter(_ALPHAS)))
        _ALPHAS[key]=cell.new_full((graphs,),alpha)
    from .runtime import _vector_grid
    blocks=triton.cdiv(frequencies,256)
    grid=_vector_grid(blocks,cell.device)
    from .jit_launch import launch
    launch(_geometry,(grid,),cell.contiguous(),positive,negative,deconv,kp,kn,sp,sn,volume,
        graphs,frequencies,kernel.slab_padding_factor if kernel.boundary=='slab' else 1,
        *positive.stride(),*negative.stride(),
        alpha,kernel.k_norm_floor,kernel.green_kernel.k_norm_floor,exclude_k0,256,
        num_warps=4,enable_fp_fusion=False)
    return kp.transpose(1,2),kn.transpose(1,2),sp,sn,interior,_ALPHAS[key],volume,None
