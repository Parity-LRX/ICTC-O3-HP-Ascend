# SPDX-License-Identifier: MIT
"""Direct multipole amplitude maps and paired linear adjoints for fixed cells."""
import torch
import triton
import triton.language as tl


@triton.jit
def _component(x,y,z,I:tl.constexpr):
    if I==0:return x
    elif I==1:return y
    else:return z


@triton.jit
def _quad(x,y,z,I:tl.constexpr,COMPACT:tl.constexpr):
    if COMPACT:
        if I==0:return x*x
        elif I==1:return y*y
        elif I==2:return z*z
        elif I==3:return x*y
        elif I==4:return x*z
        else:return y*z
    else:return _component(x,y,z,I//3)*_component(x,y,z,I%3)


@triton.jit
def _map(S,G,P,N,O,GRAPHS,K,C:tl.constexpr,FIELDS:tl.constexpr,MU:tl.constexpr,Q:tl.constexpr,
         COMPACT:tl.constexpr,ADJOINT:tl.constexpr,S0,S1,S2,P0,P1,P2,N0,N1,N2,BLOCK:tl.constexpr):
    for graph in range(GRAPHS):
        for channel in range(C):
            for tile in range(tl.program_id(0),tl.cdiv(K,BLOCK),tl.num_programs(0)):
                freq=(tile*BLOCK+tl.arange(0,BLOCK)).to(tl.int64);valid=freq<K
                px=tl.load(P+graph*P0+freq*P1,valid,other=0.)
                py=tl.load(P+graph*P0+freq*P1+P2,valid,other=0.)
                pz=tl.load(P+graph*P0+freq*P1+2*P2,valid,other=0.)
                nx=tl.load(N+graph*N0+freq*N1,valid,other=0.)
                ny=tl.load(N+graph*N0+freq*N1+N2,valid,other=0.)
                nz=tl.load(N+graph*N0+freq*N1+2*N2,valid,other=0.)
                if ADJOINT:
                    pr=tl.load(G+((graph*C+channel)*4)*K+freq,valid,other=0.);pi=tl.load(G+((graph*C+channel)*4+1)*K+freq,valid,other=0.)
                    nr=tl.load(G+((graph*C+channel)*4+2)*K+freq,valid,other=0.);ni=tl.load(G+((graph*C+channel)*4+3)*K+freq,valid,other=0.)
                    offset=((graph*(C*FIELDS)+channel)*K+freq)*2
                    tl.store(O+offset,pr+nr,valid);tl.store(O+offset+1,pi-ni,valid)
                else:
                    offset=graph*S0+freq*S1+channel*S2
                    pr=tl.load(S+offset,valid,other=0.);pi=tl.load(S+offset+1,valid,other=0.)
                    nr=pr;ni=-pi
                if MU:
                    for index in tl.static_range(3):
                        p=_component(px,py,pz,index);n=_component(nx,ny,nz,index)
                        field=C+3*channel+index
                        if ADJOINT:
                            offset=((graph*(C*FIELDS)+field)*K+freq)*2
                            tl.store(O+offset,p*pi+n*ni,valid)
                            tl.store(O+offset+1,-p*pr+n*nr,valid)
                        else:
                            offset=graph*S0+freq*S1+field*S2
                            real=tl.load(S+offset,valid,other=0.);imag=tl.load(S+offset+1,valid,other=0.)
                            pr=pr-p*imag;pi=pi+p*real;nr=nr+n*imag;ni=ni+n*real
                if Q:
                    for index in tl.static_range(6 if COMPACT else 9):
                        p=_quad(px,py,pz,index,COMPACT);n=_quad(nx,ny,nz,index,COMPACT)
                        field=C*(4 if MU else 1)+(6 if COMPACT else 9)*channel+index
                        if ADJOINT:
                            offset=((graph*(C*FIELDS)+field)*K+freq)*2
                            tl.store(O+offset,-.5*(p*pr+n*nr),valid)
                            tl.store(O+offset+1,-.5*p*pi+.5*n*ni,valid)
                        else:
                            offset=graph*S0+freq*S1+field*S2
                            real=tl.load(S+offset,valid,other=0.);imag=tl.load(S+offset+1,valid,other=0.)
                            pr=pr-.5*p*real;pi=pi-.5*p*imag;nr=nr-.5*n*real;ni=ni+.5*n*imag
                if not ADJOINT:
                    tl.store(O+((graph*C+channel)*4)*K+freq,pr,valid);tl.store(O+((graph*C+channel)*4+1)*K+freq,pi,valid)
                    tl.store(O+((graph*C+channel)*4+2)*K+freq,nr,valid);tl.store(O+((graph*C+channel)*4+3)*K+freq,ni,valid)


def _launch(spectrum,gradient,kp,kn,config,adjoint):
    channels,mu,quad,compact=config
    graphs,frequencies=kp.shape[:2]
    fields=1+(3 if mu else 0)+((6 if compact else 9) if quad else 0)
    if adjoint:
        out=torch.empty((graphs,channels*fields,frequencies),device=kp.device,dtype=torch.complex64)
        output=torch.view_as_real(out)
        g=gradient.permute(0,2,3,1).contiguous()
        source=kp.new_empty(0)
        strides=(0,0,0)
    else:
        source=torch.view_as_real(spectrum.resolve_conj())
        strides=source.stride()[:3]
        out=kp.new_empty(graphs,channels,4,frequencies);output=out
        g=kp.new_empty(0)
    from .runtime import _vector_grid
    grid=_vector_grid(triton.cdiv(frequencies,128),kp.device)
    from .jit_launch import launch
    launch(_map,(grid,),source,g,kp,kn,output,graphs,frequencies,channels,fields,mu,quad,compact,adjoint,
        *strides,*kp.stride(),*kn.stride(),128,num_warps=4,enable_fp_fusion=False)
    return out.transpose(1,2) if adjoint else out.permute(0,3,1,2)


class _Amplitude(torch.autograd.Function):
    @staticmethod
    def forward(ctx,spectrum,kp,kn,config):
        ctx.save_for_backward(kp,kn);ctx.config=config;ctx.set_materialize_grads(False)
        return _launch(spectrum,None,kp,kn,config,False)
    @staticmethod
    def backward(ctx,gradient):
        if gradient is None:return None,None,None,None
        kp,kn=ctx.saved_tensors
        return _Adjoint.apply(gradient,kp,kn,ctx.config),None,None,None


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,gradient,kp,kn,config):
        ctx.save_for_backward(kp,kn);ctx.config=config;ctx.set_materialize_grads(False)
        return _launch(None,gradient,kp,kn,config,True)
    @staticmethod
    def backward(ctx,gradient):
        if gradient is None:return None,None,None,None
        kp,kn=ctx.saved_tensors
        return _Amplitude.apply(gradient,kp,kn,ctx.config),None,None,None


def amplitudes(spectrum,kp,kn,channels,*,has_dipole,has_quadrupole,compact_quadrupole):
    return _Amplitude.apply(spectrum,kp,kn,(channels,has_dipole,has_quadrupole,compact_quadrupole))
