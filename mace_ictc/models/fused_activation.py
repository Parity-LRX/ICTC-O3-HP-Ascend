"""Normalized SiLU with fused first and mixed-second derivative kernels."""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake
from .adjoint_requirements import required_gradients
try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None

if triton is not None:
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _silu(X,Y,N,S:tl.constexpr,BLOCK:tl.constexpr):
        idx=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        x=tl.load(X+idx,idx<N,0.)
        y=tl.div_rn(x,1.+libdevice.exp(-x))*S
        tl.store(Y+idx,y,idx<N)

    @triton.jit
    def _silu_grad(X,G,DX,N,S:tl.constexpr,BLOCK:tl.constexpr):
        idx=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        x=tl.load(X+idx,idx<N,0.);g=tl.load(G+idx,idx<N,0.)
        sigmoid=tl.div_rn(1.,1.+libdevice.exp(-x))
        value=((g*S)*sigmoid)*(1.+x*(1.-sigmoid))
        tl.store(DX+idx,value,idx<N)

    @triton.jit
    def _silu_second(X,G,H,DX,DG,N,S:tl.constexpr,NEEDX:tl.constexpr,NEEDG:tl.constexpr,BLOCK:tl.constexpr):
        idx=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
        valid=idx<N;x=tl.load(X+idx,valid,0.);h=tl.load(H+idx,valid,0.)
        sigmoid=tl.div_rn(1.,1.+libdevice.exp(-x))
        complement=1.-sigmoid
        coefficient=1.+x*complement
        left=h*coefficient
        if NEEDX:
            g=tl.load(G+idx,valid,0.)
            scaled_g=g*S
            right=h*(scaled_g*sigmoid)
            sigmoid_grad=left*scaled_g-right*x
            # Preserve the literal mixed-adjoint product/add order rather
            # than collapsing to f''(x); all intermediates stay FP32.
            dx=right*complement+(sigmoid_grad*complement)*sigmoid
            tl.store(DX+idx,dx,valid)
        if NEEDG:
            tl.store(DG+idx,(left*sigmoid)*S,valid)


class _SiLUGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,grad,scale):
        ctx.scale=scale;grad=grad.contiguous();ctx.save_for_backward(x,grad)
        dx=torch.empty_like(x)
        if dx.numel():_silu_grad[(triton.cdiv(dx.numel(),256),)](x,grad,dx,dx.numel(),scale,256,num_warps=4,enable_fp_fusion=False)
        return dx

    @staticmethod
    def backward(ctx,h):
        x,grad=ctx.saved_tensors;needed=required_gradients((x,grad),ctx.needs_input_grad[:2])
        if torch.is_grad_enabled():
            sigmoid=torch.sigmoid(x)
            complement=1.-sigmoid
            left=h*(1.+x*complement)
            scaled_g=grad*ctx.scale
            right=h*(scaled_g*sigmoid)
            sigmoid_grad=left*scaled_g-right*x
            return (right*complement+(sigmoid_grad*complement)*sigmoid if needed[0] else None,
                    (left*sigmoid)*ctx.scale if needed[1] else None,None)
        h=h.contiguous();dx=torch.empty_like(x) if needed[0] else None;dg=torch.empty_like(x) if needed[1] else None
        if x.numel():
            _silu_second[(triton.cdiv(x.numel(),256),)](x,grad,h,dx if needed[0] else x,dg if needed[1] else x,
                x.numel(),ctx.scale,*needed,256,num_warps=4,enable_fp_fusion=False)
        return dx,dg,None

class _SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale):
        ctx.scale=scale;ctx.save_for_backward(x)
        y=torch.empty_like(x)
        if y.numel():_silu[(triton.cdiv(y.numel(),256),)](x,y,y.numel(),scale,256,num_warps=4,enable_fp_fusion=False)
        return y

    @staticmethod
    def backward(ctx,grad):
        x,=ctx.saved_tensors
        if torch.is_grad_enabled():
            return _SiLUGrad.apply(x,grad,ctx.scale),None
        dx=torch.empty_like(x);grad=grad.contiguous()
        if dx.numel():_silu_grad[(triton.cdiv(dx.numel(),256),)](x,grad,dx,dx.numel(),ctx.scale,256,num_warps=4,enable_fp_fusion=False)
        return dx,None

def normalized_silu(x,scale):
    if (triton is None or not x.is_cuda or x.dtype!=torch.float32 or not x.is_contiguous()
            or torch.compiler.is_compiling() or torch.is_autocast_enabled(x.device.type)
            or get_proxy_mode() is not None or is_fake(x) or torch._C._are_functorch_transforms_active()):
        return torch.nn.functional.silu(x)*scale
    return _SiLU.apply(x,scale)
