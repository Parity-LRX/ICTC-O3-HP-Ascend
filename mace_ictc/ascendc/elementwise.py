"""Native FP32 vector operations and analytic, recursively live adjoints."""
import torch
from .runtime import load_library
from mace_ictc.models.adjoint_requirements import required_gradients


class _Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y):
        ctx.save_for_backward(x,y)
        return load_library().binary(x,y,1)

    @staticmethod
    def backward(ctx,g):
        x,y=ctx.saved_tensors
        needed=required_gradients((x,y),ctx.needs_input_grad)
        return (_Multiply.apply(g.contiguous(),y) if needed[0] else None,
                _Multiply.apply(g.contiguous(),x) if needed[1] else None)


class _Affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,alpha,beta):
        ctx.alpha=alpha
        return load_library().binary(x,x,4,0,alpha,beta)

    @staticmethod
    def backward(ctx,g):
        return _Affine.apply(g.contiguous(),ctx.alpha,0.),None,None


class _Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return load_library().binary(x,x,2)

    @staticmethod
    def backward(ctx,g):
        x,=ctx.saved_tensors
        return _UnaryGradient.apply(x,g.contiguous(),2,1.)


class _UnaryGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,g,kind,scale):
        ctx.save_for_backward(x,g)
        ctx.spec=kind,scale
        return load_library().binary(x,g,6 if kind==3 else 8,0,scale)

    @staticmethod
    def backward(ctx,h):
        x,g=ctx.saved_tensors
        kind,scale=ctx.spec
        needed=required_gradients((x,g),ctx.needs_input_grad[:2])
        h=h.contiguous()
        dx=dg=None
        if needed[0]:
            if torch.is_grad_enabled():
                s=_Sigmoid.apply(x)
                common=multiply(s,affine(s,-1.,1.))
                if kind==3:
                    second=multiply(common,affine(multiply(x,affine(s,-2.,1.)),1.,2.))
                else:
                    second=multiply(common,affine(s,-2.,1.))
                dx=multiply(multiply(h,g),affine(second,scale,0.))
            else:
                second=load_library().binary(x,g,7 if kind==3 else 9,0,scale)
                dx=load_library().binary(h,second,1)
        if needed[1]:
            dg=_UnaryGradient.apply(x,h,kind,scale)
        return dx,dg,None,None


class _SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale):
        ctx.save_for_backward(x)
        ctx.scale=scale
        return load_library().binary(x,x,3,0,scale)

    @staticmethod
    def backward(ctx,g):
        x,=ctx.saved_tensors
        return _UnaryGradient.apply(x,g.contiguous(),3,ctx.scale),None


def multiply(x,y):
    return _Multiply.apply(x.contiguous(),y.contiguous())


def affine(x,alpha=1.,beta=0.):
    return _Affine.apply(x.contiguous(),float(alpha),float(beta))


def normalized_silu(x,scale=1.):
    return _SiLU.apply(x.contiguous(),float(scale))
