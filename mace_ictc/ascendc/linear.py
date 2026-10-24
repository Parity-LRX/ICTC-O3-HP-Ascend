"""Native FP32 Cube matmuls, bank selection and recursively live adjoints."""
import torch
import torch.nn.functional as F
from mace_ictc.models.adjoint_requirements import required_gradients
from mace_ictc.models.sparse_weighted_product import sparse_product_plan
from mace_ictc.ascend.sparse_product import _Product
from .runtime import load_library,is_enabled,native_matrix_enabled

_SELECT = {}


class _Matrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,ta,tb,bias):
        # Frozen-weight inference needs the weight, but not the large input
        # activation, for its input adjoint. Training retains both live inputs.
        ctx.save_for_backward(a if b.requires_grad else a.new_empty(0),
                              b if a.requires_grad else b.new_empty(0),
                              *(() if bias is None else (bias,)))
        ctx.spec=ta,tb,bias is not None
        if native_matrix_enabled():
            return load_library().mm(a.contiguous(),b.contiguous(),ta,tb,
                                     None if bias is None else bias.contiguous())
        result=torch.mm(a.T if ta else a,b.T if tb else b)
        return result if bias is None else result+bias

    @staticmethod
    def backward(ctx,g):
        a,b,*saved=ctx.saved_tensors
        ta,tb,biased=ctx.spec
        declared=(ctx.needs_input_grad[0],ctx.needs_input_grad[1],
                  ctx.needs_input_grad[4] if biased else False)
        values=(a,b,*saved)
        needed=required_gradients(values,declared[:len(values)])
        da=db=dc=None
        if needed[0]:
            da=(matmul(b,g,tb,True) if ta else matmul(g,b,False,not tb))
        if needed[1]:
            db=(matmul(g,a,True,ta) if tb else matmul(a,g,not ta,False))
        if biased and needed[2]:
            dc=g.sum(0)
        return da,db,None,None,dc


def matmul(a,b,transpose_a=False,transpose_b=False,bias=None):
    return _Matrix.apply(a,b,bool(transpose_a),bool(transpose_b),bias)


def linear(x,weight,bias=None):
    shape=x.shape
    flat=x.reshape(-1,shape[-1])
    return matmul(flat,weight,False,True,bias).reshape(*shape[:-1],weight.shape[0])


def channel_linear(features,weight,bias=None):
    angular=features.movedim(-2,-1).contiguous()
    result=linear(angular,weight,bias)
    return result.movedim(-1,-2)


def _select(values,species,angular):
    count=len(values)
    rows,channels=values[0].shape
    if count==1:
        return values[0]
    source=torch.stack(values,0).reshape(count*rows,channels,1)
    cache=getattr(species,'_ictc_ascendc_selection',None)
    key=(species._version,rows,angular,count)
    if cache is None or cache[0]!=key:
        destination=torch.arange(rows,device=species.device,dtype=torch.int64)
        expanded=species.to(dtype=torch.int64).repeat_interleave(angular)
        source_index=expanded*rows+destination
        cache=(key,source_index,destination)
        species._ictc_ascendc_selection=cache
    _,src,dst=cache
    plan_key=(str(source.device),channels)
    plan=_SELECT.get(plan_key)
    if plan is None:
        plan=sparse_product_plan(((0,0,0,0,1.),),(1,1,1,1),source.device,source.dtype)
        plan['_select_weight']=source.new_ones(1,channels,1)
        plan['_ascendc_metadata']={}
        plan['_ascendc_tiling']={}
        _SELECT[plan_key]=plan
    graph=dict(src=src,dst=dst,mask=None,destination_nodes=rows)
    proxy={**plan,'_ascendc_graph':graph}
    right=source.new_ones(rows,1,1)
    return _Product.apply(source,right,plan['_select_weight'],None,proxy,True).squeeze(-1)


def species_channel_linear(features,weight,species,bias=None):
    nodes,_,angular=features.shape
    flat=features.transpose(1,2).contiguous().reshape(nodes*angular,features.shape[1])
    values=[matmul(flat,weight[s],False,True,None if bias is None else bias[s])
            for s in range(weight.shape[0])]
    return _select(values,species,angular).reshape(nodes,angular,weight.shape[1]).transpose(1,2)


class NativeLinear(torch.nn.Linear):
    """Parameter-compatible Linear with explicit native dispatch."""
    def forward(self,x):
        if is_enabled() and x.device.type=='npu' and getattr(self,'_ascend_hp',True):
            return linear(x,self.weight,self.bias)
        return F.linear(x,self.weight,self.bias)


class NativeSiLU(torch.nn.SiLU):
    def forward(self,x):
        if is_enabled() and x.device.type=='npu' and getattr(self,'_ascend_hp',True):
            from .elementwise import normalized_silu
            return normalized_silu(x)
        return F.silu(x,inplace=self.inplace)


def prepare_model(model):
    """Retain exact Parameter objects, hooks, aliases and state_dict names."""
    converted={}
    count=0
    def visit(parent):
        nonlocal count
        for name,child in list(parent._modules.items()):
            if child is None:
                continue
            kind=(NativeLinear if type(child) is torch.nn.Linear else
                  NativeSiLU if type(child) is torch.nn.SiLU and not child.inplace else None)
            if kind is not None:
                replacement=converted.get(id(child))
                if replacement is None:
                    replacement=kind.__new__(kind)
                    replacement.__dict__=child.__dict__.copy()
                    converted[id(child)]=replacement
                    count+=1
                parent._modules[name]=replacement
            else:
                visit(child)
    visit(model)
    return count
