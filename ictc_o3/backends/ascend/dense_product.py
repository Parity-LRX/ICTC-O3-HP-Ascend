"""GEMM execution of sparse trilinear constants with bounded scratch storage."""
import torch

_CONSTANT_ELEMENTS = 16 * 1024 * 1024
_BANK_BYTES = 256 * 1024 * 1024
_SCRATCH_BYTES = 128 * 1024 * 1024


def product(left, right, weight, plan, *, index=None):
    """Keep analytic adjoints while replacing irregular gathers with GEMMs."""
    a,b,p,m=plan["sizes"]
    if a*b*p*m > _CONSTANT_ELEMENTS or weight.shape[0]*weight.shape[1]*a*b*m*4 > _BANK_BYTES:
        from .sparse_product import sparse_product
        return sparse_product(left,right,weight,plan,index=index)
    proxy=plan.get("_ascend_gemm_proxy")
    if proxy is None:
        proxy={**plan,"_ascend_gemm":True}
        plan["_ascend_gemm_proxy"]=proxy
    from .sparse_product import sparse_product
    return sparse_product(left,right,weight,proxy,index=index)


def _table(plan, sample, mode):
    tables=plan.setdefault("_ascend_gemm_tables",{})
    if mode in tables:
        return tables[mode]
    a,b,p,m=plan["sizes"]
    source=plan.get("_ascend_dense_cpu")
    if source is None:
        source=torch.zeros(a,b,p,m,device="cpu",dtype=torch.float64)
        for i,j,k,q,value in plan["entries"]:
            source[i,j,k,q]+=value
        plan["_ascend_dense_cpu"]=source
    if mode==0:
        result=source.permute(2,0,1,3).reshape(p,a*b*m)
    elif mode==1:
        result=source.permute(2,1,3,0).reshape(p,b*m*a)
    elif mode==2:
        result=source.permute(2,0,3,1).reshape(p,a*m*b)
    else:
        result=source.permute(0,1,3,2).reshape(a*b*m,p)
    tables[mode]=result.to(device=sample.device,dtype=sample.dtype).contiguous()
    return tables[mode]


def _bilinear(left,right,weight,index,table,output_width):
    n=left.shape[0]
    channels=weight.shape[1]
    a,b=left.shape[-1],right.shape[-1]
    species=weight.shape[0]
    chunk=max(1,_SCRATCH_BYTES//max(1,channels*a*b*left.element_size()))
    matrices=(weight@table).reshape(species,channels,a*b,output_width)
    output=left.new_empty(n,channels,output_width)
    for start in range(0,n,chunk):
        end=min(start+chunk,n)
        outer=(left[start:end,:,:,None]*right[start:end,:,None,:]).reshape(end-start,channels,a*b)
        if species==1 and index is None and weight.shape[-1]==1:
            # Shared scalar weights commute with the fixed coupling matrix.
            value=(outer.reshape(-1,a*b)@table.reshape(a*b,output_width)).reshape(end-start,channels,output_width)
            value=value*weight[0,:,0][None,:,None]
        else:
            packed=outer.transpose(0,1).contiguous()
            value=None
            for s in range(species):
                part=torch.bmm(packed,matrices[s]).transpose(0,1)
                if index is not None:
                    part=torch.where(index[start:end,None,None]==s,part,0.)
                value=part if value is None else value+part
        output[start:end]=value
    return output


def raw(left,right,weight,index,plan,bank,mode=0,grad=None):
    if not bank:
        raise ValueError("The contraction GEMM route requires a shared or species weight bank")
    a,b,p,m=plan["sizes"]
    table=_table(plan,left,mode)
    if mode==0:
        return _bilinear(left,right,weight,index,table,m)
    if mode==1:
        return _bilinear(right,grad,weight,index,table,a)
    if mode==2:
        result=_bilinear(left,grad,weight,index,table,b)
        return result.sum(1,keepdim=True) if right.shape[1]==1 and left.shape[1]!=1 else result
    n,channels=left.shape[:2]
    species=weight.shape[0]
    chunk=max(1,_SCRATCH_BYTES//max(1,channels*a*b*left.element_size()))
    accumulated=left.new_zeros(species,channels,a*b,m)
    for start in range(0,n,chunk):
        end=min(start+chunk,n)
        outer=(left[start:end,:,:,None]*right[start:end,:,None,:]).reshape(end-start,channels,a*b)
        packed=outer.transpose(0,1).contiguous().transpose(1,2)
        for s in range(species):
            cotangent=grad[start:end]
            if index is not None:
                cotangent=torch.where(index[start:end,None,None]==s,cotangent,0.)
            accumulated[s].add_(torch.bmm(packed,cotangent.transpose(0,1).contiguous()))
    return (accumulated.reshape(species*channels,a*b*m)@table).reshape_as(weight)
