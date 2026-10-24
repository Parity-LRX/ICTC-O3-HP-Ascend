"""ICTC polynomials evaluated by native vector products and fixed sparse maps."""
import torch
from mace_ictc.ascend.runtime import require_npu_fp32
from mace_ictc.ascend.sparse_product import sparse_product
from mace_ictc.models.sparse_weighted_product import sparse_product_plan
from .elementwise import affine, multiply

_PLANS = {}


def _plan(n,lmax):
    from mace_ictc.models import ictd_irreps as basis
    constants=[]
    for degree in range(1,lmax+1):
        key=(str(n.device),str(n.dtype),degree)
        projection=basis._dir_proj_cache_by_dev_dtype.get(key)
        if projection is None:
            projection=basis._dir_proj_cpu_f64(degree).to(n)
            basis._dir_proj_cache_by_dev_dtype[key]=projection
        coefficients=basis._dir_coefs_cache_by_dev_dtype.get(key)
        if coefficients is None:
            coefficients=basis._dir_monomial_exps_coefs(degree)[1].to(n)
            basis._dir_coefs_cache_by_dev_dtype[key]=coefficients
        scale=basis._dir_component_scale_cache_by_dev_dtype.get(key)
        if scale is None:
            scale=torch.tensor(basis._direction_component_scale_cpu_f64(degree),device=n.device,dtype=n.dtype)
            basis._dir_component_scale_cache_by_dev_dtype[key]=scale
        if any(value.requires_grad or value.is_inference() for value in (projection,coefficients,scale)):
            raise ValueError('Native direction constants must be immutable mathematical buffers')
        constants.extend((projection,coefficients,scale))
    key=(str(n.device),lmax,tuple((id(value),value._version) for value in constants))
    if key in _PLANS:
        return _PLANS[key]
    entries=[(0,0,0,0,1.)]
    output=1
    for degree in range(1,lmax+1):
        projection,coefficients,scale=constants[(degree-1)*3:degree*3]
        projection=projection.detach().cpu().double()
        coefficients=coefficients.detach().cpu().double()
        scale=float(scale)
        for angular in range(2*degree+1):
            for row,powers in enumerate(basis._counts_list(degree)):
                if projection[row,angular]!=0:
                    entries.append((*powers,output,float(projection[row,angular]*coefficients[row])*scale))
            output+=1
    plan=sparse_product_plan(entries,(lmax+1,lmax+1,lmax+1,(lmax+1)**2),n.device,n.dtype)
    _PLANS[key]=plan
    return plan


def direction_harmonics(n,lmax):
    require_npu_fp32(n)
    plan=_plan(n,lmax)
    axes=n.T.contiguous()
    values=[affine(axes,0.,1.)]
    if lmax:
        values.append(axes)
    for degree in range(2,lmax+1):
        values.append(multiply(values[-1],axes))
    powers=torch.stack(values,-1).unsqueeze(2)
    output=sparse_product(powers[0],powers[1],powers[2],plan,bank=False).squeeze(1)
    return list(output.split([2*degree+1 for degree in range(lmax+1)],-1))
