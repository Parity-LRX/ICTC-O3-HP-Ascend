"""Independent shared CG-prefix evaluation of symmetric MACE polynomials.

Original learned path weights stay live and retain their checkpoint schema.
Linear dependencies are established from fixed symmetric coefficients, then
checked against the actual bridge-U buffers. Unsupported or changed constants
that do not reconstruct to dtype roundoff use the original contraction.
"""
import itertools

import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake

from ._mace_cg import _wigner_nj
from .sparse_weighted_product import sparse_product_plan


def _compress(value, order, size):
    tuples=list(itertools.combinations_with_replacement(range(size),order))
    lookup={indices:t for t,indices in enumerate(tuples)}
    mapping=torch.tensor([lookup[tuple(sorted(indices))]
                          for indices in itertools.product(range(size),repeat=order)])
    result=value.new_zeros(value.shape[0],len(tuples),value.shape[-1])
    result.index_add_(1,mapping,value.reshape(value.shape[0],size**order,value.shape[-1]))
    return result.reshape(-1,value.shape[-1])


def _independent_columns(matrix):
    singular=torch.linalg.svdvals(matrix)
    if not singular.numel() or singular.max()==0:return []
    rank=int((singular>singular.max()*1e-12).sum())
    residual=matrix.clone();selected=[]
    for _ in range(rank):
        pivot=int(residual.square().sum(0).argmax());selected.append(pivot)
        direction=residual[:,pivot]/residual[:,pivot].norm()
        for _ in range(2):
            residual=residual-direction[:,None]*(direction@residual)[None,:]
    return selected


def _fold(value, q, qout, order):
    if order==1:return torch.einsum("ro,ai,oip->rap",qout,q,value)
    if order==2:return torch.einsum("ro,ai,bj,oijp->rabp",qout,q,q,value)
    if order==3:return torch.einsum("ro,ai,bj,ck,oijkp->rabcp",qout,q,q,q,value)
    # Avoid a fixed einsum alphabet or a hard-coded cubic polynomial.
    result=torch.tensordot(qout,value,dims=([1],[0]))
    for axis in range(1,order+1):
        result=torch.tensordot(q,result,dims=([1],[axis])).movedim(0,axis)
    return result


def _plan(owner, x):
    modules=list(owner.symmetric_contractions.contractions)
    correlation=modules[0].correlation
    if owner.lmax<0 or correlation not in (1,2,3):return None
    if any(module.correlation!=correlation for module in modules):return None
    qs=[getattr(owner.basis_bridge,f"q_{l}") for l in range(max(owner.lmax,owner.target_lmax)+1)]
    constants=[module.U_tensors(order) for module in modules for order in range(1,correlation+1)]
    if any(value.requires_grad for value in [*qs,*constants]):return None
    from .reference_u_residency import constant_key
    key=constant_key(owner,x.device,x.dtype)
    cached=getattr(owner,"_factorized_u_cache",None)
    if cached is not None and cached[0]==key:return cached[1]
    with torch.no_grad():
        qblocks=[value.detach().cpu().double() for value in qs]
        q=torch.block_diag(*qblocks[:owner.lmax+1]);size=q.shape[0]
        targets=[o3.Irrep(module.output_lmax,(-1)**module.output_lmax) for module in modules]
        paths_by_order={order:_wigner_nj([modules[0].coupling_irreps]*order,dtype=torch.float64,filter_ir_out=targets)
                        for order in range(1,correlation+1)}
        projections=[];selected_paths=[];reconstruction=[]
        for module in modules:
            ir=o3.Irrep(module.output_lmax,(-1)**module.output_lmax)
            for order in range(1,correlation+1):
                paths=[item for item in paths_by_order[order] if item[0]==ir]
                u=module.U_tensors(order).detach().cpu().double()
                if ir.l==0:u=u.unsqueeze(0)
                # Reduced-CG U tensors span a subset of the full polynomial
                # space. Their learned path count need not equal the complete
                # CG path count: project the actual buffers and retain the
                # reconstruction gate below, with every learned weight live.
                if not paths:return None
                native=torch.stack([item[2] for item in paths],-1)
                selected=_independent_columns(_compress(native,order,size))
                if not selected:return None
                basis=_compress(_fold(native[...,selected],q,qblocks[ir.l],order),order,size)
                actual=_compress(u,order,size)
                projection=torch.linalg.lstsq(basis,actual).solution
                residual=(basis@projection-actual).abs().max()
                relative=float(residual/actual.abs().max().clamp_min(1e-30))
                tolerance=4e-6 if x.dtype==torch.float32 else 2e-11
                if relative>tolerance:
                    owner._factorized_u_cache=(key,None)
                    return None
                reconstruction.append(relative)
                projections.append((module,order,projection.to(x)))
                selected_paths.extend((ir,order,paths[p][1]) for p in selected)
        pair_specs={}
        for ir,order,path in selected_paths:
            if order<2:continue
            pair=path if order==2 else path.args[0]
            left,right=pair.args;l1,l2,mid=(item.l for item in pair.op)
            if l1>l2:left,right=right,left;l1,l2=l2,l1
            key_pair=(l1,l2,mid)
            if key_pair not in pair_specs:pair_specs[key_pair]=(left,right,len(pair_specs))
        pair_entries=[];pair_offsets={};width=0
        for (l1,l2,mid),(left,right,_) in pair_specs.items():
            pair_offsets[(l1,l2,mid)]=width
            cg=o3.wigner_3j(mid,l1,l2,dtype=torch.float64)*(2*mid+1)**.5
            for out,a,b in torch.nonzero(cg,as_tuple=False).tolist():
                pair_entries.append((left.start+a,right.start+b,0,width+out,float(cg[out,a,b])))
            width+=2*mid+1
        output_offsets={};m=0
        for module in modules:
            output_offsets[module.output_lmax]=m;m+=2*module.output_lmax+1
        entries=[]
        for p,(ir,order,path) in enumerate(selected_paths):
            offset=output_offsets[ir.l]
            if order==1:
                entries.extend((width+path.start+out,size,p,offset+out,1.) for out in range(ir.dim))
                continue
            pair=path if order==2 else path.args[0]
            l1,l2,mid=(item.l for item in pair.op)
            sign=(-1)**(l1+l2-mid) if l1>l2 else 1
            base=pair_offsets[(min(l1,l2),max(l1,l2),mid)]
            if order==2:
                entries.extend((base+out,size,p,offset+out,float(sign)) for out in range(ir.dim))
            else:
                last=path.args[1];last_l=path.op[1].l
                cg=o3.wigner_3j(ir.l,mid,last_l,dtype=torch.float64)*(2*ir.l+1)**.5
                for out,a,b in torch.nonzero(cg,as_tuple=False).tolist():
                    entries.append((base+a,last.start+b,p,offset+out,float(cg[out,a,b])*sign))
        merged={}
        for a,b,p,out,value in pair_entries:
            pair_key=(min(a,b),max(a,b),p,out)
            merged[pair_key]=merged.get(pair_key,0.)+value
        pair_entries=[(*key,value) for key,value in merged.items()]
        derivative_entries=[entry for a,b,p,out,value in pair_entries
                            for entry in ((a,b,p,out,value),(b,a,p,out,value))]
        plan={"q":q.to(x),"qout":[value.to(x) for value in qblocks[:owner.target_lmax+1]],
              "pair":sparse_product_plan(pair_entries,(size,size,1,width),x.device,x.dtype) if width else None,
              "pair_adjoint":sparse_product_plan(derivative_entries,(size,size,1,width),x.device,x.dtype) if width else None,
              "final":sparse_product_plan(entries,(width+size,size+1,len(selected_paths),m),x.device,x.dtype),
              "ones":x.new_ones(x.shape[1],1),"projections":projections,
              "width":width,"sizes":[2*module.output_lmax+1 for module in modules],
              "pair_blocks":tuple((offset,2*mid+1) for (_,_,mid),offset in pair_offsets.items()),
              "reconstruction_relative":reconstruction}
        plan["final"]["weight_node_reduction"]=True
        plan["input_size"]=size
        owner._factorized_u_cache=(key,plan)
        return plan
