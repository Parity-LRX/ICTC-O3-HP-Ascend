"""Optional strict-FP32 Vector kernels for narrow angular batched products."""
import os
import torch
from . import runtime

_CACHE={}


def _align(n):return (n+7)//8*8


def try_apply(a,b,ta,tb,*,capacity=False):
    if (a.device.type!='npu' or not runtime.is_enabled()
            or runtime.mode()!='performance' and not (capacity and runtime.mode()=='capacity')):return None
    policy=os.environ.get('ICTC_ASCEND_VECTOR_BMM','auto')
    if policy not in ('0','1','auto','forward','channel','both','feature','adjoints'):
        raise ValueError('Invalid Vector BMM policy')
    if policy=='0' or a.ndim!=3 or b.ndim!=3:return None
    if a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.device!=b.device:return None
    feature=ta and not tb
    if ta:
        if not feature or policy not in ('feature','adjoints','auto') or a.shape[2]>8 or a.shape[1]>16:
            return None
    mode=2 if feature else int(tb)
    if feature:n,inner,q=a.shape
    else:n,q,inner=a.shape
    if mode==1:
        if policy in ('forward','feature') or b.shape[2]!=inner:return None
        i,c=b.shape[1],inner
    else:
        if policy=='channel' or (policy in ('feature','adjoints','auto') and not feature) or b.shape[1]!=inner:return None
        i,c=inner,b.shape[2]
    if b.shape[0]!=n or q<1 or q>64 or i<1 or i>(16 if feature else 8) or c<1:return None
    for v in (a,b):
        if v.stride(2)!=1 or v.stride(1)!=v.shape[2] or v.stride(0)<v.shape[1]*v.shape[2]:return None
    if policy=='auto' and (c<32 or n*q*c<1024**2):return None
    ops=runtime.load_library();ub,_=runtime.hardware(a.device)
    cap=int(os.environ.get('ICTC_ASCEND_VECTOR_BMM_CHANNELS','64'))
    reserve=int(os.environ.get('ICTC_ASCEND_VECTOR_BMM_RESERVE','4096'))
    if cap<8 or cap>64 or cap%8 or reserve<4096 or reserve>=ub:raise ValueError('Invalid Vector BMM UB policy')
    key=str(a.device),q,i,c,mode,cap,reserve,ub
    if key not in _CACHE:
        blocks=(c+cap-1)//cap;bc=_align((c+blocks-1)//blocks)
        rows=min(255//q,8192//(q*bc),8192//(q*i))
        while rows:
            vec=rows*q*bc
            inp=(vec if mode==1 else rows*_align(q*i))+rows*i*bc
            out=_align(rows*q*i) if mode==1 else vec
            used=4*(2*inp+2*out+4*vec+i*_align(rows*q)+_align(rows*q*i))
            if used+reserve<=ub:break
            rows-=1
        if not rows:return None
        packet=[]
        packet.extend((r*_align(q*i)+angular*(1 if feature else i))*4 for r in range(rows) for angular in range(q) for ch in range(bc))
        packet.extend((r*i*bc+ch)*4 for r in range(rows) for angular in range(q) for ch in range(bc))
        packet.extend((angular*_align(rows*q)+rowq)*4 for rowq in range(rows*q) for angular in range(i))
        packet.extend([0]*(_align(rows*q*i)-rows*q*i))
        indices=torch.tensor(packet,dtype=torch.int32,device=a.device)
        if len(_CACHE)>=128:_CACHE.clear()
        _CACHE[key]=rows,bc,indices
    rows,bc,indices=_CACHE[key]
    return ops.batched_vector(a,b,indices,[n,q,i,c,mode,rows,bc])
