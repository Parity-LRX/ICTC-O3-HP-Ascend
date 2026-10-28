# SPDX-License-Identifier: MIT
"""Single-field assignment using bounded particle/stencil vector tiles.

Coordinate Hessians retain the native sequential reduction in the dispatcher.
This kernel handles spread, gather, coordinate VJPs and source/mesh mixed VJPs.
"""
import triton
import triton.language as tl


@triton.jit
def _spline(x, u, ORDER:tl.constexpr):
    if ORDER==2:
        base=tl.floor(x).to(tl.int64);t=x-base.to(tl.float32)
        w=tl.where(u==0,1-t,t);d=tl.where(u==0,-1.,1.)
    elif ORDER==3:
        base=tl.floor(x-.5).to(tl.int64);t=x-base.to(tl.float32)
        w=tl.where(u==0,.5*(1.5-t)*(1.5-t),tl.where(u==1,.75-(t-1)*(t-1),.5*(t-.5)*(t-.5)))
        d=tl.where(u==0,t-1.5,tl.where(u==1,2*(1-t),t-.5))
    else:
        f=tl.floor(x).to(tl.int64);base=f-1;t=x-f.to(tl.float32);v=1-t;t2=t*t;t3=t2*t
        w=tl.where(u==0,v*v*v/6,tl.where(u==1,(3*t3-6*t2+4)/6,tl.where(u==2,(-3*t3+3*t2+3*t+1)/6,t3/6)))
        d=tl.where(u==0,-v*v/2,tl.where(u==1,1.5*t2-2*t,tl.where(u==2,-1.5*t2+t+.5,t2/2)))
    return base,w,d


@triton.jit
def _assignment(F,B,S,M,V,DS,DM,DF,N,C:tl.constexpr,MX:tl.constexpr,MY:tl.constexpr,MZ:tl.constexpr,
                ORDER:tl.constexpr,MODE:tl.constexpr,SLAB:tl.constexpr,FLAGS:tl.constexpr,
                ROWS:tl.constexpr,STENCIL:tl.constexpr):
    for block in range(tl.program_id(0),tl.cdiv(N*C,ROWS),tl.num_programs(0)):
        rows=block*ROWS+tl.arange(0,ROWS)
        atom=(rows//C).to(tl.int64);ch=rows%C
        stencil=tl.arange(0,STENCIL)
        mask=(rows[:,None]<N*C)&(stencil[None,:]<ORDER*ORDER*ORDER)
        ux=(stencil//(ORDER*ORDER))[None,:];uy=((stencil//ORDER)%ORDER)[None,:];uz=(stencil%ORDER)[None,:]
        fx=tl.load(F+3*atom,rows<N*C,other=0)[:,None]*MX
        fy=tl.load(F+3*atom+1,rows<N*C,other=0)[:,None]*MY
        fz=tl.load(F+3*atom+2,rows<N*C,other=0)[:,None]*MZ
        bx,wx,dx=_spline(fx,ux,ORDER);by,wy,dy=_spline(fy,uy,ORDER);bz,wz,dz=_spline(fz,uz,ORDER)
        ix=((bx+ux)%MX+MX)%MX;iy=((by+uy)%MY+MY)%MY
        if SLAB:iz=tl.minimum(tl.maximum(bz+uz,0),MZ-1)
        else:iz=((bz+uz)%MZ+MZ)%MZ
        graph=tl.load(B+atom,rows<N*C,other=0)
        offset=(((graph[:,None]*MX+ix)*MY+iy)*MZ+iz)*C+ch[:,None]
        weight=wx*wy*wz
        if MODE!=1:source=tl.load(S+rows,rows<N*C,other=0)[:,None]
        if MODE==0:
            tl.atomic_add(DM+offset,source*weight,mask,sem='relaxed')
        else:
            if MODE!=3 or FLAGS&3:
                field=tl.load(M+offset,mask,other=0)
            if MODE==1:
                value=tl.sum(tl.where(mask,field*weight,0.),1)
                tl.store(DS+rows,value,rows<N*C)
            else:
                dx=dx*MX;dy=dy*MY;dz=dz*MZ
                gx=dx*wy*wz;gy=wx*dy*wz;gz=wx*wy*dz
                if MODE==3:
                    vx=tl.load(V+3*atom,rows<N*C,other=0)[:,None]
                    vy=tl.load(V+3*atom+1,rows<N*C,other=0)[:,None]
                    vz=tl.load(V+3*atom+2,rows<N*C,other=0)[:,None]
                    direction=vx*gx+vy*gy+vz*gz
                    if FLAGS&2:tl.store(DS+rows,tl.sum(tl.where(mask,field*direction,0.),1),rows<N*C)
                    if FLAGS&4:tl.atomic_add(DM+offset,source*direction,mask,sem='relaxed')
                if MODE==2:
                    product=source*field
                    ax=tl.sum(tl.where(mask,product*gx,0.),1)
                    ay=tl.sum(tl.where(mask,product*gy,0.),1)
                    az=tl.sum(tl.where(mask,product*gz,0.),1)
                    tl.store(DF+3*atom,ax,rows<N*C)
                    tl.store(DF+3*atom+1,ay,rows<N*C)
                    tl.store(DF+3*atom+2,az,rows<N*C)


def run(frac,batch,source,mesh,direction,ds,dm,df,config,mode,flags):
    if source.shape[1]!=1 or mode==3 and flags&1:
        raise ValueError('Scalar vector assignment excludes coordinate Hessian reductions')
    if not frac.shape[0]:return
    shape,graphs,order,slab=config
    from .runtime import _vector_grid
    rows=8
    blocks=triton.cdiv(frac.shape[0]*source.shape[1],rows)
    grid=_vector_grid(blocks,frac.device)
    from .jit_launch import launch
    launch(_assignment,(grid,),frac,batch,source,mesh,direction,ds,dm,df,frac.shape[0],source.shape[1],
        *shape,order,mode,slab,flags,rows,triton.next_power_of_2(order**3),
        num_warps=4,enable_fp_fusion=False)
