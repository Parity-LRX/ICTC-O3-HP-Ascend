"""Generated mixed TP adjoints for parameter-only force-loss backward.

TP weights remain frozen; geometry cotangents and trainable TP weights retain
the general analytic route. Higher derivatives retain the native expression.
"""
import torch
from .generated_cartesian_message import _literal, _sum, _channel_read, _channel_write


def _update(p, pointer, value):
    return f'*({pointer})={value};' if p.get('edge_only',False) else f'atomicAdd({pointer},{value});'


def _stream_geometry_second(p,columns,row,halves,tile):
    path,ni,xb,bb,bs,ob,m=row;c=p['c'];base='channel_base+' if tile else ''
    lines=[]
    roles=bool(p.get('generated_second_roles',False))
    def geometry(i,mag):
        coeff=columns[bb+i*bs+mag]
        if coeff:
            expression=_sum(f'{_literal(a)}*h{h}' for h,a in coeff)
            lines.append(f'const float b{i}_{mag}='+('((HX||HG)?('+expression+'):0.0f)' if roles else expression)+';')
            lines.append(f'const float tb{i}_{mag}=HS?('+_sum(f'{_literal(a)}*t{h}' for h,a in coeff)+'):0.0f;')
    for half in halves:lines.append(f'float dg{half}=0.0f;')
    for i in range(ni):
        lines.append('{')
        for mag in range(m):geometry(i,mag)
        mags=[mag for mag in range(m) if columns[bb+i*bs+mag]]
        for half in halves:
            lines.append(f'const float z{half}='+_sum(f'b{i}_{mag}*v{half}_{mag}' for mag in mags)+';')
            lines.append(f'const float tz{half}=HS?('+_sum(f'tb{i}_{mag}*v{half}_{mag}' for mag in mags)+'):0.0f;')
            if roles:
                lines.append(f'float dxt{half}=0.0f; if constexpr(HS) dxt{half}+=g{half}*tz{half}; if constexpr(HG) dxt{half}+=tg{half}*z{half};')
                lines.append(f'if(needx) dx{half}_{i}+=w{half}*dxt{half};')
                lines.append(f'if(needg) {{ if constexpr(HX) dg{half}+=tx{half}_{i}*z{half}; if constexpr(HS) dg{half}+=x{half}_{i}*tz{half}; }}')
            else:
                lines.append(f'if(needx) dx{half}_{i}+=w{half}*(g{half}*tz{half}+tg{half}*z{half});')
                lines.append(f'if(needg) dg{half}+=tx{half}_{i}*z{half}+x{half}_{i}*tz{half};')
        lines.append('}')
    for half in halves:
        lines.append(_channel_write(f'if(needg) DG[e*{p["p"]*c}+{path*c+half*32}+{base}lane]=w{half}*dg{half};',half,c,tile))
    lines.append('if(needy) {')
    for mag in range(m):
        lines.append('{')
        for i in range(ni):geometry(i,mag)
        ids=[i for i in range(ni) if columns[bb+i*bs+mag]]
        for half in halves:
            if roles:
                tangent=f'delta{half}'
                lines.append(f'float {tangent}=0.0f;')
                for i in ids:
                    lines.append(f'{{float inner=0.0f; if constexpr(HX) inner+=tx{half}_{i}*b{i}_{mag}; if constexpr(HS) inner+=x{half}_{i}*tb{i}_{mag}; {tangent}+=g{half}*inner; if constexpr(HG) {tangent}+=tg{half}*x{half}_{i}*b{i}_{mag};}}')
            else:tangent=_sum(f'g{half}*(tx{half}_{i}*b{i}_{mag}+x{half}_{i}*tb{i}_{mag})+tg{half}*x{half}_{i}*b{i}_{mag}' for i in ids)
            lines.append(_channel_write(_update(p,f'DY+d*{p["odim"]}+{ob+mag*c+half*32}+{base}lane',f'w{half}*mask*({tangent})'),half,c,tile))
        lines.append('}')
    lines.append('}')
    return lines


def _source(p,projection):
    c=p["c"];hd=projection["hdim"]
    tile=int(p.get("generated_channel_tile",0));tiles=(c+tile-1)//tile if tile else 1
    halves=range(((tile or c)+31)//32)
    base="channel_base+" if tile else ""
    roles=bool(p.get('generated_second_roles',False))
    columns=[[] for _ in range(p["bdim"])]
    for h,rows in enumerate(projection["static_rows"]):
        for col,a in rows:columns[col].append((h,a))
    lines=["#include <torch/extension.h>","#include <ATen/cuda/CUDAContext.h>",
        "#include <c10/cuda/CUDAGuard.h>","#include <c10/cuda/CUDAException.h>","#include <cuda_runtime.h>",
        "template<bool HX,bool HS,bool HG,bool NeedY>" if roles else "template<bool HX,bool HS,bool HG>",
        "__global__ void second_kernel(const float* X,const float* SH,const float* W,const float* G,const int64_t* S,const int64_t* D,const float* M,const float* V,const float* TX,const float* TS,const float* TG,float* DX,float* DG,float* DY,int64_t E,bool masked,bool needx,bool needg,"+('bool requesty' if roles else 'bool needy')+") {",
        'const bool needy=NeedY;' if roles else '',
        "const int lane=threadIdx.x%32;",
        "const int64_t warp=(static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x)/32;",
        "const int64_t stride=static_cast<int64_t>(gridDim.x)*blockDim.x/32;",
        f"for(int64_t task=warp;task<E*{tiles};task+=stride) {{",
        f"const int64_t e=task/{tiles};",
        f"const int64_t channel_base=(task%{tiles})*{tile};",
        ("const int64_t s=e,d=e;" if p.get('edge_only',False) else "const int64_t s=S[e],d=D[e];")+
        "const float mask=masked?M[e]:1.0f;"]
    for h in range(hd):
        lines.append(f"const float h{h}="+(f"(HX||HG)?SH[e*{hd}+{h}]:0.0f;" if roles else f"SH[e*{hd}+{h}];"))
        lines.append(f"const float t{h}=HS?TS[e*{hd}+{h}]:0.0f;")
    groups={}
    for row in p["path_specs"]:groups.setdefault((row[2],row[1]),[]).append(row)
    for (xb,ni),paths in groups.items():
        lines.append("{")
        for half in halves:
            for i in range(ni):
                value=_channel_read(f"X[s*{p['xdim']}+{xb+i*c+half*32}+{base}lane]",half,c,tile)
                lines.append(f"const float x{half}_{i}={value};")
                value=_channel_read(f"HX?TX[s*{p['xdim']}+{xb+i*c+half*32}+{base}lane]:0.0f",half,c,tile)
                lines.append(f"const float tx{half}_{i}={value};")
                lines.append(f"float dx{half}_{i}=0.0f;")
        for path,ni,xb,bb,bs,ob,m in paths:
            lines.append("{")
            for half in halves:
                pointer=f"e*{p['p']*c}+{path*c+half*32}+{base}lane"
                for name,expression in ((f"w{half}",f"W[{path*c+half*32}+{base}lane]"),
                                        (f"g{half}",f"G[{pointer}]"),
                                        (f"tg{half}",f"HG?TG[{pointer}]:0.0f")):
                    lines.append(f"const float {name}={_channel_read(expression,half,c,tile)};")
                for mag in range(m):
                    value=_channel_read(f"V[d*{p['odim']}+{ob+mag*c+half*32}+{base}lane]*mask",half,c,tile)
                    lines.append(f"const float v{half}_{mag}={value};")
            if p.get('generated_geometry_streaming',False):
                lines.extend(_stream_geometry_second(p,columns,(path,ni,xb,bb,bs,ob,m),halves,tile))
                lines.append('}')
                continue
            for i in range(ni):
                for mag in range(m):
                    rows=columns[bb+i*bs+mag]
                    if not rows:continue
                    expression=_sum(f"{_literal(a)}*h{h}" for h,a in rows)
                    lines.append(f"const float b{i}_{mag}="+('((HX||HG)?('+expression+'):0.0f)' if roles else expression)+";")
                    lines.append(f"const float tb{i}_{mag}=HS?("+_sum(f"{_literal(a)}*t{h}" for h,a in rows)+"):0.0f;")
            for half in halves:
                for i in range(ni):
                    mags=[mag for mag in range(m) if columns[bb+i*bs+mag]]
                    lines.append(f"const float z{half}_{i}="+_sum([f"b{i}_{mag}*v{half}_{mag}" for mag in mags])+";")
                    lines.append(f"const float tz{half}_{i}=HS?("+_sum([f"tb{i}_{mag}*v{half}_{mag}" for mag in mags])+"):0.0f;")
                    if roles:
                        lines.append(f'float dxt{half}_{i}=0.0f; if constexpr(HS) dxt{half}_{i}+=g{half}*tz{half}_{i}; if constexpr(HG) dxt{half}_{i}+=tg{half}*z{half}_{i};')
                        lines.append(f'if(needx) dx{half}_{i}+=w{half}*dxt{half}_{i};')
                        lines.append(f'float dgt{half}_{i}=0.0f; if constexpr(HX) dgt{half}_{i}+=tx{half}_{i}*z{half}_{i}; if constexpr(HS) dgt{half}_{i}+=x{half}_{i}*tz{half}_{i};')
                    else:lines.append(f"if(needx) dx{half}_{i}+=w{half}*(g{half}*tz{half}_{i}+tg{half}*z{half}_{i});")
                gate=_sum([f'dgt{half}_{i}' if roles else f"tx{half}_{i}*z{half}_{i}+x{half}_{i}*tz{half}_{i}" for i in range(ni)])
                lines.append(_channel_write(f"if(needg) DG[e*{p['p']*c}+{path*c+half*32}+{base}lane]=w{half}*({gate});",half,c,tile))
                for mag in range(m):
                    ids=[i for i in range(ni) if columns[bb+i*bs+mag]]
                    if roles:
                        tangent=f'delta{half}_{mag}'
                        lines.append(f'float {tangent}=0.0f;')
                        for i in ids:
                            lines.append(f'{{float inner=0.0f; if constexpr(HX) inner+=tx{half}_{i}*b{i}_{mag}; if constexpr(HS) inner+=x{half}_{i}*tb{i}_{mag}; {tangent}+=g{half}*inner; if constexpr(HG) {tangent}+=tg{half}*x{half}_{i}*b{i}_{mag};}}')
                    else:tangent=_sum([f"g{half}*(tx{half}_{i}*b{i}_{mag}+x{half}_{i}*tb{i}_{mag})+tg{half}*x{half}_{i}*b{i}_{mag}" for i in ids])
                    lines.append(_channel_write('if(needy) '+_update(p,f"DY+d*{p['odim']}+{ob+mag*c+half*32}+{base}lane",f'w{half}*mask*({tangent})'),half,c,tile))
            lines.append("}")
        for half in halves:
            for i in range(ni):
                lines.append(_channel_write('if(needx) '+_update(p,f"DX+s*{p['xdim']}+{xb+i*c+half*32}+{base}lane",f'dx{half}_{i}'),half,c,tile))
        lines.append("}")
    lines.extend(["}","}"])
    arguments='x.data_ptr<float>(),sh.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),s.data_ptr<int64_t>(),d.data_ptr<int64_t>(),m.data_ptr<float>(),v.data_ptr<float>(),hx.data_ptr<float>(),hs.data_ptr<float>(),hg.data_ptr<float>(),dx.data_ptr<float>(),dg.data_ptr<float>(),dy.data_ptr<float>(),s.numel(),m.numel()!=0,needx,needg,needy'
    launch=(f'#define CALL(A,B,C) if(needy){{second_kernel<A,B,C,true><<<blocks,128,0,stream>>>({arguments});}}else{{second_kernel<A,B,C,false><<<blocks,128,0,stream>>>({arguments});}}'
            if roles else f'#define CALL(A,B,C) second_kernel<A,B,C><<<blocks,128,0,stream>>>({arguments})')
    lines.append(f"""
void generated_second(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,torch::Tensor s,torch::Tensor d,torch::Tensor m,torch::Tensor v,torch::Tensor hx,torch::Tensor hs,torch::Tensor hg,torch::Tensor dx,torch::Tensor dg,torch::Tensor dy,bool hasx,bool hass,bool hasg,bool needx,bool needg,bool needy) {{
    const c10::cuda::CUDAGuard guard(x.device());
    for(const auto& t:{{x,sh,w,g,m,v,hx,hs,hg,dx,dg,dy}}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kFloat&&t.is_contiguous(),"Invalid generated second operand");
    for(const auto& t:{{s,d}}) TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.scalar_type()==at::kLong&&t.is_contiguous(),"Invalid generated second topology");
    TORCH_CHECK(x.dim()==2&&x.size(1)=={p['xdim']}&&sh.dim()==2&&sh.size(1)=={hd}&&g.dim()==3&&g.size(1)=={p['p']}&&g.size(2)=={c}&&sh.size(0)==s.numel()&&g.size(0)==s.numel()&&d.numel()==s.numel(),"Invalid second shapes");
    TORCH_CHECK(w.dim()==2&&w.size(0)=={p['p']}&&w.size(1)=={c}&&v.dim()==2&&v.size(0)==x.size(0)&&v.size(1)=={p['odim']},"Invalid second weights/cotangent");
    TORCH_CHECK(m.numel()==0||m.numel()==s.numel(),"Invalid second mask");
    TORCH_CHECK(!hasx||hx.sizes()==x.sizes(),"Invalid feature tangent");
    TORCH_CHECK(!hass||hs.sizes()==sh.sizes(),"Invalid harmonic tangent");
    TORCH_CHECK(!hasg||hg.sizes()==g.sizes(),"Invalid gate tangent");
    TORCH_CHECK(!needx||dx.sizes()==x.sizes(),"Invalid second feature output");
    TORCH_CHECK(!needg||dg.sizes()==g.sizes(),"Invalid second gate output");
    TORCH_CHECK(!needy||dy.sizes()==v.sizes(),"Invalid second cotangent output");
    if(!s.numel())return;
    auto stream=at::cuda::getCurrentCUDAStream();const int64_t blocks=(s.numel()*{tiles}+3)/4;
    TORCH_CHECK(blocks<(1LL<<31),"Second grid exceeds supported size");
    {launch}
    if(hasx){{if(hass){{if(hasg){{CALL(true,true,true);}}else{{CALL(true,true,false);}}}}else{{if(hasg){{CALL(true,false,true);}}else{{CALL(true,false,false);}}}}}}
    else{{if(hass){{if(hasg){{CALL(false,true,true);}}else{{CALL(false,true,false);}}}}else{{if(hasg){{CALL(false,false,true);}}else{{return;}}}}}}
    #undef CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
""")
    return "\n".join(lines)+"\n"


def _extension(p,projection):
    cache=projection.setdefault("generated_second_extensions",{})
    tile=int(p.get("generated_channel_tile",0))
    if tile not in (0,32,64):raise ValueError("Unsupported generated channel tile")
    streaming=bool(p.get('generated_geometry_streaming',False))
    edge_only=bool(p.get('edge_only',False))
    key=(p["path_specs"],p["xdim"],p["odim"],p["c"],tile,streaming,edge_only)
    if p.get('compact_cartesian',False):key=(*key,'compact')
    if p.get('generated_second_roles',False):key=(*key,'second_roles')
    if key in cache:return cache[key]
    from torch.utils.cpp_extension import load_inline
    declarations="""#include <torch/extension.h>
void generated_second(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool,bool,bool,bool,bool,bool);
"""
    name=f"mace_ictc_cartesian_second_{p['xdim']}_{p['odim']}_{p['p']}"
    if tile:name+=f"_ct{tile}"
    if streaming:name+='_gs'
    if edge_only:name+='_edges'
    if p.get('compact_cartesian',False):name+='_compact'
    if p.get('generated_second_roles',False):name+='_sr'
    value=load_inline(name=name,cpp_sources=declarations,cuda_sources=_source(p,projection),functions=["generated_second"],
                      extra_cflags=["-O3"],extra_cuda_cflags=["-O3"],verbose=False)
    cache[key]=value;return value


def generated_second(x,sh,w,g,src,dst,mask,grad,p,projection,incoming,needed,need_grad):
    if p.get('generated_second_channel_tile') is not None:
        p={**p,'generated_channel_tile':p['generated_second_channel_tile']}
    from .fused_message import _channel_inputs,_channel_node,_normal_feature_gradient,_normal_message_output
    from .rematerialized_message import _tiles,_edge_slice
    hx,hs,_,hg=incoming
    if not any(value is not None for value in (hx,hs,hg)):return (None,)*5
    edge_only=bool(p.get('edge_only',False));packed=edge_only or bool(p.get('native_packed',False))
    xc,vc=(x,grad) if packed else _channel_inputs(x,grad,p)
    hxc=(hx if packed else _channel_node(hx,p)) if hx is not None else xc
    dx=torch.zeros_like(x) if needed[0] else None
    dg=torch.empty_like(g) if needed[3] else None
    dy=x.new_zeros(x.shape[0],p["odim"]) if need_grad else None
    for start,end in _tiles(src.numel(),p):
        sg=g[start:end];ss=sh[start:end]
        _extension(p,projection).generated_second(xc,ss.contiguous(),w,sg.contiguous(),src[start:end],dst[start:end],
            _edge_slice(mask,start,end),vc,hxc,ss if hs is None else hs[start:end].contiguous(),sg if hg is None else hg[start:end].contiguous(),
            xc if dx is None else dx,sg if dg is None else dg[start:end],vc if dy is None else dy,
            hx is not None,hs is not None,hg is not None,needed[0],needed[3],need_grad)
    # The kernels have consumed these packed copies. Release them before
    # allocating the normal-layout outputs on the same CUDA stream.
    if p.get('release_packed_inputs',False):del xc,vc,hxc
    if not packed:
        dx=_normal_feature_gradient(dx,p) if dx is not None else None
        dy=_normal_message_output(dy,p) if dy is not None else None
    return dx,None,None,dg,dy
