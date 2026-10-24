"""Generic fixed multilinear maps with a complete hardware-derived UB budget."""
import os
import torch
from .runtime import load_library


def _align(value):
    return (value+7)//8*8


def _metadata(plan, mode, device):
    cache = plan.setdefault('_ascendc_metadata', {})
    key = (mode, str(device))
    if key in cache:
        return cache[key]
    a,b,p,m = plan['sizes']
    width = (m,a,b,p)[mode]
    leg = (3,0,1,2)[mode]
    rows = [[] for _ in range(width)]
    for entry in plan['entries']:
        if entry[4] != 0.:
            rows[entry[leg]].append(entry)
    offsets,entries = [0],[]
    for row in rows:
        entries.extend(row)
        offsets.append(len(entries))
    stride = _align(len(entries))
    head = _align(len(offsets))
    packed = torch.zeros(head+5*stride,dtype=torch.int32,device='cpu')
    packed[:len(offsets)] = torch.tensor(offsets,dtype=torch.int32)
    if entries:
        for column in range(4):
            packed[head+column*stride:head+column*stride+len(entries)] = torch.tensor(
                [entry[column] for entry in entries],dtype=torch.int32)
        packed[head+4*stride:head+4*stride+len(entries)] = torch.tensor(
            [entry[4] for entry in entries],dtype=torch.float32).view(torch.int32)
    result = packed.to(device),width,len(entries),head,stride
    cache[key] = result
    return result


def _layout(c,rc,a,b,p,m,mode,rows,block_c,masked=False,graph=False):
    # mode 1 reads right, weight and grad, but not left.
    use = (mode != 1, mode in (0,1,3), mode != 3, mode != 0)
    pitches = [_align(block_c*a), _align(b if rc == 1 else block_c*b),
               _align(block_c*p), _align(block_c*m)]
    pitches = [pitch if active else 0 for pitch,active in zip(pitches,use)]
    nc = _align(rows*block_c)
    output_rows = rows if mode == 2 and rc == 1 else rows*block_c
    index_words = 4*nc+8*output_rows+_align(block_c)+(nc if masked else 0)
    # Two input and output slots, one raw 8-column result, three work vectors,
    # byte-offset indices and explicit alignment on every UB sub-buffer.
    floats = 2*rows*(sum(pitches)+(8 if masked else 0))+2*8*_align(output_rows)+8*nc+3*nc
    if graph:
        floats += 4*_align(2*rows)  # two buffered slots, each with src/dst int64 arrays
    return pitches,nc,output_rows,index_words,4*(floats+index_words)


def _tiling(plan, sample, c,rc,a,b,p,m,mode,width,terms,head,masked=False,graph=False):
    ops = load_library()
    ub,cores = ops.hardware()
    term_cap = min(_align(terms),1024)
    metadata_bytes = 4*(head+5*term_cap)
    cache = plan.setdefault('_ascendc_tiling', {})
    requested_c = int(os.environ.get('ICTC_ASCENDC_TILE_CHANNELS','0'))
    requested_rows = int(os.environ.get('ICTC_ASCENDC_TILE_ROWS','0'))
    default_reserve=max(4096,((ub+6*4096-1)//(6*4096))*4096)
    reserve = int(os.environ.get('ICTC_ASCENDC_UB_RESERVE',str(default_reserve)))
    if reserve < 4096 or reserve >= ub:
        raise ValueError('Native UB reserve must be at least 4096 bytes and smaller than the UB capacity')
    if requested_c < 0 or requested_rows < 0:
        raise ValueError('Native tile overrides must be positive, or zero for automatic sizing')
    key = (str(sample.device),c,rc,mode,width,terms,ub,requested_c,requested_rows,masked,graph,reserve)
    if key in cache:
        return cache[key]
    bc = min(c,requested_c or 64)
    while bc > 0:
        row_limit = min(1024//bc,4095//bc,1024)
        if requested_rows and requested_rows > row_limit:
            raise ValueError('Requested tile exceeds the native vector or transfer limit')
        rows = requested_rows or row_limit
        while rows > 0:
            layout = _layout(c,rc,a,b,p,m,mode,rows,bc,masked,graph)
            if layout[-1]+metadata_bytes+reserve <= ub:
                break
            if requested_rows:
                rows = 0
            else:
                rows -= 1
        if rows:
            break
        if requested_c:
            raise ValueError('Requested Ascend C tiling exceeds the UB budget')
        bc = bc//2
    if not bc or not rows:
        raise ValueError('The minimum native tile exceeds the device UB capacity')
    indices = []
    variants = (bc, c % bc or bc)
    for count in variants:
        pitches,nc,output_rows,words,_ = _layout(c,rc,a,b,p,m,mode,rows,count,masked,graph)
        packet = torch.zeros(words,dtype=torch.int32,device='cpu')
        for lane in range(rows*count):
            node,channel = divmod(lane,count)
            values = (node*pitches[0]+channel*a,
                      node*pitches[1]+(0 if rc == 1 else channel*b),
                      node*pitches[2]+channel*p,
                      node*pitches[3]+channel*m)
            for column,value in enumerate(values):
                packet[column*nc+lane] = value*4
        for lane in range(output_rows):
            for q in range(8):
                packet[4*nc+8*lane+q] = (q*nc+lane)*4
        tail=4*nc+8*output_rows
        packet[tail:tail+_align(count)] = torch.arange(_align(count),dtype=torch.int32)*4
        if masked:
            for lane in range(rows*count):
                packet[tail+_align(count)+lane] = (lane//count)*8*4
        indices.append(packet.to(sample.device))
    result = dict(rows=rows,channels=bc,indices=tuple(indices),
                  bytes=layout[-1]+metadata_bytes,ub_bytes=ub,cores=cores,
                  term_capacity=term_cap)
    cache[key] = result
    return result


def raw(left,right,weight,index,plan,bank,mode=0,grad=None):
    output_nodes,c,a = left.shape
    graph=plan.get('_ascendc_graph')
    n=right.shape[0]
    rc,b = right.shape[1:]
    p = weight.shape[-1]
    m = plan['sizes'][3]
    metadata,width,terms,head,stride = _metadata(plan,mode,left.device)
    if graph is not None and mode == 0:
        output_nodes = graph.get("destination_nodes",output_nodes)
    masked=graph is not None and graph['mask'] is not None
    tile = _tiling(plan,left,c,rc,a,b,p,m,mode,width,terms,head,masked,graph is not None)
    index_kind = 0 if index is None else (2 if index.dtype == torch.int32 else 1)
    shape = [n,c,rc,a,b,p,m,width,mode,int(bank),index_kind,
             tile['rows'],tile['channels'],terms,head,stride,tile['term_capacity'],
             output_nodes,int(graph is not None),int(masked)]
    return load_library().weighted_map(left,right,weight,left if grad is None else grad,
        metadata if index is None else index,metadata,*tile['indices'],
        metadata if graph is None else graph['src'],metadata if graph is None else graph['dst'],
        graph['mask'] if masked else left,shape)
