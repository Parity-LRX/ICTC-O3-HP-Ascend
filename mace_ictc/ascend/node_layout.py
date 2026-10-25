"""Restore angular path order using node-sized row transfers."""
import os
import torch


def restore_output(value, index, plan):
    policy=os.environ.get('ICTC_ASCEND_NODE_REORDER','2')
    if policy not in ('0','1','2'):
        raise ValueError('The node-reordering policy must be zero, one or two')
    if policy=='0':
        return value.index_select(1,index)
    if policy=='1':
        return value.transpose(0,1).contiguous().index_select(0,index).transpose(0,1).contiguous()
    nodes,width,channels=value.shape
    # The table describes only a fixed angular permutation at this node count.
    # Bound its cache independently of feature and learned-weight storage.
    elements=nodes*index.numel()
    if elements*8+index.numel()*index.element_size()>64*1024**2:
        return value.transpose(0,1).contiguous().index_select(0,index).transpose(0,1).contiguous()
    key=nodes,width,str(value.device),id(index),index._version
    cache=plan.setdefault('_ascendc_node_row_indices',{})
    if key not in cache:
        rows=(torch.arange(nodes,device=value.device,dtype=torch.int64)[:,None]*width+index[None,:]).reshape(-1)
        if len(cache)>=4 or sum(t.numel()*t.element_size()+i.numel()*i.element_size() for i,t in cache.values())+rows.numel()*8+index.numel()*8>64*1024**2:
            cache.clear()
        cache[key]=index,rows
    return value.reshape(nodes*width,channels).index_select(0,cache[key][1]).reshape(nodes,index.numel(),channels)
