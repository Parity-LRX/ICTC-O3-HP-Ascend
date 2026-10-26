"""Differentiable row gathers and periodic shifts for eager NPU geometry."""
import os
import torch


def edge_vectors(positions, cell, shifts, batch, source, destination):
    policy=os.environ.get('ICTC_ASCEND_GEOMETRY_LAYOUT','2')
    if policy not in ('0','1','2'):
        raise ValueError('The Ascend geometry policy must be zero, one or two')
    if policy=='0':
        return None
    from ictc_o3.ascendc.runtime import is_enabled,mode
    if not is_enabled() or mode()!='performance':
        return None
    if policy=='2' and cell.shape[0]==1:
        try:
            version=batch._version
        except RuntimeError:
            return None
        key=version,batch.numel(),batch.device
        if getattr(batch,'_ictc_ascend_single_cell_batch',None)!=key:
            if batch.numel():
                bounds=torch.stack((batch.amin(),batch.amax())).cpu().tolist()
                if bounds!=[0,0]:
                    raise ValueError('A single cell requires zero-valued graph indices')
            batch._ictc_ascend_single_cell_batch=key
        # Valid single-graph metadata selects this cell for every edge. Keep
        # the cell and shifts live so strain and higher derivatives propagate.
        from ictc_o3.ascendc.linear import matmul
        shift_vectors=matmul(shifts,cell[0])
    else:
        edge_batch=batch.index_select(0,source)
        edge_cells=cell.index_select(0,edge_batch)
        from .channel_message import _BatchedMatrix
        shift_vectors=_BatchedMatrix.apply(shifts[:,None,:],edge_cells,False,False).squeeze(1)
    return (positions.index_select(0,destination)-
            positions.index_select(0,source)+shift_vectors)
