"""Immutable sparse routing metadata shared by Ascend tensor products and contractions."""
import torch


def sparse_product_plan(entries, sizes, device, dtype):
    """Build all unique-owner reductions from immutable exact-nonzero entries."""
    plans=[]
    for leg,size in zip((3,0,1,2),(sizes[3],sizes[0],sizes[1],sizes[2])):
        rows=[[] for _ in range(size)]
        for entry in entries:
            rows[entry[leg]].append(entry)
        k=max(1,max(map(len,rows)))
        padded=[row+[(0,0,0,0,0.)]*(k-len(row)) for row in rows]
        ids=torch.tensor([[entry[:4] for entry in row] for row in padded],
                         device=device,dtype=torch.int32).permute(2,0,1).contiguous()
        coefficients=torch.tensor([[entry[4] for entry in row] for row in padded],
                                  device=device,dtype=dtype)
        plans.append((ids,coefficients))
    return {"sizes":sizes,"reductions":plans,"entries":tuple(entries)}
