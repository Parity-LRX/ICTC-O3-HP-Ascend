"""One initialized-model energy/force training step on an allocated Ascend NPU."""
import argparse
import json

import torch
import torch_npu
from mace_ictc.hp import create_model, configure_model, energy_and_forces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--lmax', type=int, default=2)
    parser.add_argument('--correlation', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--layers', type=int, default=2)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(73)
    model = create_model(channels=args.channels, lmax=args.lmax,
                         correlation=args.correlation, num_interactions=args.layers, cutoff=4.5)
    plan = configure_model(model, training=True, policy='generated')
    atoms = 27
    positions = torch.cartesian_prod(*(torch.arange(3) for _ in range(3))).float().npu()*1.7
    distance = torch.cdist(positions, positions)
    src, dst = torch.where((distance > 1e-5) & (distance < 4.5))
    graph = dict(positions=positions,
                 atomic_numbers=torch.tensor([1, 6, 8]*9, device='npu'),
                 batch=torch.zeros(atoms, dtype=torch.int64, device='npu'),
                 edge_src=src, edge_dst=dst, edge_shifts=positions.new_zeros(src.numel(), 3),
                 cell=torch.eye(3, device='npu')[None]*20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    prediction = energy_and_forces(model, **graph, training=True)
    # Synthetic zero targets demonstrate the API; this is not a fitted potential.
    loss = (prediction['energy']/atoms).square().mean()+prediction['forces'].square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    configure_model(model, policy='generated')
    prediction = energy_and_forces(model, **graph)
    print(json.dumps(dict(backend=plan['backend'], selected=plan['selected'],
                          loss=float(loss.detach()), energy_shape=list(prediction['energy'].shape),
                          force_shape=list(prediction['forces'].shape))))


if __name__ == '__main__':
    main()
