"""Predict one ASE-readable structure with a converted frozen foundation model."""
import argparse
from contextlib import redirect_stdout
import json
import sys
import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--structure', required=True)
    parser.add_argument('--head', default=None)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--mode', choices=('performance', 'capacity'), default='performance')
    parser.add_argument('--edge-chunk-size', type=int, default=None)
    parser.add_argument('--radial-chunk-size', type=int, default=32768)
    args = parser.parse_args()
    if args.device.startswith('npu'):
        import torch_npu
        from mace_ictc.ascendc import enable
        torch.npu.set_device(torch.device(args.device))
        enable(mode=args.mode)
    elif args.device != 'cpu':
        raise ValueError('Use an Ascend NPU or the CPU reference')
    from mace_ictc.cli.convert_foundation import load_model
    from mace_ictc.hp import energy_and_forces
    with redirect_stdout(sys.stderr):
        model = load_model(args.checkpoint, device=args.device, head=args.head)
    dtype = next(model.parameters()).dtype
    if args.device.startswith('npu'):
        model.configure_inference(chunk_edges=args.edge_chunk_size,
                                  radial_chunk=args.radial_chunk_size)
    atoms = read(args.structure)
    source, destination, shifts = neighbor_list('ijS', atoms, model.max_radius)
    order = np.argsort(destination, kind='stable')
    source, destination, shifts = source[order], destination[order], shifts[order]
    def tensor(value, *, integer=False):
        return torch.as_tensor(value, dtype=torch.int64 if integer else dtype).to(args.device)
    positions = atoms.positions-atoms.positions[0] if len(atoms) else atoms.positions
    graph = dict(positions=tensor(positions), atomic_numbers=tensor(atoms.numbers, integer=True),
                 batch=torch.zeros(len(atoms), device=args.device, dtype=torch.int64),
                 edge_src=tensor(source, integer=True), edge_dst=tensor(destination, integer=True),
                 edge_shifts=tensor(shifts), cell=tensor(atoms.cell.array[None]))
    result = energy_and_forces(model, **graph)
    e0 = model.atomic_reference_energies(graph['atomic_numbers'])
    atomic = result['atomic_energy'] + e0
    print(json.dumps(dict(head=model.head, atoms=len(atoms), edges=len(source),
                          energy_eV=float(atomic.sum()),
                          forces_eV_per_A=result['forces'].cpu().tolist()), indent=2))


if __name__ == '__main__':main()
