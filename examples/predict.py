"""Predict one structure with a trained or converted standard ICTC checkpoint.

Use examples/foundation.py for MACE-MH-1 archives. Run inside an Ascend allocation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu
from ase.io import read
from ase.neighborlist import neighbor_list
from mace_ictc.ascendc import enable
from mace_ictc.hp import configure_model, energy_and_forces
from mace_ictc.interfaces.ascend_lammps import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--structure', required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--out', default='prediction.json')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--mode', choices=('performance', 'capacity'), default='performance')
    args = parser.parse_args()
    torch.npu.set_device(torch.device(args.device))
    enable(mode=args.mode)
    model, e0, state = load_checkpoint(args.checkpoint, args.device)
    from mace_ictc.interfaces.foundation_hp import FoundationHP
    if isinstance(model, FoundationHP):
        raise ValueError('For foundation archives, use examples/foundation.py')
    configure_model(model, training=False, policy='generated')
    atoms = read(args.structure, index=args.index)
    if not len(atoms) or abs(atoms.cell.volume) < 1e-12:
        raise ValueError('Supply a nonempty structure and a nonsingular cell, including for molecules')
    cutoff = float(model.max_radius)
    src, dst, shifts = neighbor_list('ijS', atoms, cutoff)
    order = np.argsort(dst, kind='stable')
    def tensor(value, integer=False):
        return torch.as_tensor(value, dtype=torch.long if integer else torch.float32, device=args.device)
    numbers = tensor(atoms.numbers, True)
    if not set(atoms.numbers.tolist()).issubset(model.atomic_numbers):
        raise ValueError('Structure contains an element absent from the model')
    graph = dict(positions=tensor(atoms.positions-atoms.positions[0]), atomic_numbers=numbers,
                 batch=torch.zeros_like(numbers), edge_src=tensor(src[order], True),
                 edge_dst=tensor(dst[order], True), edge_shifts=tensor(shifts[order]),
                 cell=tensor(atoms.cell.array[None]))
    result = energy_and_forces(model, **graph)
    total = result['energy'].sum() + e0[numbers].sum()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(energy_eV=float(total.detach().cpu()),
                                     forces_eV_per_A=result['forces'].cpu().tolist(),
                                     atoms=len(atoms), cutoff_A=cutoff,
                                     checkpoint_state=state), indent=2)+'\n')
    print(f'Saved energy and forces to {target}')


if __name__ == '__main__':
    main()
