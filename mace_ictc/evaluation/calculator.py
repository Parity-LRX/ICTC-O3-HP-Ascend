"""ASE calculator for loaded MACE-ICTC models.

Supports energies, forces and the model's optional physical tensor outputs.
Use the dedicated Ascend LAMMPS engine for distributed HCCL deployment.
"""

import logging
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from mace_ictc.utils.graph_utils import radius_graph_pbc_gpu
from mace_ictc.utils.external_tensor_specs import pack_external_tensor_dict
from mace_ictc.utils.tensor_utils import map_tensor_values

logger = logging.getLogger(__name__)


class MyE3NNCalculator(Calculator):
    """ASE calculator wrapper for MACE-ICTC models.

    Args:
        model: Loaded MACE-ICTC model.
        atomic_energies_dict: Mapping from atomic number (int) to E0 (float).
        device: Torch device.
        max_radius: Neighbour-search cutoff (Å).
        external_tensor: Optional external field tensor for models trained
            with ``--external-tensor-rank``.  Shape ``(rank_dim,)`` or
            ``(1, rank_dim)`` on *device*.  ``None`` for non-field models.
        return_physical_tensors: If ``True``, call
            ``model.forward(return_physical_tensors=True)`` and store outputs
            in ``results["physical_tensors"]``.  Requires physical tensor heads.
        **kwargs: Forwarded to the ASE ``Calculator`` base class.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model: torch.nn.Module,
        atomic_energies_dict: Dict[int, float],
        device: torch.device,
        max_radius: float,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_id: Optional[int] = None,
        return_physical_tensors: bool = False,
        **kwargs: Any,
    ):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.max_radius = max_radius
        self.external_tensor = external_tensor
        self.fidelity_id = fidelity_id
        self.return_physical_tensors = return_physical_tensors

        self.keys = torch.tensor(
            list(atomic_energies_dict.keys()), device=device
        )
        self.values = torch.tensor(
            list(atomic_energies_dict.values()), device=device
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if return_physical_tensors:
            self.implemented_properties = ["energy", "forces"]  # extended below

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        pos = torch.tensor(
            self.atoms.get_positions(), dtype=torch.float64, device=self.device
        )
        A = torch.tensor(
            self.atoms.get_atomic_numbers(),
            dtype=torch.float64,
            device=self.device,
        )

        if any(self.atoms.pbc):
            cell = torch.tensor(
                self.atoms.get_cell().array,
                dtype=torch.float64,
                device=self.device,
            ).unsqueeze(0)
            pbc = tuple(bool(x) for x in self.atoms.pbc)
        else:
            cell = (
                torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0)
                * 100.0
            )
            pbc = (False, False, False)

        edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(
            pos, self.max_radius, cell, pbc=pbc
        )

        pos.requires_grad_(True)
        batch_idx = torch.zeros(len(pos), dtype=torch.long, device=self.device)

        mapped_A = map_tensor_values(A, self.keys, self.values)
        E_offset = mapped_A.sum()

        # Build forward kwargs — only pass what the model accepts
        fwd_kwargs: Dict[str, Any] = {}
        if self.external_tensor is not None:
            external_specs = getattr(self.model, "external_tensor_specs", None)
            if external_specs and isinstance(self.external_tensor, Mapping):
                packed_external = pack_external_tensor_dict(
                    self.external_tensor,
                    external_specs,
                    device=self.device,
                    dtype=pos.dtype,
                )
                if packed_external is not None:
                    fwd_kwargs["external_tensor"] = packed_external
            else:
                fwd_kwargs["external_tensor"] = self.external_tensor
        if self.return_physical_tensors:
            fwd_kwargs["return_physical_tensors"] = True
        if self.fidelity_id is not None and int(getattr(self.model, "num_fidelity_levels", 0) or 0) > 0:
            fwd_kwargs["fidelity_ids"] = torch.tensor([int(self.fidelity_id)], dtype=torch.long, device=self.device)

        out = self.model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell, **fwd_kwargs)

        if self.return_physical_tensors and isinstance(out, tuple):
            atom_energies, physical_out = out
        else:
            atom_energies = out
            physical_out = None

        E_total = atom_energies.sum() + E_offset
        grads = torch.autograd.grad(E_total, pos)[0]

        self.results["energy"] = E_total.item()
        self.results["forces"] = -grads.detach().cpu().numpy()

        if physical_out is not None:
            self.results["physical_tensors"] = {
                name: {
                    l: blk.detach().cpu().numpy()
                    for l, blk in l_dict.items()
                }
                for name, l_dict in physical_out.items()
            }
