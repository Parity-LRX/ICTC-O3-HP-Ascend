# Convert MACE checkpoints for Ascend

Install the conversion dependencies into the same environment as MACE-ICTC:

```bash
python -m pip install '.[conversion]'
```

Run conversion and numerical evaluation inside a scheduled compute allocation.
Conversion calibrates the fixed basis and path normalization in CPU FP64, then
stores the requested weight dtype. NPU execution requires FP32. Load only trusted
checkpoints: these files contain serialized Python modules.
The conversion checks use MACE 0.3.14 and e3nn 0.5.9. Keep the conversion
dependencies installed when loading the resulting foundation archive.

## MACE-MH-1

Download the official checkpoint from the
[MACE foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1),
then convert it:

```bash
python -m mace_ictc.cli.convert_foundation \
  --mace-model /path/to/mace-mh-1.model \
  --head omat_pbe --dtype float32 \
  --out /path/to/mace-mh-1-ictc-ascend.pt
```

The equivalent installed command is `ictc-ascend-convert-foundation`.
The selected head is a default for evaluation; all six original heads remain in
the archive:

- `matpes_r2scan`
- `mp_pbe_refit_add`
- `spice_wB97M`
- `oc20_usemppbe`
- `omol`
- `omat_pbe`

The adapter retains the original element embeddings, Agnesi distance transform,
ZBL repulsion, density normalization, residual paths, nonlinearities and readouts.
Tensor products and symmetric contractions use ICTC operators. Some retained
MACE stages use their original spherical component layout, with fixed basis
transforms at the operator boundaries. Conversion preserves the learned weights
and calibrates basis transforms and path normalization in CPU FP64.

This adapter supports frozen energy and force inference. Effective linear
matrices and embedding lookup tables are stored as buffers, so it is not a full
MACE-MH-1 fine-tuning implementation. The ordinary MACE-ICTC training API remains
available for its supported model architectures.

### Predict a structure

Build the native library following [HP_API.md](HP_API.md), then run:

```bash
export ICTC_ASCENDC_LIBRARY=/path/to/libictc_ascendc.so
python examples/foundation.py \
  --checkpoint /path/to/mace-mh-1-ictc-ascend.pt \
  --structure /path/to/cu.xyz --head omat_pbe --mode performance
```

Use `--mode capacity` to reduce message storage at the cost of throughput.
The atom count and neighbor graph are dynamic. Optional `--edge-chunk-size` and
`--radial-chunk-size` control intermediate storage without fixing graph shapes.
The example constructs a neighbor graph at the checkpoint cutoff and reports
energy in eV and forces in eV/Å. Neighbor construction and transfers are part of
this user example, rather than a resident-graph throughput measurement.
The example subtracts a common origin before casting coordinates to FP32.
Apply the same preparation to large absolute coordinates in your own graph
builder, preserving short displacement accuracy in the source precision.

### Python API

```python
import torch
import torch_npu
from mace_ictc.ascendc import enable
from mace_ictc.cli.convert_foundation import load_model
from mace_ictc.hp import energy_and_forces

enable(mode='performance')
model = load_model('mace-mh-1-ictc-ascend.pt', device='npu:0', head='omat_pbe')
model.configure_inference(chunk_edges=None, radial_chunk=32768)

# graph has positions [N,3], atomic_numbers [N], batch [N],
# edge_src/edge_dst [E], lattice edge_shifts [E,3], and cell [B,3,3].
result = energy_and_forces(model, **graph)
e0 = model.atomic_reference_energies(graph['atomic_numbers'])
result['atomic_energy'] += e0
result['energy'].index_add_(0, graph['batch'], e0)
```

The standard model forward and `energy_and_forces` return interaction energies.
Add the selected head's atomic reference energies exactly once, as above.
Forces are unaffected by these constant offsets. `model.select_head(name)` changes
the selected retained head without changing weights. Rebuild the force graph
after changing a head; do not switch heads between a forward and its backward.

`model.forward_vectors(vectors, species, src, dst)` accepts Cartesian edge
vectors and compact species indices. It returns both `interaction_energy` and
`node_energy`; the latter already includes E0. Obtain species indices through
`model.species_indices(atomic_numbers)`.

### LAMMPS

The same converted FP32 archive loads directly through the Ascend pair style.
For a Cu system whose sole atom type is Cu:

```lammps
units metal
atom_style atomic
read_data cu.data
pair_style mff/torch 6.0 npu
pair_coeff * * /path/to/mace-mh-1-ictc-ascend.pt Cu
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
```

Set `MFF_ASCEND_HEAD=omat_pbe` to override the archive's selected head before
starting LAMMPS. For multiple atom types, list their element symbols in type
order. The pair style adds the selected E0 once and includes the retained ZBL
term in energies, forces and global virials. See the
[Ascend LAMMPS guide](../lammps_user_mfftorch/docs/ASCEND.md) for compilation,
MPI/HCCL launch and deployment limits.

## Linear-interaction MACE models

The separate baseline converter supports compatible single-head MACE models
with linear residual interactions:

```bash
python -m mace_ictc.cli.convert_mace \
  --mace-model /path/to/mace.model --dtype float32 \
  --out /path/to/mace-ictc.pth
```

Check the printed architecture requirements. This converter rejects unsupported
nonlinear foundation models; use the foundation command for MACE-MH-1.
Weights and pretrained-model licenses are not bundled with this library.
