# High-performance reciprocal electrostatics on Ascend

The standalone `ictc_o3` API and MACE-ICTC share the same Ascend C mesh
assignment, analytic adjoints, NPU real FFT and fused Cartesian multipole
contraction. Native execution uses eager FP32 with HF32 disabled. Mesh dimensions,
atom counts, source channels and neighbor graphs are runtime inputs; no graph
capture or fixed atom capacity is required.

## Install

Follow the [installation and native build instructions](O3_API.md#1-install-and-enable-the-high-performance-backend).
Rebuild `libictc_ascendc.so` with the electrostatic kernels, and set
`ICTC_ASCENDC_LIBRARY` to the resulting library. Run numerical examples inside
your Ascend compute allocation; use `sbatch` on a Slurm cluster.

## Standalone API

```python
import torch
import torch_npu
from ictc_o3 import MeshElectrostatics, mesh_shape_for_cell
from ictc_o3.ascendc import enable

torch.npu.set_device(0)
enable(mode="performance")

# Cell rows are lattice vectors. Select the mesh once at setup time.
cell = torch.diag(torch.tensor([16., 20., 24.], device="npu"))[None]
mesh = mesh_shape_for_cell(cell, spacing=1.0)
solver = MeshElectrostatics(
    mesh_size=mesh, smearing_sigma=1.0, assignment="pcs", backend="ascend",
).npu()
positions = (torch.rand(128, 3, device="npu") @ cell[0]).requires_grad_()
batch = torch.zeros(128, dtype=torch.int64, device="npu")
q = torch.randn(128, 2, device="npu", requires_grad=True)
mu = torch.randn(128, 2, 3, device="npu", requires_grad=True)
Q = torch.randn(128, 2, 3, 3, device="npu", requires_grad=True)

atomic_energy = solver(positions, batch, cell, q, dipole=mu, quadrupole=Q)
energy = atomic_energy.sum()
forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
loss = energy.square() / 128**2 + forces.square().mean()
loss.backward()  # Includes force-loss gradients of q, mu and Q.
```

Omit `dipole` and `quadrupole` for scalar sources. Omit only `quadrupole` for
monopoles and dipoles. Use the same coordinate units for the cell, positions
and `smearing_sigma`. These learned sources have the normalization of the
training functional; they are not automatically physical charges in electron
units. The caller supplies any required physical unit conversion.

| Argument | Shape / meaning |
| --- | --- |
| `positions` | `[N,3]`, Cartesian coordinates |
| `batch` | `[N]`, int64 graph IDs in `[0,B)` |
| `cell` | `[B,3,3]`, nonsingular row-vector cells |
| `q` | `[N,S]`, required scalar sources |
| `dipole` | Optional `[N,S,3]` Cartesian moments |
| `quadrupole` | Optional `[N,S,3,3]` Cartesian moments |
| Output | `[N,1]`, atomic energy contributions |

`mesh_size=M` retains the cubic `M³` convention; `(Mx,My,Mz)` selects a
rectangular grid. `mesh_shape_for_cell` rounds dimensions to FFT-friendly
2/3/5-smooth sizes. It synchronizes and detaches the cell, so call it during
setup and keep the mesh fixed while differentiating coordinates or strain.

`backend="ascend"` requires NPU FP32 and the native library. `"auto"` selects
that path when available and otherwise uses the differentiable Torch reference.
`"reference"` explicitly selects the reference, including CPU FP64. Native
adjoints support energies, forces, cell derivatives and mixed second derivatives;
use the reference for derivatives beyond second order. Floating atomic sums
are nondeterministic, as with other mesh scatter implementations.

Low-level `mesh_spread` and `mesh_gather` are also exported from `ictc_o3`.
They take fractional coordinates, graph IDs, `mesh_size`, `num_graphs` and
`assignment="cic"|"tsc"|"pcs"`. The default mesh layout is `[B,Mx,My,Mz,S]`;
`channels_first=True` selects `[B,S,Mx,My,Mz]`.

## Functional and compatibility

The default is a Gaussian-screened **reciprocal correction** with spectral
factor `exp(-sigma² |k|² / 2)`, B-spline window deconvolution and the zero mode
removed. It does not include real-space pairs, self energy or a neutralizing
background term. The inherited `full_ewald` option enables Gaussian screening;
its name alone does not make this standalone correction a complete Ewald sum.

Scalar sources are neutralized per graph by default. Scalar atomic energies
use the source-times-potential partition; multipole graph energies are divided
uniformly among atoms. Both give the same graph-energy and force conventions
as the K100 implementation. Triclinic real FFTs retain the two distinct folded
Nyquist contributions.

`smearing_sigma=1.0` is a fixed length for new API models. `None` preserves
the older box-dependent screening convention. Existing checkpoint loading does
not silently introduce a new sigma. Choose mesh resolution and sigma for the
trained functional and verify convergence; changing them changes the potential.
`boundary="slab"` retains the inherited padded-cell approximation with
`slab_padding_factor`, rather than a separate open-boundary Poisson solver.

## MACE-ICTC model and training

```python
import torch
from mace_ictc.hp import create_model, configure_model, energy_and_forces

model = create_model(
    channels=64, lmax=2, edge_lmax=2, num_interactions=2, correlation=3,
    cutoff=4.5, atomic_numbers=(1, 6, 8),
    electrostatics=dict(
        mesh_size=(24, 32, 40), smearing_sigma=1.0,
        source_channels=2, max_multipole_l=2,
    ),
).npu()
configure_model(model, training=True, electrostatics_backend="ascend")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# graph contains positions, atomic_numbers, batch, edge_src, edge_dst,
# edge_shifts (lattice coordinates), and cell; see the main usage guide.
optimizer.zero_grad(set_to_none=True)
prediction = energy_and_forces(model, **graph, training=True)
loss = ((prediction["energy"] - target_energy)**2).mean()
loss = loss + 10 * ((prediction["forces"] - target_forces)**2).mean()
loss.backward()
optimizer.step()
```

`electrostatics=None` preserves the short-range factory. The electrostatic
mapping defaults to `reciprocal-spectral-v1`, `mesh_fft`, periodic boundary,
PCS, neutralized scalar sources and a reciprocal-only screened kernel.
`max_multipole_l=0/1/2` selects q / q+mu / q+mu+Q. A learned multipole head
requires the corresponding backbone degrees. Fresh scalar energy scales and
multipole gates use the model's gentle-start initialization and remain trainable.

For the training CLI, use the supplied wrapper:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco_long_range.sh" data/hco models/hco-lr.pth \
    --long-range-mesh-size 24,32,40 --long-range-source-channels 2 \
    --long-range-max-multipole-l 2 --long-range-smearing-sigma 1.0
```

It extends the [H/C/O energy-and-force recipe](USAGE.md#3-train-an-energy-and-force-potential)
with the native reciprocal configuration. Single-NPU energy/force training
includes parameter mixed derivatives and AdamW. DDP training is not enabled
by this addition. Checkpoints save mesh shape, sigma, multipole rank and the
screened reciprocal-only convention; preserve these on restart.

## LAMMPS and HCCL

The existing Ascend `mff/torch` pair style loads trained FP32 reciprocal
checkpoints directly. Rebuild both the native library and the LAMMPS pair style
from this source revision. The pair style passes a common mesh origin while
retaining centered short-range coordinates on each rank. Follow the
[Ascend build and launch guide](../lammps_user_mfftorch/docs/ASCEND.md).

```lammps
units metal
atom_style atomic
boundary p p p
read_data system.data
pair_style mff/torch 4.5 npu
pair_coeff * * models/hco-lr.pth H C O
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
fix integrate all nve
timestep 0.0005
run 1000
```

Use one MPI rank per allocated NPU. Only owned atoms deposit sources; ghost
atoms are excluded from the reciprocal source count. HCCL sums compressed
Fourier amplitudes or scalar spectra, with differentiable reverse sums for
learned-source and coordinate forces. The FFT mesh is replicated on each rank.
Global atom counts are refreshed at topology rebuilds. Virials differentiate
both positions and the cell under homogeneous strain. This deployment supports
the reciprocal-only functional; dispersion and a full real-space Ewald solver
are separate implementations.

## Implementation

- Ascend C CIC/TSC/PCS spread/gather avoid `[N,stencil,S]` intermediates and
  tile channel vectors to the device UB capacity.
- Analytic coordinate VJPs and their mixed adjoints support force training.
- Paired real-FFT adjoints use CANN transforms and retain second derivatives.
- Fused positive/negative multipole contractions reduce small NPU launches.
- Performance mode uses vectorized CANN assignment for large fields with few
  source channels; other shapes use native UB tiling. Capacity and native modes
  retain bounded native assignment. `ICTC_ASCEND_MESH_ASSIGNMENT=auto` is the
  default; `native` and `vectorized` select explicit schedules for profiling.
  The vectorized schedule uses more temporary memory. Low-level
  `mesh_spread(..., backend="ascend")` always uses the bounded native kernel.
- Caches hold mesh indices and fixed execution metadata. Live coordinates,
  cells, source tensors and learned Green weights are recomputed each call.

## Measured performance

Selected paired measurements on one Ascend 910B3, eager FP32 with HF32 disabled:

| Workload | Atoms | Phase | Torch/CANN reference, ms | Ascend HP, ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| Standalone q+mu+Q, 1 source channel | 4,096 | Energy + forces | 36.19 | 17.15 | 2.11x |
| Standalone q+mu+Q, 1 source channel | 4,096 | Training | 60.22 | 27.39 | 2.20x |
| Standalone q, 1 source channel | 65,536 | Training | 211.45 | 174.14 | 1.21x |
| Standalone q, 4 source channels | 65,536 | Training | 357.19 | 109.94 | 3.25x |
| Standalone q+mu+Q, 1 source channel | 65,536 | Training | 467.60 | 287.87 | 1.62x |
| MACE C64/lmax 2 with q+mu+Q | 4,096 | Energy + forces | 139.08 | 117.45 | 1.18x |
| MACE C64/lmax 2 with q+mu+Q | 4,096 | Training | 314.96 | 284.26 | 1.11x |

The standalone scan covers 1,024–65,536 atoms and monopole, dipole and
quadrupole sources. Whole-model pairs use the same Ascend HP backbone on both
sides; the electrostatic execution schedule changes. Models are initialized
C64/lmax-2 or C128/lmax-3 networks with two interactions, correlation 3 and a
nonzero long-range scale. Training includes energy, forces, parameter backward
and AdamW. Inference includes complete energy and force evaluation.
Standalone training treats moment tensors as the optimized parameters;
whole-model training updates all trainable network parameters.

Each backend runs in a fresh process on the same allocated NPU, with at least
two seconds of warmup and five synchronized timed calls. The reported latency
is the median. Graph construction and host/device input transfers are excluded.
The periodic lattice graph has 56 neighbors per atom at a 4.5-Angstrom cutoff.
Mesh spacing is approximately 1 Angstrom, PCS assignment is used and sigma is
fixed at 1 Angstrom. These are computation benchmarks, rather than fitted
potential accuracy or long-trajectory MD measurements.

The Torch/CANN reference uses the original mesh functional and full complex
FFTs. Multipole contractions are expressed in equivalent real arithmetic to
avoid the installed NPU runtime's unsupported complex batched matrix multiply.
The native path retains the same functional, including its triclinic Nyquist
terms; it introduces no radial approximation or reduced-precision arithmetic.

### Overhead relative to short-range evaluation

The following on/off pairs retain identical C64/lmax-2 backbone weights and the
same neighbor graph. Disabling long range removes its source head and mesh
module. The enabled model includes source prediction and all reciprocal
derivatives. Each variant uses nine synchronized timed calls on the same NPU;
the other settings follow the protocol above.

| Atoms | Sources | Phase | Short range, ms | With HP reciprocal correction, ms | Added, ms | Added |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4,096 | q | Energy + forces | 92.47 | 106.54 | 14.07 | 15.2% |
| 4,096 | q | Training | 237.25 | 265.55 | 28.30 | 11.9% |
| 4,096 | q+mu | Energy + forces | 90.80 | 110.79 | 19.99 | 22.0% |
| 4,096 | q+mu | Training | 237.70 | 266.97 | 29.26 | 12.3% |
| 4,096 | q+mu+Q | Energy + forces | 91.20 | 116.58 | 25.38 | 27.8% |
| 4,096 | q+mu+Q | Training | 238.38 | 284.80 | 46.43 | 19.5% |
| 16,384 | q+mu+Q | Energy + forces | 299.03 | 352.99 | 53.95 | 18.0% |
| 16,384 | q+mu+Q | Training | 895.28 | 997.04 | 101.76 | 11.4% |

Source channels are one in all these pairs. The short-range repeats in separate
source-order pairs are measured separately. These timings describe initialized
models with an active reciprocal term, rather than a fitted foundation model.

## Numerical and deployment checks

- CIC/TSC/PCS spread and gather: forward, first derivatives and mixed second
  derivatives for periodic and padded-slab boundaries, channel tails, both
  mesh layouts, empty inputs and channel counts crossing the UB tile size.
- Energy, forces, cell derivatives and force-loss gradients against the K100
  CPU FP64 functional: monopoles, dipoles and quadrupoles, odd/even rectangular
  meshes, triclinic cells, multiple source channels and empty graphs. Legacy
  cubic meshes with box-dependent screening are also checked.
- Learned Green filters with nonzero parameters, parameter-only force-loss
  backward, and batch-validation cache invalidation after an in-place update.
- Three complete MACE models: energy/force and coordinate/cell mixed derivatives,
  network parameter gradients, AdamW, strict checkpoint loading and deployment.
  The training CLI also completes an energy-and-force epoch and reloads its
  rectangular-mesh, fixed-sigma checkpoint.
- Four-rank HCCL source/position/strain gradients, including empty owned ranks.
- Fresh LAMMPS build: seven single-/four-NPU q/mu/Q cases compare energy, forces
  and global virial, then execute three MD steps. A four-NPU case shifts the
  box origin by `(1e6,-2e6,3e6)` to check the common mesh frame while retaining
  centered short-range coordinates.

Native FP32 comparisons use a component budget of `2e-5 + 1e-4*abs(reference)`
and a relative L2 check. The mesh functional and precision remain unchanged
between performance schedules.
