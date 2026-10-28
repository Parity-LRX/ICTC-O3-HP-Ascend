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
- Fixed Cartesian recovery matrices replace repeated component expansion in
  the learned dipole and quadrupole heads. Trainable source maps and gates
  are evaluated on every call.
- Fused fixed-cell geometry computes reciprocal vectors, Poisson weights,
  screening and volume in one vector kernel, reading the current cell each
  time. Requested cell derivatives and custom or learned Green kernels retain
  the differentiable geometry path.
- Direct positive/negative amplitude maps read the FFT channel planes without
  Cartesian repacking. Paired linear adjoints preserve force-loss derivatives;
  live reciprocal-vector derivatives retain the general contraction path.
- Performance mode uses vectorized CANN assignment for large fields with few
  source channels; other shapes use native UB tiling. Capacity and native modes
  retain bounded native assignment. `ICTC_ASCEND_MESH_ASSIGNMENT=auto` is the
  default; `native` and `vectorized` select explicit schedules for profiling.
  The vectorized schedule uses more temporary memory. Low-level
  `mesh_spread(..., backend="ascend")` always uses the bounded native kernel.
- Caches hold mesh indices and fixed execution metadata. Live coordinates,
  cells, source tensors and learned Green weights are recomputed each call.

Recovery, geometry and amplitude optimizations are enabled automatically for
eligible inputs. For profiling, set `ICTC_ASCEND_MULTIPOLE_RECOVERY=0`,
`ICTC_ASCEND_RECIPROCAL_GEOMETRY=0` or
`ICTC_ASCEND_MULTIPOLE_AMPLITUDE=0` to select the corresponding general path.
The geometry and direct amplitude kernels use Triton Ascend from the `hp`
installation extra; the standalone Ascend C API retains its existing path
when that optional backend is unavailable.

## Measured cost relative to short-range evaluation

The following comparisons use one Ascend 910B3 or one Hygon K100, eager
FP32 with HF32/TF32 disabled. Both use the same saved C64/lmax-2 backbone,
two interactions, correlation 3, eight Bessel functions, radial MLP width 16
and cutoff polynomial order 5. The graph has 56 neighbors per atom at a
4.5-Angstrom cutoff. Source channels are one, with nonzero deterministic
learned heads and energy scales, PCS assignment, approximately 1-Angstrom
mesh spacing and sigma fixed at 1 Angstrom. Meshes are 32x32x32 for 4,096
atoms and 64x64x32 for 16,384 atoms.

Training includes energy, forces, parameter backward through forces and
AdamW with `foreach=False`; inference includes energy and forces. The K100
backbone uses its current optimized training and inference presets. Each
variant runs in a fresh process, with one active worker on the same physical
device for the Ascend pairs. Initial parameters match exactly. Energy, forces,
coordinate second gradients and all active parameter gradients are checked
across execution schedules and hardware using the budgets below.

Each entry is the additional time over that hardware's own short-range
measurement; the percentage divides this increment by its short-range time.
Three calls are timed per synchronized block, then divided by three; values
are medians of five blocks after at least two seconds of warmup. Neighbor
graph construction and input transfers are excluded.

| Atoms | Sources | Phase | Ascend before, ms (+%) | Ascend after, ms (+%) | K100, ms (+%) | Increment reduced |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4,096 | q | E/F inference | 14.20 (17.5%) | 12.23 (15.4%) | 14.65 (36.1%) | 13.9% |
| 4,096 | q | Training | 27.35 (12.0%) | 24.96 (11.0%) | 13.41 (11.9%) | 8.7% |
| 4,096 | q+mu+Q | E/F inference | 26.10 (32.2%) | 14.90 (18.7%) | 34.81 (85.7%) | 42.9% |
| 4,096 | q+mu+Q | Training | 42.32 (18.5%) | 24.34 (10.7%) | 30.38 (26.9%) | 42.5% |
| 16,384 | q | E/F inference | 40.70 (13.6%) | 37.58 (12.6%) | 17.51 (13.3%) | 7.7% |
| 16,384 | q | Training | 84.46 (9.4%) | 78.92 (8.8%) | 20.55 (5.2%) | 6.6% |
| 16,384 | q+mu+Q | E/F inference | 53.68 (18.0%) | 31.87 (10.7%) | 36.27 (27.5%) | 40.6% |
| 16,384 | q+mu+Q | Training | 99.08 (11.1%) | 61.14 (6.8%) | 40.91 (10.3%) | 38.3% |

The optimized 4,096-atom Ascend short-range baseline is 79.50 ms for energy
and forces and 227.20 ms for training. With q+mu+Q, the complete calls take
94.40 and 251.54 ms. The corresponding K100 short-range times are 40.62 and
112.75 ms; complete q+mu+Q calls take 75.43 and 143.13 ms. Incremental and
complete-model costs are distinct measurements.

The q+mu+Q increment falls by 38-43% across these two sizes and phases.
Scalar-only training remains more expensive on Ascend: at 16,384 atoms it
adds 78.92 ms, compared with 20.55 ms on K100. These are initialized-model
compute measurements; they do not establish fitted-potential accuracy or
long-trajectory MD behavior.

The original approximately 79-ms result uses the three-call format above.
Individually synchronized calls are also measured separately: at 4,096 atoms,
Ascend short-range/q+mu+Q inference takes 86.29/101.18 ms, and training takes
234.52/258.91 ms. These synchronization formats should be compared separately.

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
- Exact recovery matrices across channel tails, source-channel counts and
  basis mutations; differentiable or customized recoveries retain the general
  path. Fused geometry also checks live cell updates and disjoint frequency
  slices. Direct amplitude maps check compact/dense quadrupoles, absent moment
  degrees and mixed second derivatives.
- Fixed-cell weighted atomic energies, multiple graphs, empty inputs and
  standalone execution with MACE and Triton unavailable.
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
