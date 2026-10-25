# Plain mff/torch on Ascend

The Ascend backend uses the ordinary `mff/torch` pair style with a persistent
PyTorch model. Ascend C evaluates supported model operators and packs and
accumulates model ghost-force buffers. HCCL transfers those buffers between NPUs.
LAMMPS retains its CPU integrator, neighbor lists, MPI domain decomposition,
atom migration and ownership-metadata communication. ML-IAP and a Kokkos device
backend are not required.

## Build

Use a fresh LAMMPS source tree for this backend. The installer refuses to
overwrite an existing CUDA `USER-MFFTORCH` package. Build on the target ARM64
compute node using the same Python environment used to run `torch_npu`:

```bash
python /path/to/ICTC-O3-HP-Ascend/lammps_user_mfftorch/install_ascend.py /path/to/lammps

cmake -S /path/to/lammps/cmake -B /path/to/build-ascend \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=/path/to/mpi/bin/mpicxx \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=OFF \
  -D PKG_USER-MFFTORCH=ON \
  -D Python3_EXECUTABLE=/path/to/venv/bin/python

cmake --build /path/to/build-ascend -j 8
```

Python development headers and `libpython` are required. The pair style links
Python, rather than CUDA LibTorch. The interpreter loads `torch_npu` and the
matching Ascend C shared library once per process. It does not start a separate
Python process at each MD step.

Source the CANN environment and build the native model library as described in
[the operator API guide](../../docs/HP_API.md). Set `ICTC_ASCENDC_LIBRARY` to its
absolute path. The native library must match the Python sources, PyTorch,
`torch_npu`, CANN and target device.

All compilation and numerical execution on a cluster must occur inside a compute
allocation. The commands above are intended for a Slurm batch script, not a
login-node terminal.

## Checkpoints

`pair_coeff` loads a trusted MACE-ICTC `.pth` training checkpoint directly. No
TorchScript or AOTInductor export is required. The saved architecture metadata,
selected raw/EMA/SWA state and atomic reference energies are restored; weights
are loaded strictly. The saved neighbor normalization, radial normalization,
cutoff and contraction configuration are retained.

Requirements:

- FP32 `pure-cartesian-ictd-fix`, ICTC angular basis and a supported Ascend HP model.
- Explicit `model_hyperparameters`, including channel count, angular degree,
  interaction depth, neighbor normalization, radial normalization, cutoff-polynomial
  order and product backend.
- Explicit `atomic_energy_keys` and `atomic_energy_values` covering the trained
  elements. Extra reference-energy entries are allowed.
- Short-range energy output. Long-range, spectral and runtime multi-fidelity
  models need a separate deployment implementation.

The cutoff passed to `pair_style` must match the checkpoint. Element symbols
passed to `pair_coeff` map LAMMPS atom types to trained elements. `NULL` mappings
are currently rejected. The default checkpoint state selection follows the
checkpoint's `default_state_source`, then the existing EMA/raw fallback policy.

## Run

```bash
source /path/to/ascend-toolkit/set_env.sh
export PYTHONPATH=/path/to/ICTC-O3-HP-Ascend:${PYTHONPATH:-}
export ICTC_ASCENDC_LIBRARY=/path/to/libictc_ascendc.so
export MFF_ASCEND_MODE=performance
export MFF_TORCH_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1

mpiexec -n 2 /path/to/build-ascend/lmp -in in.mfftorch_ascend
```

Use one MPI process per NPU. The pair style groups MPI ranks by their actual
processor names and assigns distinct local device indices in MPI rank order.
This also works with socket-only MPI implementations whose shared-memory
communicators contain only one rank. Make only the allocated devices visible through
`ASCEND_RT_VISIBLE_DEVICES`. For a single-node job, use at most as many ranks as
visible NPUs. Multi-node deployment needs a working HCCL network configuration.

### Captured execution

`MFF_ASCEND_EXECUTION=graph` captures the device energy, force and global-virial
calculation together with its HCCL exchanges. Subsequent calls replay the
recorded tasks. This uses TorchNPU ACLGraph and the same FP32 operators and
checkpoint. It does not require a TorchScript or `torch.compile` export.
This deployment capture is qualified for `MFF_ASCEND_MODE=performance`.
Use eager execution for capacity and native operator modes.

Build the current Ascend pair style before selecting this mode. Its topology
commit protocol uses host MPI to agree whether any rank needs to invalidate
the capture. Every rank then recaptures together. Model features, adjoints and
ghost forces continue to use HCCL.

The launcher configures task dispatch before PyTorch starts and divides the
actual near-CPU groups reported by `npu-smi` between visible devices:

```bash
python /path/to/ICTC-O3-HP-Ascend/lammps_user_mfftorch/launch_ascend.py \
  --execution graph --queue 1 -- \
  /path/to/mpi/bin/mpiexec -n 8 /path/to/build-ascend/lmp -in in.mfftorch_ascend
```

Use `--dry-run` to inspect the command and CPU bindings. Existing
`CPU_AFFINITY_CONF` settings are retained; `--affinity none` disables automatic
binding. For eager execution, select `--execution eager`; the launcher defaults
to task-queue level 2 unless a queue setting is already present.

Current positions and cell tensors are copied into the capture inputs on every
force evaluation. Exact topology, species and owner-schedule metadata can be
reused across neighbor rebuilds only when the MPI change flag is zero. A single
energy, force and virial program serves every call, keeping the kernel sequence
and memory pool fixed when virial requests alternate. Unrequested virial output
is discarded. The capture retains device memory until invalidation or shutdown;
capture and warmup costs are incurred again when metadata changes.

The tested TorchNPU stack requires task-queue level 0 or 1 for capture. Set
`MFF_ASCEND_EXECUTION=eager` for synchronized stage profiling; the graph runtime
records combined replay and result-download time under `graph_seconds`.

### Input example

Minimal input, with a cutoff of 3.2 Angstrom matching the saved model:

```lammps
units metal
atom_style atomic
boundary p p p
newton off

read_data system.data
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch 3.2 npu
pair_coeff * * model.pth H C O

velocity all create 300 20261025 mom yes rot no dist gaussian
fix integrator all nve
timestep 0.0005
thermo 10
thermo_style custom step temp pe ke etotal press
run 100
```

Use `units metal` and a three-dimensional simulation box. An optional
`depth N` argument checks the checkpoint interaction depth; it does not override
the trained model.

## Data path and communication

Single-rank periodic ghosts are folded into their local owners with lattice
shifts, so the model uses the owned nodes. Multiple ranks compute the energy of
owned atoms and support two halo layouts, selected with `MFF_ASCEND_HALO`:

| Setting | Neighbor halo | Intermediate features |
| --- | --- | --- |
| `layerwise` (multi-rank default) | One cutoff, plus skin | Received from owners through HCCL after each non-final interaction |
| `wide` | Interaction depth times the cutoff, plus skin | Recomputed in the multi-hop halo |

```bash
export MFF_ASCEND_HALO=layerwise
mpiexec -n 8 /path/to/build-ascend/lmp -in in.mfftorch_ascend
```

The layerwise layout uses edges whose destinations are owned atoms. Products,
message projection, self-connection and energy readouts evaluate owned rows.
Frozen initial embeddings are determined locally from species. Intermediate
features are exchanged on every force evaluation, and feature width comes from
the actual tensor. A custom autograd operation coalesces and returns feature
adjoints to owners through HCCL, then local autograd differentiates their
construction. This reverse exchange is required for forces and global virial.
The deployment exchange implements their first derivative; force-loss training
and Hessians require a separate higher-derivative implementation.

Nodes unused by the model graph are removed before device transfer. Skin edges
remain in the cached graph and are masked at the cutoff using current coordinates.
Fixed routing coefficients and UB transfer metadata are reused; ownership
indices and buffers refresh with the topology. Floating learned features are
recomputed. Single-rank computation is the same for both halo settings.

At a neighbor rebuild, LAMMPS forwards owner rank/index metadata, then exchanges
the index schedules through MPI. The schedule coalesces repeated periodic copies
of the same owner. On every force evaluation:

1. Ascend C gathers and sums ghost-force contributions into NPU send buffers.
2. HCCL point-to-point operations send and receive contiguous NPU buffers.
3. Ascend C accumulates received contributions into owned forces on the NPU.
4. One host transfer returns owned forces, owned atomic energies, the local scalar
   energy and global-virial contribution to LAMMPS.

The model writes only owned forces to LAMMPS. Its ghost force is not returned a
second time through LAMMPS MPI reverse communication. There is no host-MPI fallback
for failed HCCL model communication. Ordinary host coordinate communication and
atom migration remain part of LAMMPS.

Global virial is the homogeneous-strain derivative of owned energy, including
periodic shifts. It is evaluated from edge displacements and their energy
gradients, avoiding cancellation between absolute-coordinate and cell terms.
The force and virial path retains FP32 arithmetic. Per-atom stress is currently rejected.
Both Newton settings are supported. `newton off` avoids LAMMPS' ordinary MPI
reverse-force stage; with `newton on`, this pair style leaves its host ghost
forces zero so that stage does not count the model force twice.

## Qualification

The deployment stack is LAMMPS 30 Sep 2026, MPICH 4.2.3, Python 3.10.12,
PyTorch 2.7.1, `torch_npu` 2.7.1.post4 and CANN 9.0.0 on Ascend 910B3.
Execution uses FP32 with HF32 disabled.

Initialized H/C/O models pass CPU FP64 energy, force and global-virial comparisons
with one, two, four and eight MPI ranks on one node. Checks cover orthogonal
and restricted triclinic boxes, both Newton settings, atom sorting, repeated
neighbor rebuilds and empty ranks. A second architecture uses 12 channels,
angular degree 2, three interactions and correlation order 2. Every comparison
requires relative L2 error at most `1e-4` and pointwise error at most
`2e-5 + 1e-4 * abs(reference)`.

These initialized-model runs qualify the deployment and communication path;
they do not establish the physical accuracy of a fitted potential. Multi-node
execution and per-atom stress are outside this qualification.

Layerwise checks cover two, four and eight ranks, including actual owner changes,
sorting, repeated rebuilds, zero-edge graphs and mixed active/empty ranks. Real
feature-forward and feature-adjoint HCCL operation counts are checked alongside
the numerical results. Recorded one-, two-, four- and eight-rank checks use an
exclusive eight-NPU allocation. Partial-NPU allocations have not been qualified
with this software stack.

Captured execution additionally passes eight-rank checks for orthogonal and
triclinic boxes, a three-interaction architecture, mixed active/empty ranks,
zero-edge graphs and empty ranks. Each force evaluation is compared with a
fresh eager evaluation, alongside the FP64 initial, translated and final
configuration checks. Alternating force-only and global-virial requests,
sorting and actual owner migration are exercised. All ranks agree on capture
invalidation before any rank runs capture warmup communication.

## Measured MD performance

The following initialized H/C/O model has 64 channels, angular degree 2, two
interactions and correlation order 3. Runs use one Ascend 910B3 per MPI rank,
strict FP32, eager execution, a 4.5 Angstrom cutoff, 0.5 Angstrom skin and a
0.5 fs NVE timestep. The configurations have 56 physical neighbors and 80 cached
skin edges per atom initially. Values are median LAMMPS loop times from three
10-step runs after five warmup steps. Synchronized diagnostic runs are separate.

### Strong scaling with layerwise exchange

| NPUs | 4,096 atoms: ms/step | Speedup | Efficiency | 13,824 atoms: ms/step | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 134.90 | 1.00x | 100% | 504.08 | 1.00x | 100% |
| 2 | 92.60 | 1.46x | 72.8% | 248.75 | 2.03x | 101.3% |
| 4 | 71.52 | 1.89x | 47.2% | 135.22 | 3.73x | 93.2% |
| 8 | 75.77 | 1.78x | 22.3% | 95.74 | 5.27x | 65.8% |

Efficiency is `T1 / (ranks * Tranks)`. The larger eight-NPU point processes
144,389 atom-steps/s. At 4,096 atoms, four NPUs are faster than eight; per-rank
work is too small to sustain efficient scaling. The slightly superlinear
two-NPU result is specific to these measured shapes and timings.

### Wide and layerwise halos

| Atoms | NPUs | Wide: ms/step | Layerwise: ms/step | Wide / layerwise | Peak GiB/NPU: wide -> layerwise |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 2 | 156.99 | 92.60 | 1.70x | 8.37 -> 3.76 |
| 4,096 | 4 | 102.60 | 71.52 | 1.43x | 5.00 -> 2.10 |
| 4,096 | 8 | 76.15 | 75.77 | 1.01x | 3.01 -> 1.70 |
| 13,824 | 2 | 465.04 | 248.75 | 1.87x | 23.13 -> 12.45 |
| 13,824 | 4 | 246.01 | 135.22 | 1.82x | 12.54 -> 7.63 |
| 13,824 | 8 | 143.41 | 95.74 | 1.50x | 7.12 -> 3.48 |

Memory is the largest per-rank PyTorch NPU allocated peak. HCCL, driver memory
and unused allocator reserves are excluded. Both layouts use the same checkpoint
and initial structure. Initial energy and force comparisons across ranks and
layouts pass the numerical criteria above. The wide-layout measurements also
reuse fixed routing templates, so this comparison isolates the halo strategy.

### Single-rank overhead relative to direct PyTorch

| Atoms | Direct PT: ms/call | Engine with host transfers: ms/call | Full LAMMPS MD: ms/step |
| ---: | ---: | ---: | ---: |
| 4,096 | 134.30 | 134.24 | 134.90 |
| 13,824 | 502.91 | 504.22 | 504.08 |

Direct PT calls the same HP model with the same checkpoint, initial positions
and skin graph, keeping inputs and outputs on the NPU. The engine comparison
includes host input/output transfers. Five warmup calls precede three
alternating paired rounds of ten synchronized calls per route; each round has
nine energy/force calls and one energy/force/virial call. Engine overhead is
within timing variation at 4,096 atoms and 0.26% at 13,824 atoms. The MD column
uses evolving coordinates and includes CPU integration and neighbor work.

## Diagnostics

Set `MFF_ASCEND_STATS=/absolute/output/directory` to save one JSON report per
rank. Reports include actual HCCL P2P operation and payload-byte counts.
`MFF_ASCEND_PROFILE=1` synchronizes the model, communication and transfer stages
for diagnostic timings and adds synchronization overhead. Default unsynchronized
stage timings are host submission timings, not device execution measurements.
The communication stage includes Ascend C packing and owner accumulation as
well as HCCL transport. Set `MFF_ASCEND_PROFILE_SKIP=N` to exclude the first `N`
force calls from stage totals; `timed_calls` records the included call count.
Payload counters still cover all calls. CPU topology, position packing and output
scatter timings cover the complete run.

Reports include `halo_mode`, `halo_depth`, `interaction_depth`, separate feature
and force payload bytes, and actual forward/adjoint/force P2P operation counts.
In the layerwise layout, `model_seconds` includes feature exchange. Synchronized
`feature_forward_seconds` and `feature_reverse_seconds` isolate those intervals;
subtract them from the same rank's model total to estimate local model work.
`hccl_seconds` measures the final coordinate-force exchange. Phase maxima may
come from different ranks and must not be added to reconstruct wall time.

Reports also record the largest owned-node, compacted-model-node, edge,
ghost-row and coalesced-send-row counts. `peak_allocated_bytes` is the PyTorch NPU
allocator peak after model and HCCL initialization; `initial_allocated_bytes`
records its starting allocation. Their difference measures additional allocation
during the run. `peak_reserved_bytes` includes unused allocator cache. These are
per-rank allocator measurements, not total board memory or HCCL's external
allocations. LAMMPS `memory` reporting covers the pair style's C++ host buffers
and excludes Python/NPU allocations.

`MFF_HCCL_TIMEOUT` sets the process-group timeout in seconds (default 300).

Fixed-graph operator benchmarks are not LAMMPS step-time predictions. End-to-end
MD timing includes neighbor work, CPU integration, communication and transfers.
For comparison with a direct PyTorch call, use the same checkpoint, positions
and skin graph. Report resident-device model time, engine input/output transfers
and complete MD step time separately.

References: [HCCL batched P2P API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/cpp/c10d_npu-ProcessGroupHCCL-batch_isend_irecv.md),
[LAMMPS communication design](https://docs.lammps.org/Developer_par_comm.html).
