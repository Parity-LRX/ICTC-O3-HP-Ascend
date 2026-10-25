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

### Dynamic execution and host dispatch

The backend uses dynamic eager execution. Atom counts, neighbor edges, species
and owner schedules are refreshed after neighbor rebuilds and atom migration.
It does not capture or replay a fixed-shape device program.

The launcher configures task dispatch before PyTorch starts and divides the
actual near-CPU groups reported by `npu-smi` between visible devices:

```bash
python /path/to/ICTC-O3-HP-Ascend/lammps_user_mfftorch/launch_ascend.py \
  --queue 2 -- \
  /path/to/mpi/bin/mpiexec -n 8 /path/to/build-ascend/lmp -in in.mfftorch_ascend
```

Use `--dry-run` to inspect the command and CPU bindings. Existing
`CPU_AFFINITY_CONF` and `TASK_QUEUE_ENABLE` settings are retained unless explicitly
overridden. `--affinity none` disables automatic binding. Queue level 2 is the
launcher default; it controls host task dispatch, not graph capture.

### Dynamic work reduction

Compact owned outputs are enabled by default for multi-rank layerwise execution.
Other work-reduction options below are disabled by default, and feature gathers
use the native route. These controls apply to live atom and neighbor data; none
requires a fixed atom count, padded neighbor capacity or captured device program.

| Environment variable | Effect |
| --- | --- |
| `MFF_ASCEND_PRUNE_SKIN=1` | Filters inactive skin edges using current host coordinates before device evaluation. The original NPU cutoff mask remains enabled. |
| `MFF_ASCEND_COMPACT_OUTPUTS=1` | Omits redundant zero-ghost rows from projections, self-connections and readouts in multi-rank layerwise execution. Default: `1`; set `0` to disable. |
| `MFF_ASCEND_OWNER_LINEAR=1` | Evaluates the frozen input linear map on owned features before exchanging them. |
| `MFF_ASCEND_OWNER_OUTPUT=1` | Aggregates messages directly into owned destinations while retaining halo source features. |
| `MFF_ASCEND_FEATURE_GATHER=index_select` | Uses device `index_select` for feature gathers; adjoint accumulation still uses Ascend C. Default: `native`. |
| `MFF_ASCEND_OVERLAP_FEATURES=1` | Overlaps feature communication with independent radial work on an eager device stream. Requires `MFF_ASCEND_OWNER_LINEAR=1`. |

Skin filtering uses the coordinates already available to the CPU integrator.
It recomputes a conservative distance bound on every force call, including a
margin for FP32 device arithmetic. Device indices are reused only when the newly
computed mask is identical. Atom migration, neighbor rebuilds and changing edge
counts remain supported. Filtering costs CPU time and does not shrink the
HCCL ownership schedule, so its benefit depends on the model and system size.
It is incompatible with `assume_edges_within_radius=True`.

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

Nodes unused by the model graph are removed before device transfer. By default,
skin edges remain in the cached graph and are masked at the cutoff using current
coordinates. Optional dynamic skin filtering removes provably inactive edges
before model evaluation while retaining the NPU cutoff mask.
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

The independent CPU FP64 oracle explicitly enables FP64 internal arithmetic and
checks floating output dtypes. Its virial comes from coordinate/cell strain,
independently of the deployment's edge-displacement contraction.

The dynamic work-reduction options are checked together with actual MPI owner
changes, restricted triclinic boxes, a second architecture, mixed active/empty
ranks and zero-edge cases. Compact outputs are also checked with the standard
pre-linear feature exchange. Skin-filter checks include 54 cutoff-boundary,
periodic-coordinate and moving-edge cases against NPU cutoff decisions.

## Measured MD performance

The initialized H/C/O model has 64 channels, angular degree 2, two interactions
and correlation order 3. Each MPI rank uses one Ascend 910B3. All rank counts
and sizes use the same checkpoint, strict FP32 with HF32 disabled, dynamic eager
execution, a 4.5 Angstrom cutoff, 0.5 Angstrom skin, density 0.125 atoms/Angstrom^3
and a 0.5 fs NVE timestep. There are initially 56 physical neighbors and 80
cached skin edges per atom. Skin pruning and feature overlap are disabled;
gathering uses the native route and compact owned outputs are enabled.

Values are median complete LAMMPS loop times from three ten-step runs after
five warmup steps. CPU integration, transfers and communication are included.
Neighbor rebuilds use LAMMPS' displacement check; separate numerical checks
force rebuilds and owner migration. Each run has an exclusive eight-NPU
allocation, including runs with fewer active MPI ranks. No custom
`CPU_AFFINITY_CONF` was set. To reproduce the dispatch configuration from a
clean shell:

```bash
export MFF_ASCEND_COMPACT_OUTPUTS=1
python /path/to/ICTC-O3-HP-Ascend/lammps_user_mfftorch/launch_ascend.py \
  --affinity none --queue 2 -- \
  /path/to/mpi/bin/mpiexec -n 8 /path/to/build-ascend/lmp -in in.mfftorch_ascend
```

The input cutoff must match the checkpoint; use 4.5 Angstrom for these
measurements. The earlier 3.2 Angstrom example illustrates a different checkpoint.

### Strong scaling and memory

| Atoms | NPUs | ms/step | atom-steps/s | Speedup | Efficiency | Allocated GiB/NPU | Reserved GiB/NPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1 | 117.04 | 34,998 | 1.00x | 100.0% | 7.14 | 7.30 |
| 4,096 | 2 | 80.73 | 50,737 | 1.45x | 72.5% | 3.69 | 4.19 |
| 4,096 | 4 | 64.95 | 63,063 | 1.80x | 45.0% | 2.09 | 2.39 |
| 4,096 | 8 | 68.90 | 59,447 | 1.70x | 21.2% | 1.06 | 1.22 |
| 13,824 | 1 | 358.40 | 38,571 | 1.00x | 100.0% | 24.03 | 24.57 |
| 13,824 | 2 | 200.91 | 68,806 | 1.78x | 89.2% | 12.26 | 12.83 |
| 13,824 | 4 | 117.82 | 117,336 | 3.04x | 76.1% | 6.87 | 7.22 |
| 13,824 | 8 | 86.10 | 160,564 | 4.16x | 52.0% | 3.13 | 3.59 |
| 32,768 | 1 | 875.32 | 37,435 | 1.00x | 100.0% | 56.95 | 58.03 |
| 32,768 | 2 | 455.26 | 71,977 | 1.92x | 96.1% | 32.02 | 36.34 |
| 32,768 | 4 | 235.64 | 139,058 | 3.71x | 92.9% | 14.51 | 14.96 |
| 32,768 | 8 | 131.83 | 248,559 | 6.64x | 83.0% | 7.30 | 7.55 |

Efficiency is `T1 / (ranks * Tranks)`, using the single-rank time from the same
system size, model and runtime configuration. The eight-NPU result at 32,768
atoms is 6.64x speedup and 83.0% efficiency. At 4,096 atoms, four NPUs are faster
than eight. Increasing system size amortizes dispatch and communication, but
these three points do not establish monotonic scaling for other sizes or models.

Allocated and reserved memory are the largest per-rank PyTorch NPU allocator
peaks after model and HCCL initialization. Reserved memory includes unused
allocator cache. Neither column includes all HCCL or driver allocations.

### Matched direct PyTorch control

| Atoms | Direct PT ms/call | Engine ms/call | Engine overhead |
| ---: | ---: | ---: | ---: |
| 4,096 | 114.84 | 116.61 | 1.53% |
| 13,824 | 356.27 | 358.32 | 0.57% |
| 32,768 | 876.92 | 879.28 | 0.27% |

Direct PT uses the same checkpoint, initial positions and actual engine edge
list, keeping inputs and outputs on the NPU. The Python engine includes host
input/output transfers. Five warmup calls precede three alternating paired
rounds of ten synchronized calls per route; each round has nine energy/force
calls and one energy/force/virial call. Numerical comparisons pass for all three
sizes. The complete MD timings above use evolving coordinates and CPU integration;
small differences from the stationary control are expected.

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

With dynamic skin filtering, `edge_filter_seconds` records cumulative CPU
filtering time. `edge_candidates`, `edge_retained`, `edge_filter_updates` and
`max_active_edges` describe the live filtering decisions. These counters are
separate from the cached skin topology and HCCL ghost schedule.

`MFF_HCCL_TIMEOUT` sets the process-group timeout in seconds (default 300).

Fixed-graph operator benchmarks are not LAMMPS step-time predictions. End-to-end
MD timing includes neighbor work, CPU integration, communication and transfers.
For comparison with a direct PyTorch call, use the same checkpoint, positions
and skin graph. Report resident-device model time, engine input/output transfers
and complete MD step time separately.

References: [HCCL batched P2P API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/cpp/c10d_npu-ProcessGroupHCCL-batch_isend_irecv.md),
[LAMMPS communication design](https://docs.lammps.org/Developer_par_comm.html).
