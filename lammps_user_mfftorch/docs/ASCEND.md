# Plain mff/torch on Ascend

The Ascend backend uses the ordinary `mff/torch` pair style with a persistent
eager PyTorch model. Ascend C evaluates supported model operators and packs and
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
shifts, so the model uses the owned nodes. Multiple ranks use a depth-dependent
halo and compute the energy of owned atoms only. Nodes unused by the model graph
are removed before device transfer. Skin edges remain in the cached graph and
are masked at the model cutoff using current coordinates.

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

Global virial is computed by differentiating owned energy under homogeneous
strain, including cell and periodic shifts. Per-atom stress is currently rejected.
Both Newton settings are supported. `newton off` avoids LAMMPS' ordinary MPI
reverse-force stage; with `newton on`, this pair style leaves its host ghost
forces zero so that stage does not count the model force twice.
The current multi-rank route retains a wide halo; HCCL does not remove duplicated
model computation in that halo. Per-layer feature exchange is future work.

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

References: [HCCL batched P2P API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/cpp/c10d_npu-ProcessGroupHCCL-batch_isend_irecv.md),
[LAMMPS communication design](https://docs.lammps.org/Developer_par_comm.html).
