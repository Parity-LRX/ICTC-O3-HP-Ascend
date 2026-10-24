# Inherited LAMMPS interface

The `USER-MFFTORCH` sources are retained from ICTC-O3-HP for compatibility and
future porting. This release qualifies eager Python execution on one Ascend NPU.
It does not qualify a LAMMPS runtime on Ascend.

## Source layout

| Location | Purpose |
| --- | --- |
| `src/USER-MFFTORCH/` | Pair styles, LibTorch model loading and auxiliary solvers |
| `cmake/Packages/USER-MFFTORCH.cmake` | Package registration for the corresponding LAMMPS CMake layout |
| `cmake/Modules/Packages/USER-MFFTORCH.cmake` | Registration for the alternate CMake layout |
| `examples/in.mfftorch_minimal` | Inherited input example |

The existing GPU build uses CUDA/Kokkos and CUDA-aware LibTorch execution.
Its CUDA build options and device strings do not select an Ascend NPU.
The inherited export modules are not installed as Ascend command-line tools.
See [upstream provenance](../../UPSTREAM.md) for the original project and revision.

## Requirements for an Ascend deployment

An Ascend interface needs an NPU-compatible model runtime, device and stream
management, graph transfer and MPI halo handling. Energy and force agreement
must then be checked against the Python model before MD validation. Pressure,
virial and long-range outputs require their own checks.

Use the qualified [Python model example](../../examples/model.py) and
[operator API](../../docs/HP_API.md) for the current Ascend backend.
Run compilation, model export and numerical examples inside a scheduled compute
allocation on a cluster.
