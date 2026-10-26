# Upstream provenance

This independent Ascend port starts from the original ICTC-O3-HP library:

- Source: https://github.com/Parity-LRX/ICTC-O3-HP.git
- Branch: main
- Revision: 9fdea47eef620013e9ea591f7d90534bb27671c8
- Upstream package version: 0.2.12
- Ascend distribution version: 0.1.0

The original Python namespace and parameter schema are retained for checkpoint
compatibility. `mace_ictc/ascend/` contains the new device backend; `ictc_o3/`
exposes standalone NPU operators. The original source checkout is unchanged.

Original notices and the PolyForm Noncommercial / preserved MIT license scopes
remain in this derivative. The unused bundled NVIDIA projection helper is
omitted. Retired CUDA/Triton kernels, CUDA/Kokkos/ML-IAP deployment code and
compiled-export entry points have been removed from this distribution. The
plain Ascend LAMMPS interface is maintained separately from those old paths.
Private validation harnesses,
measurements and cluster credentials are maintained outside this library.
