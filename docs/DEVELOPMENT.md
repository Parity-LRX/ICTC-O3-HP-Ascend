# Ascend native development

ICTC-O3-HP-Ascend integrates Ascend C kernels into PyTorch through
`torch.library`, `PrivateUse1` registration and the current TorchNPU stream.
It uses CANN's kernel-launch integration described in the
[official PyTorch adaptation guide](https://asc.gitcode.com/guide/programming_guide/advanced_programming/ai_framework_adaptation/pytorch_framework.html).
The deployment artifact is a PyTorch operator library, `libictc_ascendc.so`.

## Source layout

```text
ictc_o3/                    Standalone O(3) implementation
  basis/                    Bases, coupling coefficients and layouts
  ops/                      Equivariant layers and contraction plans
  autograd/                 Derivative dependency utilities
  backends/ascend/          PyTorch execution schedules
  ascendc/
    CMakeLists.txt          Native build and runtime installation
    cmake/
      AscendC.cmake         Installed CANN ASC toolchain discovery
      TorchNPU.cmake        PyTorch/TorchNPU includes, libraries and ABI
    include/map_params.h    Shared Host/kernel tiling descriptor
    include/kernel_launch.h Host-callable kernel launch declarations
    op_host/                Tensor validation, tiling, launch and registration
    op_kernel/              Vector and Cube device kernels
    build.py                Explicit Python build entry point
    runtime.py              Explicit library loading and dispatch selection
mace_ictc/                  Models, training, conversion and deployment
lammps_user_mfftorch/       Plain LAMMPS interface and HCCL integration
examples/                   User workflows and a Slurm allocation template
docs/                       Operator, model, deployment and development guides
```

The native directories describe responsibilities in this direct-launch
project. The `op_host` files contain PyTorch tensor validation, runtime tiling,
stream selection and operator registration. Thin ASC-compiled launchers invoke
each device kernel with `<<<...>>>`, and PyTorch loads the resulting library with
`torch.ops.load_library`.

## Build environment

The qualified target is Ascend 910B3 on Linux ARM64, with CANN 9.0.0,
Python 3.10, PyTorch 2.7.1, `torch_npu` 2.7.1.post4 and Triton-Ascend 3.2.1.
Use the driver, firmware, toolkit and framework releases matched by your site.
The [TorchNPU compatibility guide](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
publishes the framework compatibility matrix. The builder checks framework
major/minor compatibility; that check does not replace the complete matrix.

Install the platform's PyTorch and TorchNPU first, then install `.[hp]`.
This extra supplies the project's CMake dependency (3.20 or later), Ninja and
Triton-Ascend. Source the site's CANN `set_env.sh` before compilation and
execution. A host C++ compiler and the CANN compiler development package are
required.

The native build follows the CANN 9
[ASC CMake interface](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00039.html):
`find_package(ASC)` configures BiSheng, the project enables the `ASC` and `CXX`
languages, and kernels compile with the target SoC selected explicitly.
Host bindings call declarations in `kernel_launch.h`; the corresponding
`<<<...>>>` launchers are compiled together with each device kernel. Each
device source contains one kernel entry point and explicitly selects its
Vector or Cube execution type.

## Build and install

Run the following inside a compute allocation after sourcing CANN. On Slurm,
submit the command through the [batch template](../examples/ascend_job.sbatch)
as shown in the [user guide](USAGE.md#1-install-and-build).

```bash
python -m ictc_o3.ascendc.build \
  --output /path/to/build/ictc_ascendc \
  --soc Ascend910B3 \
  --jobs 4
export ICTC_ASCENDC_LIBRARY=/path/to/build/ictc_ascendc/libictc_ascendc.so
```

To install the runtime separately from the build directory:

```bash
python -m ictc_o3.ascendc.build \
  --output /path/to/build/ictc_ascendc \
  --soc Ascend910B3 \
  --install-prefix /path/to/ictc-native \
  --jobs 4
export ICTC_ASCENDC_LIBRARY=/path/to/ictc-native/lib/libictc_ascendc.so
```

The builder prints the resulting library path. Installation uses the
`ictc_runtime` CMake component and places only the runtime library and its
build metadata under `PREFIX/lib`. Framework and CANN libraries must remain
available in the deployment environment. The installed library retains search
paths for the build environment's framework and toolkit libraries; rebuild it
in a different software environment.

| Option or setting | Meaning |
| --- | --- |
| `--output` | Required out-of-source CMake build directory |
| `--soc` | Target chip name, such as `Ascend910B3` |
| `SOC_VERSION` | Target used when `--soc` is omitted; otherwise the current NPU is queried |
| `--cann-path` | Explicit CANN toolkit root |
| CANN environment | Toolkit fallback order: `ASCEND_CANN_PACKAGE_PATH`, `ASCEND_HOME_PATH`, `ASCEND_INSTALL_PATH` |
| `--install-prefix` | Optional runtime installation directory |
| `--build-type` | `Release` (default), `Debug` or `RelWithDebInfo` |
| `--jobs` | Positive compilation parallelism, default 4 |
| `ICTC_ASCENDC_CMAKE` | Optional absolute path to a CMake executable |
| `CXX` | Host C++ compiler selection |

An explicit target avoids NPU discovery during compilation. The site still
determines where compilation is permitted. The builder forwards the selected
toolkit to CANN's compiler discovery in its child processes; it does not change
the caller's shell environment. See the CANN 9
[compiler options](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/compiler/BishengCompiler/atlas_bisheng_10_0010.html)
for target chip naming and architecture options. Only the target above is
qualified by this project.

`ictc_ascendc_build.json` accompanies the library and records the toolkit path,
SoC, build type, framework versions, C++ ABI and Host architecture. Use a new
build directory when the target,
toolkit, framework environment or ABI changes. Rebuild after native source
updates. Existing build caches with conflicting target/toolkit/ABI paths are
rejected rather than reused silently.

## Runtime and compatibility

```python
import torch
import torch_npu
from ictc_o3.ascendc import enable

torch.npu.set_device(0)
enable(mode="performance")
```

`enable()` reads `ICTC_ASCENDC_LIBRARY` or accepts an explicit path. Importing
the package never starts compilation. The legacy `mace_ictc.ascendc.build`
command and `mace_ictc.ascendc` API alias the owned `ictc_o3` implementation.
Existing checkpoint class paths remain available.

Native operands use eager FP32 with autocast and HF32 disabled. Host tiling
queries actual UB and core capacities; N/E and neighbor sets remain dynamic.
Python autograd wrappers provide the existing first and higher derivative
paths. Training, inference and the plain LAMMPS interface share this library.
See [O(3) API](O3_API.md), [model API](HP_API.md) and
[LAMMPS deployment](../lammps_user_mfftorch/docs/ASCEND.md) for their contracts.

Both wheels and source distributions include the native CMake files, Host and
device sources, shared headers and basis assets. Native binaries are compiled
for the deployment environment after installation. Keep build products,
checkpoints, raw validation data and benchmark harnesses outside the source
checkout.
