# MACE-ICTC on Ascend: user guide

Train an energy-and-force potential, convert a supported MACE checkpoint, and
run prediction or molecular dynamics on Ascend NPUs. The installed commands
start with `ictc-ascend-`; the Python packages are `ictc_o3` and `mace_ictc`.

## Choose a workflow

| Starting point | Recipe |
| --- | --- |
| Structures with reference energies and forces | [Prepare data](#2-prepare-training-data), then [train a potential](#3-train-an-energy-and-force-potential) |
| A supported ordinary MACE model | [Convert to ICTC](#5-convert-an-ordinary-mace-model) |
| Official MACE-MH-1 weights | [Convert and select a head](#6-use-mace-mh-1) |
| A trained or converted standard ICTC checkpoint | [Predict a structure](#7-predict-with-a-standard-ictc-checkpoint) |
| A potential ready for MD | [Build and run LAMMPS](#9-build-and-run-lammps) |
| Standalone equivariant layers | [Use the O(3) API](#8-call-standalone-o3-operators) |
| Custom model integration | [Model and backend API](HP_API.md) |
| Recorded performance and accuracy checks | [Benchmark protocols and validation](#benchmark-protocols-and-numerical-validation) |

The production workflow uses eager FP32, HF32 disabled, dynamic atom counts and
dynamic neighbor graphs. Ascend C kernels use hardware-derived UB tiling;
dense transforms use CANN GEMMs. Model execution does not require `torch.compile`. In custom code, import
`mace_ictc` or `ictc_o3` before importing e3nn; package initialization registers
the compatibility needed by e3nn 0.4.4 constants on newer PyTorch versions.

## 1. Install and build

Use a Linux ARM64 environment with a compatible Ascend driver and CANN toolkit.
The qualified stack is Python 3.10, PyTorch 2.7.1, `torch_npu` 2.7.1.post4,
CANN 9.0.0 and Triton-Ascend 3.2.1 on Ascend 910B3. Install the matching
PyTorch/`torch_npu` wheels supplied for your platform before these commands.

```bash
git clone https://github.com/Parity-LRX/ICTC-O3-HP-Ascend.git
export ICTC_REPO=/absolute/path/to/ICTC-O3-HP-Ascend
export ICTC_ENV=/absolute/path/to/ascend-python-environment
export ICTC_SITE_ENV=/absolute/path/to/site_env.sh
```

Create `site_env.sh` with your site's toolkit and runtime setup, for example:

```bash
source /path/to/ascend-toolkit/set_env.sh
# Add your site's Ascend driver and MPI libraries if required.
```

### Slurm execution

Run installation, preprocessing, conversion, compilation, training, prediction
and MD inside compute allocations. The supplied
[`ascend_job.sbatch`](../examples/ascend_job.sbatch) accepts a command and its
arguments. Its site example requests `HW_910b`, one NPU, eight CPU cores and
32 GiB of host memory. Override partition, account, node and resource settings
for your cluster. It preserves the working directory and writes `ictc-JOBID.log`.

Install and build from the repository directory:

```bash
cd "$ICTC_REPO"
sbatch --account=YOUR_ACCOUNT examples/ascend_job.sbatch \
  python -m pip install -e '.[hp,train,preprocess,conversion]' \
    --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

For standalone O(3) operators, install only the `hp` extra instead of the
full workflow dependencies shown above:

```bash
sbatch --account=YOUR_ACCOUNT examples/ascend_job.sbatch \
  python -m pip install -e '.[hp]' \
    --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

The `hp` extra supplies CMake and Triton-Ascend, which provides the `triton`
namespace. Native compilation also needs a host C++ compiler and the CANN
compiler development package. Continue with the same native library build below.

After installation completes successfully:

```bash
sbatch --account=YOUR_ACCOUNT examples/ascend_job.sbatch \
  python -m ictc_o3.ascendc.build \
    --output /absolute/path/to/build/ictc_ascendc --jobs 4
```

Set the resulting library path for later jobs:

```bash
export ICTC_ASCENDC_LIBRARY=/absolute/path/to/build/ictc_ascendc/libictc_ascendc.so
mkdir -p /path/to/my-potential/{data,models}
cd /path/to/my-potential
```

Use the path printed by the builder. Rebuild after updating native sources,
CANN, PyTorch or `torch_npu`. Imports never start compilation. Keep the
conversion dependencies installed when loading foundation archives. The
`conversion` extra pins MACE 0.3.14 and e3nn 0.4.4 to match the
[upstream requirements](https://github.com/ACEsuit/mace/blob/v0.3.14/setup.cfg).

The `hp` extra includes CMake 3.20 or later. Native compilation uses the
installed CANN ASC toolchain. To choose the target explicitly and install a
runtime library outside the build tree:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  python -m ictc_o3.ascendc.build \
    --output /absolute/path/to/build/ictc_ascendc \
    --soc Ascend910B3 --install-prefix /absolute/path/to/ictc-native --jobs 4
```

After that job succeeds, set `ICTC_ASCENDC_LIBRARY` to
`/absolute/path/to/ictc-native/lib/libictc_ascendc.so`. The adjacent
`ictc_ascendc_build.json` records the toolkit path, target, framework versions
and C++ ABI. Build and install options are described in the
[development guide](DEVELOPMENT.md). The old `mace_ictc.ascendc.build` command
remains a compatibility alias.

The batch template sources `ICTC_SITE_ENV`, selects `ICTC_ENV/bin`, bounds CPU
threads and keeps caches in `.cache/ictc/`. It does not invent an NPU visibility
list; keep the scheduler's device isolation. For ordinary model workloads it
uses `TASK_QUEUE_ENABLE=1` unless you explicitly set it.

## 2. Prepare training data

Provide an extended XYZ dataset with element symbols, Cartesian positions,
cell/PBC information, one total energy per configuration and three force
components per atom. Use consistent units; these examples use Å, eV and eV/Å.
Labels below are `REF_energy` and `REF_forces`.

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  ictc-ascend-preprocess \
    --input-file data/reference.extxyz --output-dir data/hco \
    --elements H C O --atomic-energy-keys 1 6 8 \
    --energy-key REF_energy --force-key REF_forces \
    --max-radius 4.5 --train-ratio 0.9 --seed 42 --num-workers 4
```

Replace the elements and atomic numbers for your dataset. Include every present
atomic number. The output contains `processed_train.h5`, `processed_val.h5`,
`fitted_E0.csv` and split indices. Training must use the same cutoff as
preprocessing; the loader checks it. Keep H5 generation enabled.

The default split is random. Datasets with a `source` field on every frame
use the preprocessor's source-tail holdout; inspect the reported split and
choose a suitable independent validation set for your application.

### Atomic reference energies

The trainer does not read `fitted_E0.csv` automatically. Supply E0 explicitly;
the supplied [training recipe](../examples/train_hco.sh) reads its `Atom,E0`
columns and passes both `--atomic-energy-keys` and `--atomic-energy-values`.
It uses `DATA_DIR/fitted_E0.csv`, or the file selected by `E0_CSV`:

```bash
export E0_CSV=/absolute/path/to/reference-E0.csv
```

The CSV must have these two columns, with your calculation's atomic energies:

```text
Atom,E0
1,-13.6
6,-1000.0
8,-2000.0
```

These numbers illustrate the file format. Fitted E0 values may be
underdetermined for fixed-composition datasets; use your chosen reference
energies when available. The H5 dataset retains total energies, and the trainer
adds atomic reference energies once when forming total predictions. Without
explicit E0 arguments, the CLI uses inherited H/C/N/O fallback values.

For sharded preprocessing, merged H5 files link to `shard_*/` by absolute path.
Retain those directories and paths. The unsharded example above is the simplest
starting point.

## 3. Train an energy-and-force potential

After preprocessing has completed:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" data/hco models/hco.pth
```

This trains C64/lmax 2, two interactions, correlation 3, a 4.5 Å cutoff and a
Bessel radial basis, with AdamW and an energy-plus-force MSE loss. The recipe
loads the native Ascend C library explicitly. Its main execution options are:

```bash
--device npu:0 --dtype float32 \
--ascendc --ascendc-mode performance \
--product-backend ictd-bridge-u --angular-basis ictd --freeze-conv-tp-weight \
--operator-plan generated --generated-layout fused_pack
```

`--ascendc` loads `ICTC_ASCENDC_LIBRARY` before building the model. Without that
flag, the existing generated NPU route uses Triton/GEMM dispatch.
`--operator-plan generated` requires supported HP coverage; `auto` permits
reference fallback. Fixed convolution TP coefficients are frozen explicitly;
radial, linear, self-connection and symmetric-contraction weights are trained.
Energy-and-force training retains the second derivatives required by the force
loss. The native execution mode does not change the parameterization or dtype.

Hyperparameters are configurable. Arguments after the checkpoint override the
recipe defaults:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" data/hco models/hco-large.pth \
    --channels 128 --lmax 3 --max-ell 3 --num-interaction 3 --correlation 2 \
    --batch-size 2 --epochs 200 --lr 3e-4
```

Native symmetric contractions cover correlation 1, 2 and 3. Reduce batch size
when memory is insufficient. `--gradient-accumulation-steps 4` accumulates four
microbatches per update. This release supports single-NPU training; the CLI
rejects multi-NPU DDP on NPU. HCCL multi-NPU LAMMPS is a separate deployment path.

### Long-range terms

The supplied Ascend recipe explicitly sets:

```bash
--long-range-mode none --long-range-dispersion-mode none
```

The qualified Ascend workflow is short range. Native K100 monopole/dipole/
quadrupole electrostatics has not been ported and qualified here. Inherited
long-range configuration fields are not an Ascend HP recipe, and the Ascend
LAMMPS loader rejects checkpoints with added long-range terms. Use the
[K100 user guide](https://github.com/Parity-LRX/ICTC-O3-HP-K100/blob/main/docs/USAGE.md)
for that supported implementation.

## 4. Restart, warm-start and evaluate

The output checkpoint contains the best validation weights. Enable recovery
snapshots for resumable optimizer state:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" data/hco models/hco.pth \
    --checkpoint-interval-steps 1000
```

Continue using the same architecture, E0, cutoff and training configuration:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" data/hco models/hco.pth \
    --resume-checkpoint models/hco.recovery.pth --resume-training-state \
    --checkpoint-interval-steps 1000 --epochs 200
```

The recovery file exists after its interval has been reached. Optimizer state,
global step and epoch numbering are restored. Stateful scheduler history is not
serialized; retain the original scheduler configuration when resuming.
For a weights-only warm-start, omit `--resume-training-state`.
Use `--finetune` only when intentionally permitting missing/new parameter keys;
it is not required for a compatible strict weights-only restart.

For one validation pass:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" data/hco models/evaluation.pth \
    --resume-checkpoint models/hco.pth --eval-only
```

Inspect validation energy/force errors before using the model for MD.

## 5. Convert an ordinary MACE model

The standard converter expects a serialized `ScaleShiftMACE` module, usually a
`.model` file. A state-dict-only `.ckpt` needs its original architecture rebuilt
and exported first. Renaming a file does not provide that architecture.

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  ictc-ascend-convert-mace \
    --mace-model models/source.model --dtype float32 \
    --out models/mace-ictc.pth
```

Conversion maps the learned tensors into the ICTC representation and retains
E0, energy scale/shift, neighbor normalization, cutoff and supported radial
network dimensions. It calibrates fixed basis/path normalization in CPU FP64
and stores FP32 weights for Ascend. Use a trusted source checkpoint.

Supported standard models use ordinary/residual interactions, at least two
interactions, contiguous natural-parity hidden degrees with uniform channel
multiplicity, Bessel radial features, uniform correlation and a SiLU readout.
For nonlinear foundation architectures, use the next section. See the
[conversion guide](CONVERSION.md) for details.

Use the original and converted models on identical structures at matching
precision before deployment. Compare total energies and full force vectors.

### Continue training a converted standard model

Use a compatible training architecture, matching dataset cutoff and the source
E0. The CLI reconstructs stored channel/degree/interaction settings. Its radial
MLP builder uses `[64,64,64]`; additional converter-supported radial widths need
a matching model/trainer API configuration.

Extract the source E0 in a scheduled job. Save this as `extract_e0.py`:

```python
import csv
import torch

checkpoint = torch.load('models/mace-ictc.pth', map_location='cpu', weights_only=False)
with open('models/source-E0.csv', 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['Atom', 'E0'])
    writer.writerows(zip(checkpoint['atomic_energy_keys'].tolist(),
                        checkpoint['atomic_energy_values'].tolist()))
```

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" python extract_e0.py
```

After the job completes:

```bash
export E0_CSV="$PWD/models/source-E0.csv"
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  bash "$ICTC_REPO/examples/train_hco.sh" data/hco models/mace-finetuned.pth \
    --resume-checkpoint models/mace-ictc.pth --lr 1e-4
```

This starts a fresh optimizer with strict weight loading. Inspect architecture
and load messages. Unset `E0_CSV` before returning to dataset-fitted offsets.

## 6. Use MACE-MH-1

Download the official model from the
[MACE foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1).
Use the foundation converter for its nonlinear architecture:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  ictc-ascend-convert-foundation \
    --mace-model models/mace-mh-1.model --head omat_pbe --dtype float32 \
    --out models/mh1-ictc.pt
```

All six original heads are retained; `--head` selects the evaluation default.
The converted model preserves Agnesi transforms, ZBL repulsion, density
normalization, nonlinear residuals and original readouts. This foundation
adapter supports frozen-weight energy-and-force inference. MH-1 fine-tuning
is not implemented by this adapter.

Predict a structure whose elements are covered by the checkpoint:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  python "$ICTC_REPO/examples/foundation.py" \
    --checkpoint models/mh1-ictc.pt --structure data/cu.extxyz \
    --head omat_pbe --mode performance
```

The example prints total energy and force vectors as JSON. It adds the selected
head's E0 once. Use `--mode capacity` to select bounded message storage.
The [conversion guide](CONVERSION.md) lists the heads and the Python API.

## 7. Predict with a standard ICTC checkpoint

Use the same example for a trained model or a converted ordinary MACE model:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  python "$ICTC_REPO/examples/predict.py" \
    --checkpoint models/hco.pth --structure data/reference.extxyz \
    --index 0 --out prediction.json --mode performance
```

`prediction.json` contains total energy in eV and forces in eV/Å. The loader
restores saved architecture, normalization, raw/EMA/SWA selection and E0.
The example builds neighbors at the checkpoint cutoff and adds E0 exactly once.
Supply a nonsingular cell even for nonperiodic molecules, and use PBC flags
appropriate to the structure. Coordinates are shifted by a common origin before
casting to FP32, preserving short displacements at large absolute positions.

### Performance and capacity modes

| Setting | Intended use |
| --- | --- |
| `performance` | Default for throughput and full energy-and-force training |
| `capacity` | Frozen inference with chunk recomputation and smaller intermediates |

Select a mode before constructing a forward graph. With the Python API:

```python
from mace_ictc.ascendc import enable
from mace_ictc.hp import configure_model, energy_and_forces

enable(mode='performance')  # or 'capacity'
# model: an NPU FP32 model with frozen inference parameters
configure_model(model, training=False, policy='generated')
result = energy_and_forces(model, **graph)
```

Automatic chunks adapt to tensor widths, device memory and live allocations.
Memory savings and throughput depend on the model and system size. The HP API
returns interaction energies; the prediction examples add checkpoint E0.
Forces use autograd even with frozen weights. The helper enables gradients
internally; do not use `torch.inference_mode()` for force prediction.

## 8. Call standalone O(3) operators

```python
import torch
import torch_npu
from ictc_o3 import EdgeTensorProduct, direction_harmonics
from ictc_o3.ascendc import enable

torch.npu.set_device(0)
enable(mode='performance')
op = EdgeTensorProduct.create(channels=64, input_lmax=2, edge_lmax=2, device='npu')
features = torch.randn(1024, 64 * 9, device='npu', requires_grad=True)
directions = torch.nn.functional.normalize(torch.randn(1024, 3, device='npu'), dim=-1)
harmonics = torch.cat(direction_harmonics(directions, 2, policy='generated'), -1)
gates = torch.randn(1024, op.gate_width, device='npu', requires_grad=True)
output = op(features, harmonics, gates)
output.square().mean().backward()
```

Run a complete example in an allocation:

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" \
  python "$ICTC_REPO/examples/operators.py" --ascendc
```

The [standalone O(3) API](O3_API.md) specifies tensor-product layouts, graph messages,
symmetric contractions, direction harmonics, dense transforms and derivatives.
It also documents native build/activation through `ictc_o3.ascendc` and the
composed-layer example [`examples/o3_layers.py`](../examples/o3_layers.py).
Channels and angular degrees are runtime model choices; UB tiling includes
tails and does not require a fixed atom count.

## 9. Build and run LAMMPS

The Ascend interface is the plain `mff/torch` pair style. LAMMPS runs CPU
integration, neighbors and MPI domain decomposition; Ascend evaluates the model.
HCCL exchanges intermediate features, their adjoints and ghost-force buffers.
Use one MPI rank per visible NPU. Kokkos and ML-IAP are not needed.

### Compile

Use a fresh LAMMPS source tree, matching MPI and Python development headers.
Save these commands in a shell script and submit it with the batch template:

```bash
python "$ICTC_REPO/lammps_user_mfftorch/install_ascend.py" /path/to/lammps
cmake -S /path/to/lammps/cmake -B /path/to/build-ascend \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=/path/to/mpi/bin/mpicxx \
  -D BUILD_MPI=ON -D BUILD_OMP=OFF -D PKG_USER-MFFTORCH=ON \
  -D Python3_EXECUTABLE="$ICTC_ENV/bin/python"
cmake --build /path/to/build-ascend -j 8
```

```bash
sbatch --account=YOUR_ACCOUNT "$ICTC_REPO/examples/ascend_job.sbatch" bash build_lammps.sh
```

Use the same interpreter, CANN and native `.so` as model execution. The pair
style loads a persistent Python model in each MPI process and reads standard
training checkpoints or converted foundation archives directly. No tracing or
compiled model export is needed.

### Input and launch

Copy [`in.mfftorch_ascend`](../examples/in.mfftorch_ascend) into your working
directory and provide `system.data`. Its H/C/O example uses:

```lammps
pair_style mff/torch 4.5 npu
pair_coeff * * models/hco.pth H C O
```

Match the cutoff to the checkpoint and list elements in LAMMPS atom-type order.
For the converted MH-1 Cu example, change these lines to:

```lammps
pair_style mff/torch 6.0 npu
pair_coeff * * models/mh1-ictc.pt Cu
```

Set the runtime controls, then submit a single-node eight-NPU job:

```bash
export MFF_ASCEND_MODE=performance
export MFF_ASCEND_HALO=layerwise
export MFF_ASCEND_HEAD=omat_pbe
export MFF_TORCH_THREADS=2
export ICTC_OMP_THREADS=2
sbatch --account=YOUR_ACCOUNT --gres=npu:8 --exclusive \
  --cpus-per-task=32 --mem=80G "$ICTC_REPO/examples/ascend_job.sbatch" \
  python "$ICTC_REPO/lammps_user_mfftorch/launch_ascend.py" \
    --queue 2 -- /path/to/mpi/bin/mpiexec -n 8 \
    /path/to/build-ascend/lmp -in in.mfftorch_ascend
```

Use the site's MPI configuration to keep all ranks within the allocation.
The one-node 1/2/4/8-rank qualification used exclusive eight-NPU allocations;
partial-NPU isolation and multi-node deployment were not qualified on that stack.
For a single-rank starting run, request the same qualified allocation and change
`mpiexec -n 8` to `-n 1`. Single-NPU training above is independent of this MPI setup.

Global virial, changing atom ownership, neighbor rebuilds, triclinic cells,
empty ranks and both Newton settings are covered by deployment checks.
Per-atom stress is not implemented. See the
[Ascend deployment guide](../lammps_user_mfftorch/docs/ASCEND.md) for build details,
HCCL routing, diagnostics and measured complete MD steps.

## Benchmark protocols and numerical validation

<details>
<summary>Measurement setup and validation coverage</summary>

### C64 overview: refined OOM sampling

The C64 overview uses 25 successful measurements from the 2026-10-27
refinement, with fresh processes for every size and backend. It retains the
matched checkpoints, two interactions, correlation 3, 4.5 Å cutoff, 2 Å lattice
spacing and 56 directed neighbors per atom. Balanced periodic integer boxes
provide intermediate atom counts. The separate four-model sweep retains its
original measurements.

The bars above the C64 panels connect the last successful size to the first
observed OOM. Each bracket spans 16.7–20% of its last successful atom count:

| Workload | Backend | Last success (atoms) | First OOM (atoms) |
| --- | --- | ---: | ---: |
| Energy-and-force training | Original MACE/e3nn | 12,288 | 14,336 |
| Energy-and-force training | HP | 20,480 | 24,576 |
| Energy-and-force inference | Original MACE/e3nn | 24,576 | 28,672 |
| Energy-and-force inference | HP performance | 40,960 | 49,152 |
| Energy-and-force inference | HP capacity | 98,304 | 114,688 |

Training measures the ordinary energy-and-force loss, parameter backward and
AdamW. Each process passes the CPU FP64 small-case derivative gate; shared
sizes also pass full energy/force comparisons and mapped training-gradient
checks. Timings are medians of five synchronized calls after at least three
warmup calls and two seconds. The memory reference line uses the allocator's
reported device capacity of 60.96 GiB; curves show peak PyTorch allocations.

### Recorded operator and model sweeps

The MindChemistry comparison uses unchanged MindChemistry 0.2.0 from the official
[`r0.7` source](https://github.com/mindspore-ai/mindscience/tree/b233dc1ab152dfe29e0ff2cafb55c780dff34c67/MindChemistry)
with MindSpore 2.10.0. The path-preserving `uvu` tensor products match
natural-parity coupling paths, normalization, channel counts and live external
path weights. Forward + VJP includes gradients of features, direction harmonics
and path weights. Each point is the median of five synchronized calls after
three warmups. Graph compilation is performed separately for each measured
shape and excluded from timing. The PyNative C128/lmax-3 VJP endpoint reaches
OOM at 65,536 edges. CPU FP64 basis/path checks and input gradients agree to
relative errors around `1e-7`.

The broad scan uses original `mace-torch==0.3.14` and `e3nn==0.5.9`,
with the baseline's usual scripted operators and no `torch.compile`.
The installation recipe follows MACE's e3nn 0.4.4 pin; these figures retain
their recorded e3nn 0.5.9 environment.
The same initialized physical models are related by an explicit checkpoint
conversion and checked for energies, forces, coordinate second derivatives
and mapped parameter gradients. Models use two residual interactions,
correlation 3, full CG paths, eight Bessel functions, radial MLP widths
`[64,64,64]`, a 4.5 Å cutoff, H/C/O species and 56 directed neighbors/atom
(neighbor normalization 50). No long-range term or repulsion is added.
Inference freezes parameters and computes energy and forces. Training includes
`zero_grad`, the energy-and-force loss, backward and AdamW. Equivalent initial
functions and gradients do not imply identical AdamW trajectories after a
parameter rescaling.

Whole-model timings are medians of five individually synchronized calls after
at least two seconds and three calls of warmup. The largest C256/lmax-3
capacity sample uses one warmup and three timed calls. Separate three-call-block results
are not mixed into these curves. The broad performance/training data and
standalone TP data retain their qualified production paths; all 21 capacity
points use the updated implementation. Peak PyTorch allocation includes
resident graph/model tensors and, for training, optimizer state; external CANN
and driver allocations are excluded. Each new size
runs in a fresh process. Graph construction, neighbor lists, transfers and
compilation are excluded. Each measurement uses a single NPU; concurrent workers
are independent measurements.

Standalone TP curves use natural-parity, path-preserving `uvu` products,
lmax 2/3, and live external path weights. Forward-plus-VJP includes gradients
of features, harmonics and weights. TP inputs and basis/layout preparation
are outside timing. Each point uses at least two seconds of warmup, followed
by three rounds of three individually synchronized calls; the reported value
is the median of round medians. CPU FP64 small cases and full-batch sampled
outputs/gradients are checked. Whole-model points beyond baseline OOM retain
small-case CPU FP64 and full-size finite checks.

Single-NPU MACE-MH-1 measurements retain the v0.1.0 protocol: the same official checkpoint,
the `omat_pbe` head, a 6 Å cutoff and 78 neighbors/atom; median of five
synchronized calls after two warmups. These pretrained measurements are
separate from the four randomly initialized models above.

C64 multi-NPU results use complete dynamic LAMMPS runs on one eight-NPU node,
with 1/2/4/8 active NPUs, a 4.5 Å cutoff, 0.5 Å skin and 0.125 atoms/Å³.
There are initially 56 physical neighbors and 80 cached skin edges per atom.
Points are median complete loop times from three ten-step runs after five
warmup steps. Compact owned outputs and native gathers are enabled. The
[deployment guide](../lammps_user_mfftorch/docs/ASCEND.md#measured-md-performance)
provides the complete latency, throughput and memory measurements.

Pretrained MACE-MH-1 multi-NPU runs retain all six heads and select `omat_pbe`,
using periodic FCC Cu, a 6 Å cutoff and a 0.5 Å skin in performance mode.
Strong scaling fixes 10,976 atoms; weak scaling uses 4,000 atoms/NPU, reaching
32,000 atoms on eight NPUs. The same five-warmup/three-ten-step complete-loop
timing includes CPU integration, neighbor checks, transfers, energy, forces,
global virials and HCCL. See the [MH-1 scaling results](../lammps_user_mfftorch/docs/ASCEND.md#mace-mh-1-scaling).

Validation covers energies, forces, force-loss parameter gradients and position
gradients against CPU FP64 for four model configurations, as well as partial
inputs, empty graphs, channel tails and derivatives through third order.
Full-model checks compare energies, forces and parameters after 70 AdamW updates.
MH-1 conversion is checked for all six heads on periodic Cu and mixed H/C/O.
LAMMPS checks cover global virial, both Newton settings, changing ownership,
empty ranks, triclinic cells and large coordinate offsets. Core model checks use
relative L2 `1e-4` and componentwise `atol=2e-5`, `rtol=1e-4`; small-system
MH-1 deployment uses those componentwise tolerances and the corresponding
combined norm budget against original MACE in CPU FP64. Full-size MD consistency
checks compare energy and virial per atom and compose the two FP32 reference
budgets for both componentwise and norm checks. Selected large-system force
components are also checked with original-MACE FP64 central differences.

</details>

## License

Project-owned software and documentation are licensed under [MIT](../LICENSE).
Third-party software, model weights and datasets retain their upstream licenses;
see [LICENSE_SCOPE.md](../LICENSE_SCOPE.md).
