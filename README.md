# ICTC-O3-HP-Ascend

High-performance ICTC-O3 operators optimized for Ascend NPUs, together with the
MACE-ICTC framework for atomistic modeling, energy-and-force training and
molecular dynamics.

Version 0.1.0 · [MIT License](LICENSE) · Distribution: `ictc-o3-hp-ascend`

Start with the [user guide](docs/USAGE.md) for installation, training on your
own data, MACE checkpoint conversion, structure prediction and LAMMPS deployment.
Copyable recipes are in [`examples/`](examples/). Standalone operators and models
are installed together as `ictc_o3` and `mace_ictc`.

## Highlights

- Full MACE training: up to 4.76× the throughput of original PyTorch MACE/e3nn,
  including energy, forces, force-loss second derivatives and AdamW.
- Energy-and-force inference: up to 4.48× faster than original MACE/e3nn;
  C64/lmax 2 reaches 54.8k atom-steps/s at 16,384 atoms.
- Standalone O(3) tensor products: up to 3.90× forward and 3.85× forward-plus-VJP
  throughput over e3nn. The MindChemistry PyNative comparison reaches 23.18×
  forward and 18.37× forward plus gradients at C64/lmax 2.
- Large systems: C64/lmax-2 capacity inference completes 65,536 atoms;
  performance-mode training completes 16,384 atoms on one 910B3.
- Multi-NPU MD: 6.64× speedup and 83.0% efficiency on eight NPUs,
  reaching 248,559 atom-steps/s for a 32,768-atom complete LAMMPS step.
- Pretrained MACE-MH-1: all six heads and nonlinear model components retained;
  release measurements show 2.05× faster energy-and-force inference at 4,000 atoms.

The native backend combines Ascend C kernels, hardware-derived Unified Buffer
(UB) tiling and channel-contiguous CANN GEMMs. Models support changing atom
counts and neighbor graphs, configurable channels/degrees, and correlation
orders 1–3. Execution uses eager FP32 with HF32 disabled.

![Single-Ascend C64/lmax-2 training and energy-and-force inference throughput and peak allocated memory.](docs/figures/ascend-training-inference.svg)

## Installation

Use a Linux ARM64 environment with matching Ascend drivers, CANN and
PyTorch/`torch_npu`. The qualified stack is CANN 9.0.0, Python 3.10,
PyTorch 2.7.1, `torch_npu` 2.7.1.post4 and Triton-Ascend 3.2.1.
After sourcing the CANN environment, run inside a compute allocation:

```bash
git clone https://github.com/Parity-LRX/ICTC-O3-HP-Ascend.git
cd ICTC-O3-HP-Ascend
python -m pip install -e '.[hp,train,preprocess,conversion]' 'cmake>=3.16' \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
python -m mace_ictc.ascendc.build --output /path/to/build/ictc_ascendc --jobs 4
export ICTC_ASCENDC_LIBRARY=/path/to/build/ictc_ascendc/libictc_ascendc.so
```

Install the platform's matching PyTorch/`torch_npu` wheels first. Triton-Ascend
supplies the `triton` namespace. Rebuild the native library after updating its
sources or the device software stack. On Slurm clusters, use `sbatch`; the
[user guide](docs/USAGE.md#1-install-and-build) includes a batch template.

## Train, convert and predict

The following commands run inside an allocation. The complete
[training recipe](examples/train_hco.sh) reads atomic energy offsets from the
preprocessor's `fitted_E0.csv` and explicitly enables native Ascend C operators:

```bash
ictc-ascend-preprocess \
  --input-file data/reference.extxyz --output-dir data/hco \
  --elements H C O --atomic-energy-keys 1 6 8 \
  --energy-key REF_energy --force-key REF_forces --max-radius 4.5
bash examples/train_hco.sh data/hco models/hco.pth
python examples/predict.py \
  --checkpoint models/hco.pth --structure data/reference.extxyz \
  --out prediction.json --mode performance
```

Train an ordinary short-range potential on one NPU. For an existing MACE model,
convert its checkpoint instead:

```bash
# Supported ordinary ScaleShiftMACE model.
ictc-ascend-convert-mace \
  --mace-model models/source.model --dtype float32 --out models/converted.pth

# MACE-MH-1: all heads retained; omat_pbe selected for evaluation.
ictc-ascend-convert-foundation \
  --mace-model models/mace-mh-1.model --head omat_pbe \
  --dtype float32 --out models/mh1-ictc.pt
python examples/foundation.py \
  --checkpoint models/mh1-ictc.pt --structure data/cu.extxyz --head omat_pbe
```

The [user guide](docs/USAGE.md) covers data labels, E0, configurable model sizes,
restart/evaluation, standard-model fine-tuning and performance/capacity modes.
The MH-1 foundation adapter supports frozen inference.

## Python API

```text
ictc_o3/       # Standalone O(3) operators and Ascend C entry points
mace_ictc/     # Models, training, conversion and deployment
```

### Standalone O(3) operators

Use these layers in your own equivariant network without constructing a MACE
model. The [standalone O(3) API](docs/O3_API.md) covers tensor products, graph
messages, symmetric contractions, direction harmonics, channel linear layers
and scalar activations, with tensor layouts and gradient examples.
Both namespaces ship in one distribution and share implementation modules and
native kernels.

```python
import torch_npu
from ictc_o3 import EdgeTensorProduct, PackedGraphMessage, create_symmetric_contraction
from ictc_o3.ascendc import enable

enable(mode='performance')  # Reads ICTC_ASCENDC_LIBRARY.
tp = EdgeTensorProduct.create(channels=64, input_lmax=2, edge_lmax=2)
# output = tp(features, harmonics, gates)
```

Build the shared native library with `python -m ictc_o3.ascendc.build` using
the [installation options](docs/O3_API.md#1-install-and-enable-the-high-performance-backend).
Run [`examples/operators.py`](examples/operators.py) for an edge TP, or
[`examples/o3_layers.py`](examples/o3_layers.py) for graph messages, channel
mixing, symmetric contraction and a mixed second derivative training step.

### MACE-ICTC models

```python
import torch_npu
from mace_ictc.ascendc import enable
from mace_ictc.hp import create_model, configure_model, energy_and_forces

enable(mode='performance')  # 'capacity' selects chunked frozen inference.
model = create_model(channels=64, lmax=2, num_interactions=2, correlation=3)
configure_model(model, training=True, policy='generated')

# graph: positions [N,3], atomic_numbers [N], batch [N],
# edge_src/edge_dst [E], lattice edge_shifts [E,3], cell [B,3,3].
prediction = energy_and_forces(model, **graph, training=True)
# Build the energy-and-force loss, backward, then update the optimizer.

model.eval().requires_grad_(False)
configure_model(model, training=False, policy='generated')
prediction = energy_and_forces(model, **graph)
```

The model API returns interaction energies; prediction examples restore
checkpoint atomic reference energies exactly once. Forces require autograd,
including during frozen inference. Run [`examples/model.py`](examples/model.py)
with `--ascendc` for a complete synthetic training step.

See the [model and backend API](docs/HP_API.md) for model configuration,
execution modes and hardware scheduling.

## LAMMPS deployment

The plain `mff/torch` pair style keeps a persistent NPU model per MPI rank.
HCCL exchanges intermediate features, their adjoints and ghost-force buffers;
LAMMPS handles CPU integration, neighbors and atom migration.

```lammps
units metal
atom_style atomic
read_data system.data
pair_style mff/torch 4.5 npu
pair_coeff * * models/hco.pth H C O
```

Training checkpoints and converted foundation archives load directly.
Use one MPI rank per NPU, with the checkpoint cutoff and elements in atom-type
order. The [user guide](docs/USAGE.md#9-build-and-run-lammps) provides compilation,
a complete input and an eight-NPU launch; the
[deployment guide](lammps_user_mfftorch/docs/ASCEND.md) explains HCCL routing,
global virial and diagnostics. This path uses neither Kokkos nor ML-IAP.

## Performance on Ascend

All results use Ascend 910B3 and FP32 with HF32 disabled. Performance mode
prioritizes throughput; capacity inference uses chunk recomputation to bound
intermediate storage. Chunk sizes adapt to tensor widths and available device
memory. Atom counts and neighbor graphs remain dynamic; `torch.compile` is disabled.

### O(3) operators

![Standalone tensor-product throughput: current HP and e3nn on one Ascend 910B3.](docs/figures/ascend-tp-throughput.svg)

At the largest shared sizes in each of four configurations, HP delivers
1.87–3.90× e3nn's forward throughput and 2.19–3.85× its forward-plus-VJP
throughput. C128/lmax 3 reaches 2.70 million TP instances/s at 262,144
instances, 3.90× e3nn. VJP includes gradients of all three inputs. The
horizontal axis counts independent TP instances.

<details>
<summary>MindChemistry comparison</summary>

![O(3) tensor-product throughput against MindChemistry on Ascend.](docs/figures/ascend_mindchemistry_tp.svg)

The v0.1.0 MindChemistry comparison covers C64/C128, angular degrees 1–3,
and both MindSpore PyNative and Graph execution. At C64/lmax 2 and 65,536
edges, an isolated repeat measures 12.20 ms versus 282.92 ms for forward,
and 45.12 ms versus 828.93 ms for forward plus all-input gradients:
23.18× and 18.37× speedups over MindChemistry PyNative.

</details>

### MACE training, inference and memory

C64/lmax 2 reaches 54,823 atom-steps/s for energy-and-force inference at
16,384 atoms, 3.62× original MACE/e3nn. Full training reaches 18,486
atom-steps/s at 8,192 atoms, 3.67× the baseline. Training includes energy,
forces, the energy-and-force loss, mixed second derivatives and AdamW.

Performance mode completes 32,768-atom C64/lmax-2 inference and
16,384-atom training. Capacity mode completes 65,536-atom inference at
27,678 atom-steps/s. Its curves use the updated chunk-recomputation implementation.

<details>
<summary>Full model scan: throughput and memory</summary>

![Training and energy-and-force inference throughput across four MACE configurations.](docs/figures/ascend-mace-throughput.svg)

The scan covers C64/lmax 2, C128/lmax 2, C128/lmax 3 and C256/lmax 3,
with two residual interactions and correlation 3. At C128/lmax 3 and
2,048 atoms, HP performance mode is 4.48× faster for inference and
4.76× faster for full training than original MACE/e3nn.

![Peak allocated memory versus atom count across four MACE configurations.](docs/figures/ascend-mace-memory.svg)

Curves connect measured sizes and end at their last successful sample.
Crosses in the main memory panels mark observed e3nn and performance-mode
OOM sizes. The updated capacity curves confirm successful sizes;
capacity-mode OOM boundaries have not been remeasured.

</details>

### Pretrained MACE-MH-1

![Pretrained MACE-MH-1 energy-and-force inference throughput and peak allocated memory.](docs/figures/ascend-mh1-throughput.svg)

Official MACE-MH-1 retains all six heads. The v0.1.0 measurements reach
7,440 atom-steps/s at 4,000 atoms in performance mode, a 2.05× speedup over
original PyTorch MACE. The release capacity implementation reaches 4,265
atom-steps/s and uses 6.13 GiB versus 31.71 GiB for original MACE.
Both HP modes complete the 6,912-atom sample; original MACE reaches OOM there.

### Multi-NPU molecular dynamics

![Complete LAMMPS step scaling on Ascend](docs/figures/ascend_lammps_scaling.svg)

At 32,768 atoms, the C64/lmax-2/two-interaction/correlation-3 model reaches
6.64× speedup on eight NPUs: 83.0% efficiency, 131.83 ms per complete MD step
and 248,559 atom-steps/s. Timing includes CPU integration, neighbor work,
transfers, energy, forces and communication. HCCL exchanges intermediate
features, their adjoints and ghost-force contributions between NPUs.

<details>
<summary>Benchmark setup and numerical checks</summary>

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

MACE-MH-1 measurements retain the v0.1.0 protocol: the same official checkpoint,
the `omat_pbe` head, a 6 Å cutoff and 78 neighbors/atom; median of five
synchronized calls after two warmups. These pretrained measurements are
separate from the four randomly initialized models above.

Multi-NPU results use complete dynamic LAMMPS runs on one eight-NPU node,
with 1/2/4/8 active NPUs, a 4.5 Å cutoff, 0.5 Å skin and 0.125 atoms/Å³.
There are initially 56 physical neighbors and 80 cached skin edges per atom.
Points are median complete loop times from three ten-step runs after five
warmup steps. Compact owned outputs and native gathers are enabled. The
[deployment guide](lammps_user_mfftorch/docs/ASCEND.md#measured-md-performance)
provides the complete latency, throughput and memory measurements.

Validation covers energies, forces, force-loss parameter gradients and position
gradients against CPU FP64 for four model configurations, as well as partial
inputs, empty graphs, channel tails and derivatives through third order.
Full-model checks compare energies, forces and parameters after 70 AdamW updates.
MH-1 conversion is checked for all six heads on periodic Cu and mixed H/C/O.
LAMMPS checks cover global virial, both Newton settings, changing ownership,
empty ranks, triclinic cells and large coordinate offsets. Core model checks use
relative L2 `1e-4` and componentwise `atol=2e-5`, `rtol=1e-4`; MH-1 deployment
uses those componentwise tolerances and the corresponding combined norm budget.

</details>

## License

ICTC-O3-HP-Ascend project-owned code and documentation are licensed under the
[MIT License](LICENSE). Third-party source notices, dependencies, pretrained
weights and datasets retain their original terms. See
[LICENSE_SCOPE.md](LICENSE_SCOPE.md), [NOTICE](NOTICE) and [UPSTREAM.md](UPSTREAM.md).
