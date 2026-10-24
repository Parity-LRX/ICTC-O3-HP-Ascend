# ICTC-O3-HP-Ascend

ICTC-O3 Cartesian tensor operators and the MACE-ICTC framework for Ascend NPUs.
Ascend C kernels cover tensor products, fused graph messages, symmetric
contractions, direction harmonics, nonlinearities and FP32 channel transforms.
Hardware-derived UB tiling supports odd channel counts, tails and streamed
coupling tables. Performance and capacity modes combine these kernels with
CANN GEMMs according to the measured workload.

Version 0.1.0. Distribution: `ictc-o3-hp-ascend`. Python packages: `ictc_o3`
for standalone operators and `mace_ictc` for models, training and checkpoints.
This is an independent port of [ICTC-O3-HP](https://github.com/Parity-LRX/ICTC-O3-HP).

## Install

Use a dedicated Linux environment with a matching Ascend driver, CANN Toolkit,
CANN operators and PyTorch NPU extension. The development stack is Ascend 910B3,
CANN 9.0.0, Python 3.10, PyTorch 2.7.1, torch_npu 2.7.1.post4 and
Triton-Ascend 3.2.1. Source the site's CANN environment before running Python.

```bash
python -m pip install 'torch==2.7.1' 'torch_npu==2.7.1.post4'
python -m pip install 'triton-ascend==3.2.1' pybind11 \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
python -m pip install -e '.[train]'
python -m pip install 'cmake>=3.16' ninja
python -m mace_ictc.ascendc.build --output /path/to/build/ictc_ascendc --jobs 4
export ICTC_ASCENDC_LIBRARY=/path/to/build/ictc_ascendc/libictc_ascendc.so
```

Triton-Ascend supplies the `triton` namespace. Install it after other packages
that may install community Triton, and check that its Ascend backend is present.
Avoid installing another `mace_ictc` checkout into the same environment.
On a cluster, run compilation and all examples in a scheduled compute allocation.
Use node-local temporary storage for compiler scratch when shared-storage
unlink semantics cause compiler cleanup errors.

## Whole-model training and inference

```python
import torch
import torch_npu
from mace_ictc.hp import create_model, configure_model, energy_and_forces
from mace_ictc.ascendc import enable

enable(mode='performance')  # 'capacity' saves memory; 'native' uses custom Cube GEMMs

model = create_model(
    channels=64, lmax=2, num_interactions=2, correlation=3, device='npu',
)
plan = configure_model(model, training=True, policy='generated')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)

# graph contains positions, atomic_numbers, batch, edge_src, edge_dst,
# edge_shifts in lattice coordinates, and cell [B,3,3], all on the NPU.
prediction = energy_and_forces(model, **graph, training=True)
loss = (prediction['energy'] - reference_energy).square().mean() \
     + (prediction['forces'] - reference_forces).square().mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

model.eval().requires_grad_(False)
configure_model(model, policy='generated')
prediction = energy_and_forces(model, **graph)
```

The factory creates initialized weights. It does not provide a trained potential.
`policy='reference'` selects the inherited differentiable PyTorch implementation;
`policy='generated'` requires NPU coverage and reports unsupported configurations.
Run [`examples/model.py`](examples/model.py) for a complete synthetic training step.
Add `--ascendc --ascendc-mode performance` to use the built library. The default
without `enable()` retains the Triton/GEMM implementation.

## Standalone O3 operators

```python
import torch
import torch_npu
from ictc_o3 import EdgeTensorProduct, direction_harmonics
from mace_ictc.ascendc import enable

enable(mode='capacity')

op = EdgeTensorProduct.create(channels=64, input_lmax=2, edge_lmax=2)
features = torch.randn(1024, 64*9, device='npu', requires_grad=True)
directions = torch.nn.functional.normalize(torch.randn(1024, 3, device='npu'), dim=-1)
harmonics = torch.cat(direction_harmonics(directions, 2, policy='generated'), -1)
gates = torch.randn(1024, op.gate_width, device='npu', requires_grad=True)
output = op(features, harmonics, gates)
```

See [the API guide](docs/HP_API.md) for buffer layouts, graph messages,
species banks and symmetric contractions, and [the upstream provenance](UPSTREAM.md).

## Backend design and scope

- Tensor products share geometry projections across channels, consume the original
  ICTC coupling constants and preserve path order.
- Symmetric contractions share CG prefixes and read species weights directly, without
  materializing a node-sized path-weight bank or sorting nodes by species.
- Native vector schedules account for double buffers, work vectors, gather and
  graph indices, masks and coupling metadata before selecting row/channel tiles.
  The tested 910B3 reports 196,352 available UB bytes and 40 Vector Cores. The
  automatic reserve is one sixth of reported UB, rounded up to 4 KiB. These are
  scheduling choices, with no fixed model hyperparameters.
- Learned weight names and shapes are retained; configuration does not freeze or cast parameters.
- Performance mode uses analytic tensor-product adjoints that preserve contiguous
  channel blocks. Training retains ordinary per-path GEMM autograd, which is faster
  for the measured force-loss workload. Standalone tensor products and symmetric
  contractions have analytic first and mixed second adjoints; higher derivatives
  retain differentiable PyTorch formulas.
- Capacity mode fuses source gathering, tensor products and FP32 atomic
  destination accumulation, with analytic first and mixed second adjoints.
  Native mode additionally selects custom strict-FP32 Cube matrix kernels.
  Both transpose flags and wide ND transfer strides are handled explicitly.
- Channels, element count, interaction depth and graph size are runtime parameters.
  Native symmetric contractions currently cover correlation orders 1, 2 and 3.
- Whole-model edge chunks are sized from the actual channel, angular and path widths.
  Set `message_chunk_edges=16384` for smaller forward intermediates; automatic sizing
  favors throughput and can increase the allocation peak. See the API guide for
  the scratch estimate. Performance mode retains PyTorch graph accumulation;
  capacity/native modes perform it inside Ascend C kernels.
- FP32 execution disables HF32 and requires autocast to be off. No reduced-precision
  approximation is used.

Ordinary framework layout operations, loss construction, small broadcast
products and some reductions still use PyTorch NPU. A full native execution
option is provided for tuning; it is slower than the selected vendor-GEMM
combination at the measured points.

The current execution route is eager on one NPU. Multi-NPU HCCL, compiled export,
long-range electrostatics and Ascend LAMMPS deployment are not qualified in this
release. Inherited CUDA sources remain available for compatibility and reference;
their presence does not establish NPU support.

## Numerical validation

On an Ascend 910B3, four initialized model configurations cover angular degrees
1–3, correlation orders 1–3 and one to three interactions. Energies, forces and
the parameter and position gradients of the energy-and-force loss pass CPU FP64 comparisons;
the maximum observed relative L2 error across all three modes is `2.31e-6`.

Validation also covers 28 sparse-operator checks, both packed layouts, chunked
masked graph messages, empty graphs, mutable species metadata, rectangular
channel transforms for multiple atomic species and an energy-and-force trainer
step with AdamW. Native Cube checks include every transpose/bias combination,
frozen operands and mixed derivatives, including a reduction width of 65,537.
Nonlinearity and direction-polynomial checks include third derivatives. These checks
use synthetic inputs and do not establish the accuracy of a fitted potential.

## Training and inference performance

One active Ascend 910B3, FP32 with HF32 disabled, eager execution. Each model has
two interactions and correlation order 3. The following points use 4,096 atoms
and 56 directed neighbors per atom, with identical initialized weights and graphs
for every route. All latencies include both energy and forces.

| Model | Operation | Performance ms / atom-steps/s | Capacity ms / atom-steps/s | Allocated peak, performance / capacity (MiB) |
| --- | --- | ---: | ---: | ---: |
| C32, lmax 1 | Inference | 69.6 / 58,878 | 92.9 / 44,113 | 1,771 / 776 |
| C32, lmax 1 | Training + AdamW | 160.8 / 25,473 | 292.1 / 14,025 | 3,621 / 2,007 |
| C64, lmax 2 | Inference | 206.3 / 19,850 | 313.9 / 13,047 | 9,496 / 2,330 |
| C64, lmax 2 | Training + AdamW | 634.0 / 6,460 | 921.7 / 4,444 | 21,901 / 4,613 |

The comparison baseline is the previous optimized Ascend HP implementation,
using automatic edge chunks, structured inference adjoints and GEMM training.
Performance mode reduces C64 inference latency from 230.1 to 206.3 ms and
training from 740.0 to 634.0 ms, increasing training throughput by 16.7%.
Its C64 training allocation peak increases from 16,032 to 21,901 MiB.
Capacity mode reduces C64 allocation peaks by 76.3% for inference and 71.2%
for training versus the previous HP route, with higher latency.

Full native mode measures 337.2/1,053.5 ms for C64 inference/training, with
2,330/4,611 MiB peaks. It is an explicit alternative, not the performance default.
Memory values are PyTorch allocation peaks above the warmed baseline; they exclude
driver memory and external runtime allocations and is not an OOM-capacity test.

Measurements are medians of three alternating paired rounds, three calls per
round, after three warmup calls. Inference freezes model weights and computes
both energy and forces. Training includes force-loss second derivatives, AdamW
and `zero_grad`. Timing excludes JIT compilation, graph construction, neighbor
rebuilding and host transfers. The graph is a periodic perturbed lattice at
0.125 atoms/Å³ with a 4.5 Å cutoff. The node is allocated exclusively, and the
model runs on one NPU. Every point passes energy, force, parameter-gradient and
position-gradient checks before timing.

## License

The original project's PolyForm Noncommercial and preserved MIT scopes are retained.
See [LICENSE](LICENSE), [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [NOTICE](NOTICE).
