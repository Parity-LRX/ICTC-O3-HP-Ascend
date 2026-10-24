# ICTC-O3-HP-Ascend

ICTC-O3 Cartesian tensor operators and the MACE-ICTC framework for Ascend NPUs.
Tensor products use shared geometry projections and FP32 batched GEMMs.
Symmetric contractions combine shared CG prefixes, bounded GEMM scratch storage
and analytic adjoints. Triton-Ascend supplies sparse fallbacks and generated
direction harmonics.

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

## Standalone O3 operators

```python
import torch
import torch_npu
from ictc_o3 import EdgeTensorProduct, direction_harmonics

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
- Matrix contraction schedules bound scratch storage and retain a sparse fallback
  when constant or species-bank matrices exceed the configured memory limits.
- Learned weight names and shapes are retained; configuration does not freeze or cast parameters.
- Tensor-product derivatives use PyTorch NPU matrix operations. Symmetric contractions
  have analytic first and mixed second adjoints; higher derivatives retain
  differentiable PyTorch formulas.
- Channels, element count, interaction depth and graph size are runtime parameters.
  Native symmetric contractions currently cover correlation orders 1, 2 and 3.
- Edge chunking bounds forward intermediates. PyTorch gathers and destination reductions
  remain part of the current graph path.
- FP32 execution disables HF32 and requires autocast to be off. No reduced-precision
  approximation is used.

The current execution route is eager on one NPU. Multi-NPU HCCL, compiled export,
long-range electrostatics and Ascend LAMMPS deployment are not qualified in this
release. Inherited CUDA sources remain available for compatibility and reference;
their presence does not establish NPU support.

## Numerical validation

On an Ascend 910B3, four initialized model configurations cover angular degrees
1–3, correlation orders 1–3 and one to three interactions. Energies, forces and
the parameter and position gradients of the energy-and-force loss pass CPU FP64 comparisons;
the maximum observed relative L2 error is `2.23e-6`.

Validation also covers 28 sparse-operator checks, both packed layouts, chunked
masked graph messages, empty graphs, mutable species metadata, rectangular
channel transforms for multiple atomic species and an energy-and-force trainer
step with AdamW. These checks
use synthetic inputs and do not establish the accuracy of a fitted potential.

## Training and inference performance

One active Ascend 910B3, FP32 with HF32 disabled, eager execution. Each model has
two interactions and correlation order 3. The following points use 4,096 atoms
and 56 directed neighbors per atom, with identical initialized weights and graphs
for Ascend HP and the inherited PyTorch NPU reference.

| Model | Operation | Reference → HP (ms/step) | Speedup | HP atom-steps/s |
| --- | --- | ---: | ---: | ---: |
| C32, lmax 1 | Energy + forces | 748.4 → 127.4 | 5.87× | 32,146 |
| C32, lmax 1 | Energy + force loss + AdamW | 1,486.1 → 308.4 | 4.82× | 13,280 |
| C64, lmax 2 | Energy + forces | 3,271.2 → 332.6 | 9.84× | 12,316 |
| C64, lmax 2 | Energy + force loss + AdamW | 6,883.6 → 832.7 | 8.27× | 4,919 |

For C64/lmax 2, the incremental allocated NPU memory peak changes from
23,936 to 8,191 MiB for inference and from 43,577 to 16,059 MiB for training.
This is the PyTorch allocation peak above the warmed baseline; it excludes
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
