# Ascend operator and model API

Import standalone operators from `ictc_o3` and model entry points from
`mace_ictc.hp`. Import `torch_npu` before selecting an NPU device.
All native operators require eager FP32 with autocast disabled. Configuration
sets CANN precision to `must_keep_origin_dtype` and disables HF32 globally.
These process settings are not automatically restored.

## Explicit Ascend C backend

Build inside an NPU compute allocation, with the CANN environment sourced and
a host C++ compiler and CMake available:

```bash
python -m mace_ictc.ascendc.build --output /path/to/build/ictc_ascendc --jobs 4
export ICTC_ASCENDC_LIBRARY=/path/to/build/ictc_ascendc/libictc_ascendc.so
```

Use the exact library path printed by the builder. The binary depends on the
local CANN, PyTorch, torch_npu, C++ ABI and target SoC; rebuild when these change.
Importing the package never starts compilation.

```python
import torch_npu
from mace_ictc.ascendc import enable

ops = enable(mode='performance')  # reads ICTC_ASCENDC_LIBRARY
ub_bytes, vector_cores = ops.hardware()
# Construct/configure the model or standalone operators after enable().
```

Rebuild the Ascend C library after updating the source tree. Native transfer
policies, matrix adjoints and queue descriptors require matching library sources.

This process-wide opt-in selects an execution mode:

| Mode | Tensor products and graph messages | Symmetric contractions | Dense transforms |
| --- | --- | --- | --- |
| `performance` (default) | Channel-contiguous grouped GEMMs; Vector gate/reduction epilogues for large working sets | Ascend C vector kernels | CANN GEMMs with analytic adjoints |
| `capacity` | Fused Ascend C gather, tensor product and atomic accumulation | Ascend C vector kernels | CANN GEMMs with analytic adjoints |
| `native` | Fused Ascend C graph kernels | Ascend C vector kernels | Ascend C FP32 Cube kernels |

All three modes use native direction polynomials and nonlinearities. Choose a
mode before constructing a forward graph. Explicit `enable(path, mode=...)`
also accepts a library path. Model configuration replaces standard Linear and SiLU
modules with compatible dispatch modules while retaining the exact Parameter
objects and state_dict names. Ordinary framework layout operations, loss
construction, small broadcast products and some reductions still run through
PyTorch NPU. Configuration without `enable()` retains the Triton/GEMM backend.
The full native mode is available for inspection and tuning; rewriting a GEMM
does not imply higher throughput than the vendor implementation.

The native Cube implementation specializes every transpose combination so that
the Host tiling, MatmulType and runtime flags match for FP32. It also enables
the SDK's cyclic ND-to-NZ transfer path for source row strides above 65535
elements, and explicitly disables HF32. Bias uses a separate FP32 vector kernel.

### UB tiling

The planner queries the actual Unified Buffer capacity and Vector Core count.
Tiles span rows and channels; tail channels and rows use separate index tables.
The byte budget includes both input/output queue slots, intermediate products,
three work vectors, byte-offset gather indices, masks, buffered graph indices,
coupling metadata and a reserved margin. The C++ launcher independently checks
the same allocation bound before launching a kernel. Large coupling tables are
streamed in bounded chunks. Global addresses use 64-bit offsets.

Optional process environment overrides for controlled tuning:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `ICTC_ASCENDC_TILE_CHANNELS` | `0` | Automatic channel tile, at most 64; a positive value requests a tile |
| `ICTC_ASCENDC_TILE_ROWS` | `0` | Largest row tile within the budget; positive requests are checked |
| `ICTC_ASCENDC_UB_RESERVE` | Automatic | One sixth of UB rounded up to 4 KiB; at least 4096 bytes |

The automatic schedule uses the actual operator widths, channel count, metadata
and device limits. It does not fix model hyperparameters. Invalid overrides
raise an error instead of launching a tile that exceeds UB or transfer limits.
All queue storage remains FP32; tiling does not introduce reduced precision.

## Tensor products

```python
from ictc_o3 import EdgeTensorProduct

op = EdgeTensorProduct.create(
    channels=64, input_lmax=2, edge_lmax=3, output_lmax=2,
    layout='ir_mul', device='npu',
)
y = op(features, harmonics, gates)
blocks = op.unpack_output(y)
```

| Argument | Shape and convention |
| --- | --- |
| `features` | `[E,C*(L_in+1)^2]`, degrees concatenated in increasing order |
| `harmonics` | `[E,(L_edge+1)^2]`, original component-normalized ICTC basis |
| `gates` | `[E,P*C]` or `[E,P,C]`; path order is `op.paths` |
| `y` | `[E,op.output_width]`, output degrees and path multiplicities retained |
| `blocks[l]` | `[E,C*path_count_l,2*l+1]` |

`ir_mul` stores each angular block as `[angular,channel]`; `mul_ir` uses
`[channel,angular]`. `op.pack_input({l: block})` accepts `[E,C,2*l+1]` blocks.
An existing path-preserving ICTC tensor-product module can be wrapped with
`EdgeTensorProduct(tp, input_lmax=..., edge_lmax=...)`. Its coefficients and
trainability are preserved. A new factory module has fixed unit path coefficients;
external gates remain live.

With Ascend C enabled in `performance` mode, standalone `EdgeTensorProduct`
groups coupling paths by input angular degree and uses channel-contiguous
batched GEMMs. Each path writes directly into one packed output buffer.
The `ir_mul` input blocks retain their batch strides without an additional
feature copy. Path-weight adjoints recompute the required GEMM bases instead
of retaining all intermediate path outputs. Analytic matrix adjoints remain
differentiable through higher derivatives. Both layouts, independent input,
edge and output degrees, live coefficients, and empty batches are supported.
Capacity mode retains the native sparse execution path.

## Graph messages

`PackedGraphMessage` has the same factory and packed buffers as `EdgeTensorProduct`.
Its call is `op(features, harmonics, gates, src, dst, mask=None)`, with features
on `N` nodes and all remaining floating inputs on `E` edges. `src` and `dst`
are NPU int64 vectors. The optional fixed mask has shape `[E]` or `[E,1]`.
Output contains `N` destination rows.

`GraphMessage` accepts `{l: [N,C,2*l+1]}` features and
`{l: [E,1,2*l+1]}` harmonics, and returns destination blocks.
The `edge_chunk_size` constructor option controls edge intermediate size.
Performance mode uses differentiable PyTorch `index_add`. Capacity and native
modes gather sources and accumulate destinations inside the tensor-product
kernel with FP32 atomics, including fixed masks and analytic adjoints. Repeated
destination indices are supported; floating-point summation order may differ.
Trainable masks retain the differentiable unfused path.

## Direction harmonics

`direction_harmonics(unit_directions, lmax, policy='generated')` returns a list
of `[E,2*l+1]` tensors in increasing degree order. Directions have shape `[E,3]`;
the caller normalizes displacements. `policy='reference'` selects the original
PyTorch polynomials. First and second direction derivatives have native kernels;
higher derivatives retain the polynomial formula.

## Symmetric contraction

```python
from ictc_o3 import create_symmetric_contraction

sc = create_symmetric_contraction(
    num_elements=3, channels=64, lmax=2, target_lmax=2,
    correlation=3, device='npu', training=True,
)
y = sc(node_features, node_attributes, node_type_idx=species)
```

`node_features` has shape `[N,C*(L+1)^2]`; `node_attributes` is the original
`[N,S]` one-hot interface; `species` is the corresponding fixed integer `[N]`
map. Output has shape `[N,C*(L_target+1)^2]`.
`configure_symmetric_contraction(existing_sc, policy='generated')` preserves
learned weights and checkpoint keys. Orders 1, 2 and 3 are supported. Constants
must reconstruct within the original factorization precision budget; a failed
reconstruction raises an error. The shared-prefix circuit uses a real spherical
basis internally with ICTC basis maps at its boundaries.
The Ascend schedule evaluates the shared-prefix circuit with FP32 matrix products
and analytic adjoints, recomputing temporary outer products in backward. Temporary
outer products are chunked; learned species banks remain independent of node count.
Large constant or species-bank matrices retain the sparse AI Core schedule.
When Ascend C is enabled, the shared-prefix products use UB-tiled vector maps;
native mode also executes the fixed basis and learned bank projections on Cube.

## Channel operations

Call `from ictc_o3 import strict_fp32; strict_fp32()` before using these functions
independently. Tensor-product factories, symmetric-contraction configuration and
whole-model configuration already apply these process-wide precision settings.

`channel_linear(features, weight, bias=None)` maps `[...,I,M]` and `[O,I]`
to `[...,O,M]`. `species_channel_linear(features, weight, species, bias=None)`
uses `[N,I,M]`, `[S,O,I]` and integer `[N]` species indices. Bias is allowed
only for scalar angular blocks (`M=1`). Both preserve learned weight and
higher derivatives. The species linear path evaluates shared GEMMs with device
selection; it avoids per-node matrix expansion, and its arithmetic grows with
species count. With Ascend C enabled, `normalized_silu(x, scale=1.)` uses its
FP32 vector kernel and analytic first and mixed second adjoints.

## Whole models

`create_model(channels=64, lmax=3, edge_lmax=None, num_interactions=2,
correlation=3, atomic_numbers=(1,6,8), cutoff=5., device='npu', dtype=torch.float32)`
creates an initialized bridge-U model. No architecture values are fixed by the
NPU kernels. Correlation is currently limited to 1/2/3 by the model factory.

`check_model(model, training=False)` is read-only. `configure_model(model,
training=False, policy='generated', message_chunk_edges=None)` selects the
Ascend backend for NPU models. Policies are `auto`, `generated` and `reference`.
The returned dictionary identifies the backend, selected route and coverage
reasons. The public runtime exposes Ascend scheduling options only.

The default `message_chunk_edges=None` estimates a per-interaction chunk from
its actual channel, angular and path widths, with a 3 GiB forward-scratch budget
and a maximum of 131,072 edges. This estimate excludes saved activations and other
model storage; it does not guarantee a total memory limit. Set a positive integer,
for example `message_chunk_edges=16384`, to limit edge intermediates more tightly.
Standalone graph-message factories retain their explicit 16,384-edge default.
The grouped performance route is also selected for wide/angular messages whose
active edge tile times channels times output angular width reaches 64 Mi elements.
Consequently a small explicit edge chunk can still use grouped GEMMs when the
actual model width amortizes their launch overhead.

For graphs with at least 65,536 edges and chunks of at least 32,768 edges,
or messages meeting the volume criterion above, performance graph messages keep
channels contiguous and group paths by their actual input degree. Each group
uses one GEMM without zero padding. Recursive
matrix adjoints support force inference and energy-and-force training. Smaller
workloads retain the established path schedule to avoid extra launch overhead.
Grouped path splits assemble bond and gate gradients without one full slice-adjoint buffer
per path. Shared edge-chunk splits also join gate and harmonic cotangents without
allocating a full-edge slice gradient for each chunk. Standalone edge products
retain their established analytic adjoints.
Large working sets also fuse path gates and destination accumulation on Vector
Cores. These messages keep `[edges, angular rows, channels]` contiguous through
the first and mixed second adjoints. Constant geometry columns are arranged to
merge adjacent output intervals into fewer DMA transfers; learned path weights
and the public output order are preserved.
First adjoints share node cotangent reads. Mixed second adjoints combine the
requested edge gradients and the two node-gradient terms in one Vector kernel,
reducing repeated transfers and destination updates. Their UB schedule includes
all active operands, all requested outputs and the selected input/output queue slots. Tiles that
exceed this budget retain separate native adjoints.
Input-degree groups accumulate compact `[nodes, group angular rows, channels]`
buffers across all edge chunks and restore the public angular/path order once
at node count. Native epilogues can reuse each group accumulator instead of
allocating and adding another full-node tensor per chunk. The accumulator is
private to one forward call; its autograd chain retains first, mixed second
and higher derivatives. Coupling coefficients and learned parameter tensors
are unchanged. Native narrow-matrix
adjoints keep their original matrix storage, including transposed operands;
unsupported widths retain CANN execution.

Tiles covering all channels combine consecutive angular rows into single DMA blocks
when both source and destination intervals are contiguous. Channel tails and
row gaps retain strided transfers. This changes transfer segmentation while
preserving FP32 arithmetic and differentiation formulas. Set
`ICTC_ASCEND_CONTIGUOUS_DMA=0` to retain the preceding transfer schedule.

Periodic geometry uses single-axis row gathers. A single-cell batch computes
periodic shifts as one matrix product; multiple cells use the corresponding
batched matrices. Positions, cell tensors and shifts remain differentiable.
Only fixed species, shape and mathematical permutation metadata may be cached.
Learned weights and feature values are read on every call.
Set `ICTC_ASCEND_FUSED_ADJOINTS=0` to retain the native gate/reduction epilogue
with separate adjoints; the default is `auto`.
The automatic policy uses the actual active angular width and channel/edge
working set, then checks the complete UB and DMA limits. Small messages and
trainable masks retain PyTorch postprocessing. Set `ICTC_ASCEND_EPILOGUE=0` before
execution to retain grouped GEMMs with PyTorch gates and accumulation.
Switch configurations when switching workloads: call
`configure_model(..., training=True)` for training and
`configure_model(..., training=False)` for inference. The adjoint policy and
chunk policy are reported as `tensor_product_adjoint` and `edge_chunk_size`.
The performance adjoint policy is `automatic`; graph width, effective chunk size
and UB feasibility determine the graph schedule at execution.
Capacity and native modes use analytic tensor-product adjoints for both workloads.
The return value also reports `kernel_backend` and `ascendc_mode`.
Standalone tensor products use the analytic route, including mixed second adjoints
and a differentiable fallback for higher derivatives. Learned coefficients and
external gates remain live inputs in both routes.

The default NPU operator task queue is retained. Performance-mode native
products, matrix blocks and graph epilogues inspect live PyTorch allocations
before launching the next operation. At three quarters of total device memory
or above, they finish the current stream to release eligible queued temporaries.
These checks also apply during first and mixed second adjoints and direct model
calls, while ordinary workloads retain pipeline overlap.

`energy_and_forces` additionally marks performance messages whose estimated
full-edge storage reaches one third of total device memory. It finishes marked
streams before returning energy and forces, for inference and force-loss
training. The estimate uses the same channel/angular/path formula as automatic
chunk sizing. External loss backward and optimizer updates follow this call as
usual. Actual peaks also depend on saved autograd graphs and external runtime
storage.

Runtime stream synchronization is included in reported latency. Capacity/native
messages use their existing schedules. Set `ICTC_ASCEND_BOUND_GRAPH_QUEUE=0` to
disable these performance-mode queue bounds. The runtime does not overwrite
`TASK_QUEUE_ENABLE`.

`energy_and_forces(model, positions, atomic_numbers, batch, edge_src, edge_dst,
edge_shifts, cell, training=False)` returns `energy [B]`, `forces [N,3]` and
`atomic_energy [N]`. `edge_shifts` are lattice coordinates and `cell` is
`[B,3,3]`. The graph is supplied by the caller; neighbor construction and atomic
reference-energy offsets are outside this call. Use `training=True` to retain
force-loss derivatives. Inference computes forces with autograd and detaches
results, so do not wrap it in `torch.inference_mode()`.

Existing checkpoints retain `mace_ictc` class paths and parameter names. Load
and rebuild their original architecture first, move the model to NPU FP32,
then call `configure_model`. A CPU float64 checkpoint must have its explicit
internal compute dtype attributes updated consistently when converting to FP32;
configuration itself does not perform that conversion.

## Foundation checkpoint adapter

Use `mace_ictc.cli.convert_foundation` and its `load_model` helper for the
nonlinear MACE-MH-1 architecture. Its `configure_inference` method selects
Ascend ICTC operators while preserving the original six heads, radial transform,
ZBL, density normalization and readouts. This is a frozen inference adapter.
See [CONVERSION.md](CONVERSION.md) for a complete command, user example,
energy-offset convention and LAMMPS deployment.
