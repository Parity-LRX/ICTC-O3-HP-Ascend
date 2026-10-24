# Ascend operator and model API

Import standalone operators from `ictc_o3` and model entry points from
`mace_ictc.hp`. Import `torch_npu` before selecting an NPU device.
All native operators require eager FP32 with autocast disabled. Configuration
sets CANN precision to `must_keep_origin_dtype` and disables HF32 globally.
These process settings are not automatically restored.

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

## Graph messages

`PackedGraphMessage` has the same factory and packed buffers as `EdgeTensorProduct`.
Its call is `op(features, harmonics, gates, src, dst, mask=None)`, with features
on `N` nodes and all remaining floating inputs on `E` edges. `src` and `dst`
are NPU int64 vectors. The optional fixed mask has shape `[E]` or `[E,1]`.
Output contains `N` destination rows.

`GraphMessage` accepts `{l: [N,C,2*l+1]}` features and
`{l: [E,1,2*l+1]}` harmonics, and returns destination blocks.
The `edge_chunk_size` constructor option controls edge intermediate size.
Destination accumulation uses differentiable PyTorch `index_add`; it is not
currently fused into the tensor-product AI Core kernel.

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
species count. `normalized_silu(x, scale=1.)` uses the NPU PyTorch SiLU operator.

## Whole models

`create_model(channels=64, lmax=3, edge_lmax=None, num_interactions=2,
correlation=3, atomic_numbers=(1,6,8), cutoff=5., device='npu', dtype=torch.float32)`
creates an initialized bridge-U model. No architecture values are fixed by the
NPU kernels. Correlation is currently limited to 1/2/3 by the model factory.

`check_model(model, training=False)` is read-only. `configure_model(model,
training=False, policy='generated', message_chunk_edges=16384)` selects the
Ascend backend for NPU models. Policies are `auto`, `generated` and `reference`.
The returned dictionary identifies the backend, selected route and coverage
reasons. CUDA-specific scheduling options from upstream have no NPU semantics.

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
