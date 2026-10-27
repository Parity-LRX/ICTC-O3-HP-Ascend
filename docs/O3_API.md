# Standalone high-performance O(3) API

`ictc_o3` exposes Ascend-optimized equivariant layers for use in your own
PyTorch models. You can call tensor products, graph messages, symmetric
contractions, direction harmonics and channel operations without constructing
a MACE model or loading a MACE checkpoint.

```text
ictc_o3/       # Public O(3) operators and Ascend C backend entry points
mace_ictc/     # MACE-ICTC models, training, conversion and deployment
```

Both packages are installed by the same `ictc-o3-hp-ascend` distribution.
Operator implementations, fixed basis data, derivative utilities and native
kernels are owned by `ictc_o3`. Standalone import and execution do not import
`mace_ictc` or require the MACE model package. The model package consumes these
operators; its former operator module paths remain compatibility aliases.

```text
ictc_o3/basis/             # Fixed bases, coupling coefficients and layouts
ictc_o3/ops/               # Symmetric contractions and mathematical plans
ictc_o3/autograd/          # Derivative dependency utilities
ictc_o3/backends/ascend/   # Eager NPU execution schedules
ictc_o3/ascendc/           # Ascend C source, builder, loader and native adjoints
```

## 1. Install and enable the high-performance backend

Use the PyTorch/torch_npu versions supplied for your CANN environment. Install
the operator dependencies from the repository root:

```bash
pip install -e '.[hp]' \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

Build the native library inside an Ascend compute allocation after sourcing
the site's CANN environment:

```bash
python -m ictc_o3.ascendc.build --output /path/to/build/ictc_ascendc --jobs 4
export ICTC_ASCENDC_LIBRARY=/path/to/build/ictc_ascendc/libictc_ascendc.so
```

Use the library path printed by the builder. Rebuild when CANN, PyTorch,
torch_npu, the C++ ABI, the target SoC or native kernel sources change.
On a Slurm cluster, submit builds and examples through `sbatch`; the
[user guide](USAGE.md#1-install-and-build) provides the allocation template.
For native source organization, explicit `--soc` / `--cann-path` selection and
`--install-prefix`, see the [development guide](DEVELOPMENT.md).

```python
import torch
import torch_npu
from ictc_o3.ascendc import enable, hardware, load_library

torch.npu.set_device(0)
ops = enable(mode="performance")  # Reads ICTC_ASCENDC_LIBRARY.
ub_bytes, vector_cores = hardware()
```

`enable(path=None, *, mode="performance")` loads the native library and selects
the process-wide dispatch mode. It also enforces strict FP32: autocast must be
disabled, CANN uses `must_keep_origin_dtype`, and HF32 is disabled. These
process settings are not automatically restored. Call it before constructing
a forward graph. A supplied `path` overrides the environment variable.

`load_library(path=None)` only loads/registers the library; it does not enable
operator dispatch or configure precision. Once loaded, later calls without a
path reuse that library even if `ICTC_ASCENDC_LIBRARY` changes. Loading another
library explicitly in the same process raises an error.
`hardware(device=None)` returns `(UB capacity in bytes, Vector Core count)`
for the current NPU or a supplied NPU index, device string or `torch.device`.
The legacy `mace_ictc.ascendc` entry points alias this backend and share its
library, execution mode and hardware cache. Former tensor-product and
contraction class paths remain loadable for existing serialized modules;
new standalone modules serialize with `ictc_o3` class paths.
Importing these backend entry points does not build or load the native library.

`performance` is the default mode for training and inference. It combines
channel-contiguous GEMMs with native Vector kernels and differentiable
analytic adjoints. The optional `capacity` and `native` modes select alternative
memory/execution schedules; see [backend modes and tiling](HP_API.md#explicit-ascend-c-backend).
Omitting `enable()` retains the underlying Triton/GEMM dispatch.

## 2. Public operators and conventions

```python
from ictc_o3 import (
    EdgeTensorProduct, PackedGraphMessage, GraphMessage,
    direction_harmonics,
    create_symmetric_contraction, configure_symmetric_contraction,
    channel_linear, species_channel_linear, normalized_silu, strict_fp32,
)
```

All floating operands and parameters in the high-performance path are FP32
tensors on the same NPU. Execution uses ordinary eager autograd.
`N` is the number of nodes, `E` the number of edges, `C` the channel count,
`S` the number of species and `P` the number of tensor-product paths.
A degree `l` block has `2*l+1` angular components.

| Operator | Input → output |
| --- | --- |
| `EdgeTensorProduct` | Packed edge features, harmonics and gates → packed edge outputs |
| `PackedGraphMessage` | Packed node features and edge inputs → packed destination-node outputs |
| `GraphMessage` | Degree dictionaries → destination-node degree dictionary |
| `direction_harmonics` | Unit directions `[E,3]` → list of `[E,2*l+1]` blocks |
| `create_symmetric_contraction` | Factory for a trainable species-conditioned polynomial contraction |
| `configure_symmetric_contraction` | Select a backend on an existing compatible contraction |
| `channel_linear` | `[...,I,M]`, `[O,I]` → `[...,O,M]` |
| `species_channel_linear` | `[N,I,M]`, `[S,O,I]`, species `[N]` → `[N,O,M]` |
| `normalized_silu` | Even scalar features → `scale * silu(features)` |

The angular coordinates use the component-normalized ICTC basis. Equal shapes
do not make arbitrary e3nn spherical components interchangeable with ICTC
components: apply the corresponding basis maps when integrating another model.
The factories use natural parity, `p=(-1)^l`; arbitrary parity-labelled irreps
are not factory arguments. `normalized_silu` and scalar biases preserve O(3)
equivariance only on even scalars, not on higher-degree blocks or odd scalars.

## 3. Edge tensor product

This is a complete forward/backward example after installation and library setup:

```python
import torch
import torch_npu
from ictc_o3 import EdgeTensorProduct, direction_harmonics
from ictc_o3.ascendc import enable

torch.npu.set_device(0)
enable(mode="performance")

edges, channels, input_lmax, edge_lmax = 1024, 64, 2, 2
op = EdgeTensorProduct.create(
    channels=channels, input_lmax=input_lmax, edge_lmax=edge_lmax,
    output_lmax=2, layout="ir_mul", device="npu",
)
blocks = {
    l: torch.randn(edges, channels, 2*l+1, device="npu", requires_grad=True)
    for l in range(input_lmax + 1)
}
features = op.pack_input(blocks)
directions = torch.randn(edges, 3, device="npu", requires_grad=True)
unit = torch.nn.functional.normalize(directions, dim=-1)
harmonics = torch.cat(direction_harmonics(unit, edge_lmax, policy="generated"), -1)
gates = torch.randn(edges, op.gate_width, device="npu", requires_grad=True)

output = op(features, harmonics, gates)
output_blocks = op.unpack_output(output)
output.square().mean().backward()
```

Factory signature:

```text
EdgeTensorProduct.create(
    *, channels=64, input_lmax=2, edge_lmax=2, output_lmax=None,
    layout="ir_mul", device="npu",
)
```

`output_lmax=None` selects `input_lmax`. Channels must be positive and degrees
nonnegative integers. The factory creates all paths `(l_in,l_edge,l_out)`
satisfying the triangle rule and even `l_in+l_edge+l_out` within the requested
degree ranges. Independent input, edge and output maximum degrees are supported.

| Buffer/property | Shape or meaning |
| --- | --- |
| `features` | `[E,C*(input_lmax+1)**2]`, degree blocks concatenated in increasing order |
| `harmonics` | `[E,(edge_lmax+1)**2]`, degree blocks concatenated in increasing order |
| `gates` | `[E,P*C]` or `[E,P,C]`, ordered by `op.paths` |
| `op.paths` | Tuple of actual `(l_in,l_edge,l_out)` paths |
| `op.gate_width` | `P*C` |
| `op.output_width` | `sum_l(path_count_l*C*(2*l+1))` |
| `output` | `[E,op.output_width]` |
| `op.unpack_output(output)[l]` | `[E,path_count_l*C,2*l+1]` |

Output path multiplicities are retained. Within an output degree, paths use
their relative order in `op.paths`, with channels inside each path.
Use `unpack_output` rather than treating the result as `C` channels per degree.

For input packing, `ir_mul` stores each degree as `[angular,channel]`, while
`mul_ir` stores `[channel,angular]`. Output packing additionally retains the
path dimension. `pack_input` accepts canonical `[E,C,2*l+1]` blocks for either
layout and applies the required reordering. `unpack_output` returns canonical
blocks for either layout; degrees without paths have zero channels.

Factory path coefficients are fixed to one. External gates are live autograd
inputs. To reuse learned coefficients from an existing compatible ICTC module:

```python
op = EdgeTensorProduct(existing_tp, input_lmax=2, edge_lmax=2, layout="ir_mul")
```

The wrapper preserves the original Parameter objects and trainability.
`existing_tp` must be an ICTC path-preserving tensor product with NPU FP32
parameters and FP32 internal arithmetic. This is not an arbitrary e3nn TP adapter.

## 4. Graph messages

`PackedGraphMessage.create` accepts the tensor-product factory arguments and
`edge_chunk_size=16384`. The chunk size must be a positive integer; it bounds
edge intermediates and does not change the graph or neighbor count.

The following continues the imports and backend setup above:

```python
from ictc_o3 import PackedGraphMessage

nodes, edges, channels, lmax = 256, 2048, 32, 2
message = PackedGraphMessage.create(
    channels=channels, input_lmax=lmax, edge_lmax=lmax, output_lmax=lmax,
    edge_chunk_size=1024, layout="mul_ir", device="npu",
)
node_blocks = {
    l: torch.randn(nodes, channels, 2*l+1, device="npu", requires_grad=True)
    for l in range(lmax + 1)
}
node_features = message.pack_input(node_blocks)
unit = torch.nn.functional.normalize(torch.randn(edges, 3, device="npu"), dim=-1)
edge_harmonics = torch.cat(direction_harmonics(unit, lmax, policy="generated"), -1)
edge_gates = torch.randn(edges, message.gate_width, device="npu", requires_grad=True)
src = torch.randint(nodes, (edges,), device="npu", dtype=torch.int64)
dst = torch.randint(nodes, (edges,), device="npu", dtype=torch.int64)

node_output = message(node_features, edge_harmonics, edge_gates, src, dst)
node_output.square().mean().backward()
```

The call gathers source features, applies the gated TP and sums contributions
at destination nodes. `node_output` has shape `[N,message.output_width]`.
Indices must be NPU int64 vectors `[E]` with values in `[0,N)`.
Repeated destinations are supported. An optional `mask` of shape `[E]` or
`[E,1]` multiplies edge contributions; trainable masks keep a differentiable path.
FP32 summation order can vary with the selected reduction schedule.

For dictionary-based composition, use `GraphMessage.create(...)` with the same
factory arguments and call:

```python
from ictc_o3 import GraphMessage

message = GraphMessage.create(channels=channels, input_lmax=lmax, edge_lmax=lmax)
edge_blocks = {l: edge_harmonics[:, l*l:(l+1)**2].unsqueeze(1)
               for l in range(lmax + 1)}
destination_blocks = message(node_blocks, edge_blocks, edge_gates, src, dst)
```

Input feature blocks are `[N,C,2*l+1]`, harmonic blocks `[E,1,2*l+1]`,
and returned blocks `[N,path_count_l*C,2*l+1]`.

## 5. Symmetric contraction

```python
from ictc_o3 import create_symmetric_contraction

nodes, channels, lmax, elements = 128, 32, 2, 3
sc = create_symmetric_contraction(
    num_elements=elements, channels=channels, lmax=lmax, target_lmax=2,
    correlation=3, device="npu", policy="generated", training=True,
).train()
features = torch.randn(nodes, channels*(lmax+1)**2, device="npu", requires_grad=True)
species = torch.randint(elements, (nodes,), device="npu", dtype=torch.int64)
attributes = torch.nn.functional.one_hot(species, elements).float()
output = sc(features, attributes, node_type_idx=species)
output.square().mean().backward()
```

`features` packs degree-major `[N,C,2*l+1]` blocks flattened in
`[channel,angular]` order (`mul_ir`). Output uses the same convention with
shape `[N,C*(target_lmax+1)**2]`. `target_lmax=None` selects `lmax`.
Correlation orders 1, 2 and 3 are supported.

Species are fixed indices into learned element banks. Pass the matching
one-hot `[N,S]` attributes and integer species map `[N]`. Omitting
`node_type_idx` derives it with `attributes.argmax(-1)` on the generated path;
soft or trainable species mixtures are not its interface.

To select the high-performance backend on an existing compatible contraction:

```python
from ictc_o3 import configure_symmetric_contraction

selection = configure_symmetric_contraction(existing_sc, policy="generated", training=True)
```

Configuration preserves weights and state dictionary keys. `policy="reference"`
selects the original PyTorch contraction; `"generated"` and `"auto"` select the
Ascend implementation and require supported NPU FP32 inputs/parameters.
Constants must satisfy the implementation's reconstruction precision budget;
unsupported factorization raises an error on execution.

The `training` argument is accepted for API compatibility; it does not set
module train/eval state or freeze parameters. Use `.train()`, `.eval()` and
`.requires_grad_(False)` explicitly. Learned weights remain trainable by default.

## 6. Channel linear layers and scalar activation

`channel_linear` applies one matrix to every angular component; it mixes channels
within a degree. Call it separately for each degree:

```python
from ictc_o3 import channel_linear, species_channel_linear, normalized_silu

x = torch.randn(128, 32, 5, device="npu", requires_grad=True)  # l=2
weight = torch.nn.Parameter(torch.randn(64, 32, device="npu"))
y = channel_linear(x, weight)  # [128,64,5]

species = torch.randint(3, (128,), device="npu", dtype=torch.int64)
bank = torch.nn.Parameter(torch.randn(3, 64, 32, device="npu"))
z = species_channel_linear(x, bank, species)  # [128,64,5]

scalars = torch.randn(128, 64, device="npu", requires_grad=True)
activated = normalized_silu(scalars, scale=1.)
```

These examples assume `enable()` has configured precision. If using only these
functions without backend activation or an operator factory, call
`strict_fp32()` first. No operand is silently cast.

Optional bias is `[O]` for `channel_linear` and `[S,O]` for
`species_channel_linear`, and requires scalar angular width `M=1`.
The species layer uses shared GEMMs and selection rather than expanding an
`[N,O,I]` matrix for every node; arithmetic grows with species count.

`normalized_silu(x, scale=1.)` returns `scale * torch.nn.functional.silu(x)`.
`scale` is a fixed Python number, not a trainable tensor. The function does not
estimate a normalization constant or normalize its input.

## 7. Harmonics, autograd and complete examples

`direction_harmonics(unit_directions, lmax, *, policy="auto")` returns a list
in increasing degree order. Normalize nonzero displacements before calling it;
normalization and radial cutoff/envelope functions belong in the caller.
`"generated"` requires the generated eager NPU FP32 path; `"reference"` selects
the original PyTorch polynomials. `"auto"` tries generated harmonics, then the
reference formula. This policy is independent of Ascend C execution modes.

TPs, messages, contractions, linears and scalar activation support ordinary
backward and the mixed second derivatives needed for energy-and-force training.
Direction harmonics use native first/second derivatives and retain the polynomial
formula at higher orders. Floating reduction order can change FP32 rounding;
the backend does not require reduced precision.

For force training, compute coordinate gradients with `create_graph=True`,
then differentiate the combined energy/force loss into parameters. For frozen
force inference, freeze parameters while keeping coordinate autograd enabled;
do not wrap force evaluation in `torch.inference_mode()`.

Two runnable examples are provided:

```bash
# Edge TP with live feature and gate gradients.
python examples/operators.py --ascendc --channels 64 --lmax 2 --edges 1024

# Graph message → channel mixing → symmetric contraction → species readout.
# Includes a direction-gradient loss, mixed second backward and AdamW.
python examples/o3_layers.py --nodes 128 --edges 1024 --channels 16 --lmax 2
```

Execute them inside your compute allocation, for example through
[`examples/ascend_job.sbatch`](../examples/ascend_job.sbatch) with the environment
variables in the [user guide](USAGE.md#1-install-and-build).
Channels, angular degrees, correlation, edge chunking and graph sizes are caller
choices within each operator's stated coverage. UB tiling uses the actual widths
and device limits, including channel and row tails.

For MACE-ICTC model construction, training and checkpoint conversion, see the
[model API](HP_API.md#whole-models) and [user guide](USAGE.md).
