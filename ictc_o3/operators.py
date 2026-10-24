"""Public Ascend ICTC-O3 operators; feature and checkpoint conventions are retained."""
import torch
import torch.nn.functional as F

from mace_ictc.ascend.linear import channel_linear, species_channel_linear
from mace_ictc.ascend.runtime import require_npu_fp32, strict_fp32
from mace_ictc.ascend.tensor_product import tensor_product, graph_message
from mace_ictc.hp.operators import _degree, _positive_integer


class EdgeTensorProduct(torch.nn.Module):
    """Path-preserving TP with live external gates and FP32 NPU batched GEMMs.

    ir_mul stores each degree as [angular,channel]; mul_ir stores
    [channel,angular]. Output multiplicities retain the original path order.
    """
    def __init__(self, tp, *, input_lmax, edge_lmax, layout="ir_mul"):
        super().__init__()
        from mace_ictc.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
        if not isinstance(tp, EdgeWeightedPathPreservingTensorProduct):
            raise TypeError("Expected an ICTC path-preserving tensor product")
        _degree(input_lmax, "input_lmax")
        _degree(edge_lmax, "edge_lmax")
        require_npu_fp32(tp.weight)
        if layout not in ("ir_mul", "mul_ir"):
            raise ValueError("layout must be ir_mul or mul_ir")
        if tp.internal_compute_dtype != torch.float32:
            raise ValueError("Tensor-product internal compute dtype must be FP32")
        if not tp.paths or any(i > input_lmax or j > edge_lmax for i, j, _ in tp.paths):
            raise ValueError("Declared input degrees must cover every tensor-product path")
        strict_fp32()
        self.tp, self.input_lmax, self.edge_lmax, self.layout = tp, input_lmax, edge_lmax, layout

    @classmethod
    def create(cls, *, channels=64, input_lmax=2, edge_lmax=2,
               output_lmax=None, layout="ir_mul", device="npu"):
        import torch_npu
        _positive_integer(channels, "channels")
        _degree(input_lmax, "input_lmax")
        _degree(edge_lmax, "edge_lmax")
        output_lmax = input_lmax if output_lmax is None else output_lmax
        _degree(output_lmax, "output_lmax")
        if torch.device(device).type != "npu":
            raise ValueError("This operator factory requires an Ascend NPU")
        from mace_ictc.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
        paths = [(i, j, k) for i in range(input_lmax+1) for j in range(edge_lmax+1)
                 for k in range(output_lmax+1) if abs(i-j) <= k <= i+j and (i+j+k)%2 == 0]
        tp = EdgeWeightedPathPreservingTensorProduct(
            channels=channels, lmax=max(input_lmax, edge_lmax, output_lmax),
            allowed_paths=paths, internal_compute_dtype=torch.float32,
        ).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            tp.weight.fill_(1.)
        tp.weight.requires_grad_(False)
        return cls(tp, input_lmax=input_lmax, edge_lmax=edge_lmax, layout=layout)

    @property
    def paths(self):
        return tuple(tuple(p) for p in self.tp.paths)

    @property
    def gate_width(self):
        return self.tp.num_paths*self.tp.channels

    @property
    def output_width(self):
        return sum(count*self.tp.channels*(2*l+1) for l, count in self.tp.path_counts_by_l.items())

    def pack_input(self, blocks):
        parts = []
        for l in range(self.input_lmax+1):
            value = blocks[l]
            if value.shape[1:] != (self.tp.channels, 2*l+1):
                raise ValueError("Input block must have shape [E,C,2*l+1]")
            if self.layout == "ir_mul":
                value = value.transpose(1, 2)
            parts.append(value.reshape(value.shape[0], self.tp.channels*(2*l+1)))
        return torch.cat(parts, -1)

    def _blocks(self, features):
        require_npu_fp32(features, self.tp.weight)
        if features.ndim != 2 or features.shape[1] != self.tp.channels*(self.input_lmax+1)**2:
            raise ValueError("Packed features must have shape [E,C*(L_in+1)^2]")
        result, offset = {}, 0
        for l in range(self.input_lmax+1):
            width = self.tp.channels*(2*l+1)
            value = features[:, offset:offset+width]
            result[l] = (value.reshape(features.shape[0], 2*l+1, self.tp.channels).transpose(1, 2)
                         if self.layout == "ir_mul" else value.reshape(features.shape[0], self.tp.channels, 2*l+1))
            offset += width
        return result

    def _harmonics(self, harmonics, rows):
        require_npu_fp32(harmonics, self.tp.weight)
        if harmonics.shape != (rows, (self.edge_lmax+1)**2):
            raise ValueError("Harmonics must have shape [E,(L_edge+1)^2]")
        return {l: harmonics[:, l*l:(l+1)**2].unsqueeze(1) for l in range(self.edge_lmax+1)}

    def _pack_output(self, blocks):
        rows, parts = next(iter(blocks.values())).shape[0], []
        for l in range(self.tp.lmax+1):
            count, width = self.tp.path_counts_by_l.get(l, 0), 2*l+1
            value = blocks[l].reshape(rows, count, self.tp.channels, width)
            if self.layout == "ir_mul":
                value = value.transpose(2, 3)
            parts.append(value.reshape(rows, count*self.tp.channels*width))
        return torch.cat(parts, -1)

    def unpack_output(self, output):
        result, offset = {}, 0
        for l in range(self.tp.lmax+1):
            count, width = self.tp.path_counts_by_l.get(l, 0), 2*l+1
            span = count*self.tp.channels*width
            value = output[:, offset:offset+span]
            if self.layout == "ir_mul":
                value = value.reshape(output.shape[0], count, width, self.tp.channels).transpose(2, 3)
            result[l] = value.reshape(output.shape[0], count*self.tp.channels, width)
            offset += span
        return result

    def forward(self, features, harmonics, gates):
        blocks = tensor_product(self.tp, self._blocks(features),
                                self._harmonics(harmonics, features.shape[0]), gates)
        return self._pack_output(blocks)


class PackedGraphMessage(EdgeTensorProduct):
    """Packed node gather, AI Core TP and chunked destination aggregation."""
    def __init__(self, *args, edge_chunk_size=16384, **kwargs):
        super().__init__(*args, **kwargs)
        _positive_integer(edge_chunk_size, "edge_chunk_size")
        self.edge_chunk_size = edge_chunk_size

    @classmethod
    def create(cls, *, edge_chunk_size=16384, **kwargs):
        op = EdgeTensorProduct.create(**kwargs)
        return cls(op.tp, input_lmax=op.input_lmax, edge_lmax=op.edge_lmax,
                   layout=op.layout, edge_chunk_size=edge_chunk_size)

    def forward(self, features, harmonics, gates, src, dst, mask=None):
        for value in (src, dst):
            if value.ndim != 1 or value.dtype != torch.int64 or value.device != features.device:
                raise ValueError("Graph indices must be int64 vectors on the NPU")
        if dst.shape != src.shape:
            raise ValueError("Source and destination index shapes must match")
        result = graph_message(self.tp, self._blocks(features),
            self._harmonics(harmonics, src.numel()), gates, src, dst, mask,
            edge_chunk_size=self.edge_chunk_size)
        return self._pack_output(result)


class GraphMessage(PackedGraphMessage):
    """Dictionary block interface for the same Ascend graph message map."""
    def forward(self, features, harmonics, gates, src, dst, mask=None):
        return graph_message(self.tp, features, harmonics, gates, src, dst, mask,
                             edge_chunk_size=self.edge_chunk_size)


def direction_harmonics(directions, lmax, *, policy="auto"):
    _degree(lmax, "lmax")
    if policy not in ("auto", "generated", "reference"):
        raise ValueError("policy must be auto, generated or reference")
    if policy != "reference":
        require_npu_fp32(directions)
        from mace_ictc.models.generated_direction_harmonics import generated_direction_harmonics
        result = generated_direction_harmonics(directions, lmax)
        if result is not None:
            return result
        if policy == "generated":
            raise ValueError("Ascend harmonics require eager NPU FP32 and Triton-Ascend")
    from mace_ictc.models.ictd_irreps import direction_harmonics_all
    return direction_harmonics_all(directions, lmax)


def configure_symmetric_contraction(module, *, policy="auto", training=False):
    from mace_ictc.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    if not isinstance(module, ICTDBridgeUSymmetricContractionSO3):
        raise TypeError("Expected an ICTC bridge-U symmetric contraction")
    if policy not in ("auto", "generated", "reference"):
        raise ValueError("policy must be auto, generated or reference")
    if policy != "reference":
        require_npu_fp32(*module.parameters())
        if any(c.correlation not in (1,2,3) for c in module.symmetric_contractions.contractions):
            raise ValueError("Ascend symmetric contraction supports correlation 1, 2 and 3")
        strict_fp32()
    module._ascend_hp = policy != "reference"
    return {"backend": "ascend", "selected": "generated" if module._ascend_hp else "reference"}


def create_symmetric_contraction(*, num_elements, lmax, target_lmax=None, channels=64,
                                correlation=3, device="npu", policy="generated", training=False):
    import torch_npu
    for value, name in ((num_elements, "num_elements"), (channels, "channels"), (correlation, "correlation")):
        _positive_integer(value, name)
    if correlation not in (1, 2, 3):
        raise ValueError("Ascend symmetric contraction supports correlation 1, 2 and 3")
    _degree(lmax, "lmax")
    target_lmax = lmax if target_lmax is None else target_lmax
    _degree(target_lmax, "target_lmax")
    from mace_ictc.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    module = ICTDBridgeUSymmetricContractionSO3(num_elements=num_elements, channels=channels,
        lmax=lmax, target_lmax=target_lmax, correlation=correlation).to(device=device, dtype=torch.float32)
    configure_symmetric_contraction(module, policy=policy, training=training)
    return module


def normalized_silu(features, scale=1.):
    require_npu_fp32(features)
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise TypeError("scale must be a fixed numeric scalar")
    return F.silu(features)*float(scale)


__all__ = ["EdgeTensorProduct", "PackedGraphMessage", "GraphMessage", "direction_harmonics",
           "create_symmetric_contraction", "configure_symmetric_contraction",
           "channel_linear", "species_channel_linear", "normalized_silu", "strict_fp32"]
