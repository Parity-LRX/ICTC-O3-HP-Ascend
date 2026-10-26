"""Inference adapter for the nonlinear MACE foundation architecture.

The original radial embedding, density normalization, nonlinearities, residual
linears and all readout heads are retained. Convolution and symmetric products
use calibrated ICTC operators. No learned tensor is approximated or retrained.
"""
from __future__ import annotations

import copy
from collections import Counter
from threading import RLock

import torch
from torch import nn
from torch.nn import functional as F

from mace_ictc.interfaces.mace_converter import (
    _calibrate_conv_tp, _copy_symmetric_contraction,
    _e3nn_linear_effective_matrices, _mace_conv_tp_paths,
    _reorder_linear_input_paths,
)
from mace_ictc.mace_basis import orthogonal_Q_blocks
from mace_ictc.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from mace_ictc.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3


_LOAD_LOCK = RLock()


def load_foundation_checkpoint(path, *, map_location="cpu"):
    """Load official checkpoints containing legacy e3nn TorchScript buffers.

    Older e3nn stores raw TorchScript bytes in __codegen__; newer versions tag
    TorchScript and FX buffers. Restore either container independently of the
    installed e3nn version, retaining every checkpoint tensor verbatim.
    """
    from e3nn.util.codegen import CodeGenMixin
    with _LOAD_LOCK:
        original = CodeGenMixin.__setstate__

        def restore(module, state):
            codegen = state.get("__codegen__")
            if not isinstance(codegen, dict):
                return original(module, state)
            import io
            import pickle
            state = dict(state)
            del state["__codegen__"]
            original(module, state)
            for name, value in codegen.items():
                kind, buffer = ("torchscript", value) if isinstance(value, bytes) else value
                if not isinstance(name, str) or not isinstance(buffer, bytes):
                    raise ValueError("Invalid e3nn generated-module buffer")
                if kind == "torchscript":
                    child = torch.jit.load(io.BytesIO(buffer), map_location=map_location)
                elif kind == "fx":
                    child = pickle.loads(buffer)
                else:
                    raise ValueError(f"Unsupported e3nn generated-module format: {kind}")
                module.add_module(name, child)
            module.__codegen__ = list(codegen)

        CodeGenMixin.__setstate__ = restore
        try:
            model = torch.load(path, map_location=map_location, weights_only=False)
        finally:
            CodeGenMixin.__setstate__ = original
    # Metadata added by newer e3nn forwards; the checkpoint operators and
    # tensors stay intact. This matches the qualified A100 foundation loader.
    from e3nn.o3._spherical_harmonics import _spherical_harmonics
    loaded_module = model.get("model") if isinstance(model, dict) else model
    if loaded_module is None and isinstance(model, dict):
        return model
    if not isinstance(loaded_module, nn.Module):
        raise ValueError("Expected a trusted MACE model or a converted foundation archive")
    for module in loaded_module.modules():
        if module.__class__.__name__ not in ("SphericalHarmonics", "Activation"):
            continue
        forward_code = getattr(type(module).forward, "__code__", None)
        forward_names = () if forward_code is None else forward_code.co_names
        if (module.__class__.__name__ == "SphericalHarmonics" and "sph_func" in forward_names
                and not hasattr(module, "sph_func")):
            module.sph_func = _spherical_harmonics
        if (module.__class__.__name__ == "Activation" and "paths" in forward_names
                and not hasattr(module, "paths")):
            module.paths = [(mul, (ir.l, ir.p), act)
                for (mul, ir), act in zip(module.irreps_in, module.acts)]
    return model


def _degrees(irreps):
    result = {}
    for mul, ir in irreps:
        if ir.p != (-1) ** ir.l or ir.l in result:
            raise ValueError(f"Expected unique natural-parity degree blocks: {irreps}")
        result[ir.l] = int(mul)
    if set(result) != set(range(max(result) + 1)):
        raise ValueError(f"Expected contiguous degrees: {irreps}")
    return result


def _blocks(flat, degrees):
    result, offset = {}, 0
    for degree, channels in degrees.items():
        width = channels * (2 * degree + 1)
        result[degree] = flat[:, offset:offset + width].reshape(-1, channels, 2 * degree + 1)
        offset += width
    if offset != flat.shape[1]:
        raise ValueError("Irrep dimensions do not match the feature tensor")
    return result


class _NarrowRadial(torch.autograd.Function):
    """Keep narrow edge activations; rematerialize frozen radial networks in VJP.

    Element embeddings are fixed lookup tables. Only geometric arguments carry
    derivatives. The native Linear/LayerNorm/SiLU order is retained per edge.
    Higher coordinate derivatives fall back to the full differentiable formula.
    """
    @staticmethod
    def forward(ctx, radial, cutoff, species, src, dst, owner):
        ctx.owner = owner
        ctx.save_for_backward(radial, cutoff, species, src, dst)
        outputs, densities = [], []
        for begin in range(0, src.numel(), owner.radial_chunk):
            end = min(begin + owner.radial_chunk, src.numel())
            h, density = owner.radial_formula(radial[begin:end], cutoff[begin:end],
                species[src[begin:end]], species[dst[begin:end]])
            outputs.append(h)
            densities.append(density)
        if not outputs:
            return radial.new_empty((0, owner.hidden_width + 1)), radial.new_empty((0, 1))
        return torch.cat(outputs), torch.cat(densities)

    @staticmethod
    def backward(ctx, grad_h, grad_density):
        radial, cutoff, species, src, dst = ctx.saved_tensors
        owner = ctx.owner
        higher = torch.is_grad_enabled()
        if higher:
            with torch.enable_grad():
                local_r, local_c = radial.clone(), cutoff.clone()
                h, density = owner.radial_formula(local_r, local_c, species[src], species[dst])
                active = [v for v, need in zip((local_r, local_c), ctx.needs_input_grad[:2]) if need]
                values = iter(torch.autograd.grad((h, density), active, (grad_h, grad_density),
                    create_graph=True, allow_unused=True))
                grads = [next(values) if need else None for need in ctx.needs_input_grad[:2]]
            return *grads, None, None, None, None
        dr = torch.empty_like(radial) if ctx.needs_input_grad[0] else None
        dc = torch.empty_like(cutoff) if ctx.needs_input_grad[1] else None
        for begin in range(0, src.numel(), owner.radial_chunk):
            end = min(begin + owner.radial_chunk, src.numel())
            with torch.enable_grad():
                local_r = radial[begin:end].detach().requires_grad_(ctx.needs_input_grad[0])
                local_c = cutoff[begin:end].detach().requires_grad_(ctx.needs_input_grad[1])
                h, density = owner.radial_formula(local_r, local_c,
                    species[src[begin:end]], species[dst[begin:end]])
                active = [v for v in (local_r, local_c) if v.requires_grad]
                values = iter(torch.autograd.grad((h, density), active,
                    (grad_h[begin:end], grad_density[begin:end]), allow_unused=True))
                for target, value in ((dr, local_r), (dc, local_c)):
                    if value.requires_grad:
                        gradient = next(values)
                        target[begin:end] = torch.zeros_like(value) if gradient is None else gradient
        return dr, dc, None, None, None, None


class FoundationConvolution(nn.Module):
    def __init__(self, interaction, species_count):
        super().__init__()
        self.input_degrees = _degrees(interaction.conv_tp.irreps_in1)
        if len(set(self.input_degrees.values())) != 1:
            raise ValueError("HP TP requires uniform convolution channel multiplicities")
        self.channels = next(iter(self.input_degrees.values()))
        self.input_lmax = max(self.input_degrees)
        self.edge_lmax = interaction.conv_tp.irreps_in2.lmax
        self.output_degrees = _degrees(interaction.linear_1.irreps_out.simplify())
        paths = _mace_conv_tp_paths(interaction.conv_tp)
        self.tp = EdgeWeightedPathPreservingTensorProduct(channels=self.channels,
            lmax=max(self.edge_lmax, max(p[2] for p in paths)), allowed_paths=paths,
            path_policy="full", internal_compute_dtype=torch.float64).double()
        with torch.no_grad():
            self.tp.weight.fill_(1.)
        _, scales = _calibrate_conv_tp(interaction.conv_tp, self.tp,
            input_lmax=self.input_lmax, lmax=self.edge_lmax, channels=self.channels)
        eye = torch.eye(species_count, dtype=torch.float64)
        with torch.no_grad():
            self.register_buffer("source_table", interaction.source_embedding(eye).detach())
            self.register_buffer("target_table", interaction.target_embedding(eye).detach())
        net = interaction.conv_tp_weights.net
        if not isinstance(net[-1], nn.Linear):
            raise ValueError("Expected a final dense radial gate projection")
        self.radial_hidden = copy.deepcopy(net[:-1])
        self.density_fn = copy.deepcopy(interaction.density_fn)
        self.hidden_width = net[-1].in_features
        last = nn.Linear(self.hidden_width + 1, net[-1].out_features, bias=False, dtype=torch.float64)
        self.projection = nn.Sequential(nn.Identity(), last)
        with torch.no_grad():
            for old_index, path in enumerate(paths):
                new_index = self.tp.paths.index(path)
                old_rows = slice(old_index * self.channels, (old_index + 1) * self.channels)
                new_rows = slice(new_index * self.channels, (new_index + 1) * self.channels)
                last.weight[new_rows, :-1] = net[-1].weight[old_rows] * scales[path]
                last.weight[new_rows, -1] = (0. if net[-1].bias is None else
                    net[-1].bias[old_rows] * scales[path])
            self.tp.weight.fill_(1.)
        counts = Counter(p[2] for p in paths)
        matrices = _e3nn_linear_effective_matrices(interaction.linear_1,
            channels_in_by_l={l: self.channels * counts[l] for l in counts},
            channels_out_by_l=self.output_degrees)
        matrices = _reorder_linear_input_paths(matrices, interaction.conv_tp, self.tp,
            channels=self.channels, target_lmax=max(self.output_degrees))
        for degree, matrix in matrices.items():
            self.register_buffer(f"message_matrix_{degree}", matrix.T.contiguous())
        for degree, q in enumerate(orthogonal_Q_blocks(self.edge_lmax)):
            self.register_buffer(f"q_{degree}", q)
        self.radial_chunk = 32768
        self.generated = False
        self.message_calls = 0

    def configure(self, *, chunk_edges=None, radial_chunk=32768):
        from mace_ictc.ascend.runtime import require_npu_fp32, strict_fp32
        require_npu_fp32(*self.parameters())
        strict_fp32()
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError("Foundation inference requires explicitly frozen parameters")
        if chunk_edges is not None and (not isinstance(chunk_edges, int) or isinstance(chunk_edges, bool) or chunk_edges < 1):
            raise ValueError("chunk_edges must be a positive integer or None")
        if not isinstance(radial_chunk, int) or isinstance(radial_chunk, bool) or radial_chunk < 1:
            raise ValueError("radial_chunk must be a positive integer")
        self.generated = True
        self.radial_chunk = radial_chunk
        self.chunk_edges = chunk_edges
        self.tp.internal_compute_dtype = torch.float32
        self.tp._ascend_hp = True
        self.tp._ascend_tp_schedule = "structured"

    def radial_formula(self, radial, cutoff, source_species, target_species):
        features = torch.cat((radial, self.source_table[source_species],
            self.target_table[target_species]), dim=-1)
        hidden = self.radial_hidden(features)
        hidden = torch.cat((hidden, torch.ones_like(cutoff)), dim=-1) * cutoff
        density = torch.tanh(self.density_fn(features).square()) * cutoff
        return hidden, density

    def forward(self, projected, harmonics, radial, cutoff, species, src, dst, owned):
        blocks = {l: value @ getattr(self, f"q_{l}").T
            for l, value in _blocks(projected, self.input_degrees).items()}
        if self.generated:
            hidden, edge_density = _NarrowRadial.apply(radial, cutoff, species, src, dst, self)
            gates = self.projection(hidden)
            from mace_ictc.ascend.tensor_product import graph_message
            self.tp._ascend_destination_nodes = int(owned)
            outputs = graph_message(self.tp, blocks,
                {l: value.unsqueeze(-2) for l, value in enumerate(harmonics)},
                gates, src, dst, edge_chunk_size=self.chunk_edges)
        else:
            hidden, edge_density = self.radial_formula(radial, cutoff, species[src], species[dst])
            gates = self.projection(hidden)
            edge_outputs = self.tp({l: value[src] for l, value in blocks.items()},
                {l: value.unsqueeze(-2) for l, value in enumerate(harmonics)}, gates)
            outputs = {l: value.new_zeros(owned, *value.shape[1:]).index_add(0, dst, value)
                for l, value in edge_outputs.items()}
        density = edge_density.new_zeros((owned, 1)).index_add(0, dst, edge_density)
        messages = []
        for degree, value in outputs.items():
            matrix = getattr(self, f"message_matrix_{degree}")
            mixed = F.linear(value.transpose(-2, -1), matrix).transpose(-2, -1)
            messages.append((mixed @ getattr(self, f"q_{degree}")).flatten(1))
        self.message_calls += 1
        return torch.cat(messages, -1), density


class FoundationProduct(nn.Module):
    def __init__(self, product, use_reduced_cg):
        super().__init__()
        source = product.symmetric_contractions
        input_degrees = _degrees(source.irreps_in)
        output_degrees = _degrees(source.irreps_out)
        self.channels = next(iter(input_degrees.values()))
        if set(input_degrees.values()) != {self.channels} or set(output_degrees.values()) != {self.channels}:
            raise ValueError("Expected uniform symmetric-product multiplicities")
        self.lmax, self.target_lmax = max(input_degrees), max(output_degrees)
        elements = source.contractions[0].weights_max.shape[0]
        self.agnostic = bool(getattr(product, "use_agnostic_product", False))
        self.sc = ICTDBridgeUSymmetricContractionSO3(num_elements=elements,
            channels=self.channels, lmax=self.lmax, target_lmax=self.target_lmax,
            correlation=source.contractions[0].correlation, use_reduced_cg=use_reduced_cg).double()
        _copy_symmetric_contraction(source, self.sc)
        # Preserve the checkpoint's actual CG constants, including their stored
        # precision, rather than silently regenerating a different U bank.
        with torch.no_grad():
            for degree, (original, converted) in enumerate(zip(source.contractions,
                    self.sc.symmetric_contractions.contractions)):
                for order in range(1, original.correlation + 1):
                    value = getattr(original, f"U_matrix_{order}")
                    getattr(converted, f"U_matrix_{order}").copy_(
                        self.sc._transform_u_tensor(value, degree))
        for degree, q in enumerate(orthogonal_Q_blocks(max(self.lmax, self.target_lmax))):
            self.register_buffer(f"q_{degree}", q)
        self.calls = 0

    def configure(self):
        from ictc_o3 import configure_symmetric_contraction
        return configure_symmetric_contraction(self.sc, policy="generated")

    def forward(self, features, attrs, species):
        # Native product input is [N,C,(L+1)^2]; bridge-U accepts degree-major flat features.
        blocks = [features[:, :, l*l:(l+1)**2] @ getattr(self, f"q_{l}").T
            for l in range(self.lmax + 1)]
        flat = torch.cat([value.flatten(1) for value in blocks], -1)
        if self.agnostic:
            attrs = attrs.new_ones((attrs.shape[0], 1))
            species = torch.zeros_like(species)
        output = self.sc(flat, attrs, node_type_idx=species)
        self.calls += 1
        return torch.cat([(value @ getattr(self, f"q_{l}")).flatten(1)
            for l, value in _blocks(output,
                {l: self.channels for l in range(self.target_lmax + 1)}).items()], -1)


class FoundationHP(nn.Module):
    """Exact architecture adapter, selecting one original head per evaluation.

    CPU/FP64 uses the differentiable ICTC reference. configure_inference enables
    Ascend grouped TP and native bridge-U symmetric contractions in FP32.
    Distributed calls exchange projected features after linear_up, retaining
    only owned hidden states. All heads remain in the serialized checkpoint.
    """
    def __init__(self, model, *, head="omat_pbe"):
        super().__init__()
        if model.__class__.__name__ != "ScaleShiftMACE":
            raise ValueError("Expected ScaleShiftMACE")
        if any(layer.__class__.__name__ != "RealAgnosticResidualNonLinearInteractionBlock"
            for layer in model.interactions):
            raise ValueError("Foundation adapter supports nonlinear residual interactions")
        if hasattr(model, "joint_embedding"):
            raise ValueError("Additional charge/spin embeddings need an explicit input adapter")
        self.native = copy.deepcopy(model).double().eval()
        self.atomic_numbers = tuple(int(z) for z in model.atomic_numbers.tolist())
        self.architecture = "mace-foundation-nonlinear"
        self.supports_training = False
        mapping = torch.full((max(self.atomic_numbers)+1,), -1, dtype=torch.int64)
        mapping[torch.tensor(self.atomic_numbers)] = torch.arange(len(self.atomic_numbers))
        self.register_buffer("_atomic_number_to_index", mapping)
        self.heads = list(model.heads)
        if head not in self.heads:
            raise ValueError(f"Head {head!r} not found in {self.heads}")
        self.head = head
        self.head_index = self.heads.index(head)
        self.num_interaction = len(model.interactions)
        self.max_radius = float(model.r_max)
        self.edge_lmax = int(model.spherical_harmonics._lmax)
        self.convolutions = nn.ModuleList([FoundationConvolution(layer, len(model.atomic_numbers))
            for layer in self.native.interactions])
        self.products = nn.ModuleList([FoundationProduct(product, bool(getattr(model, "use_reduced_cg", False)))
            for product in self.native.products])
        identity = torch.eye(len(model.atomic_numbers), dtype=torch.float64)
        with torch.no_grad():
            self.register_buffer("atomic_energy_table",
                self.native.atomic_energies_fn(identity).detach())
            self.register_buffer("embedding_table", self.native.node_embedding(identity).detach())
            self.register_buffer("first_projected_table",
                self.native.interactions[0].linear_up(self.embedding_table).detach())
        for layer in self.native.interactions:
            layer.conv_tp = nn.Identity()
            layer.conv_tp_weights = nn.Identity()
            layer.source_embedding = nn.Identity()
            layer.target_embedding = nn.Identity()
            layer.density_fn = nn.Identity()
            layer.linear_1 = nn.Identity()
        for product in self.native.products:
            product.symmetric_contractions = nn.Identity()
        # Element lookups and ICTC direction operators replace these modules.
        # Keeping a second embedding and an unused scripted SH callable would
        # duplicate checkpoint state without participating in inference.
        self.native.node_embedding = nn.Identity()
        self.native.spherical_harmonics = nn.Identity()
        self.generated = False
        self.requires_grad_(False)

    def select_head(self, head):
        """Select one retained original readout head without changing weights."""
        if head not in self.heads:
            raise ValueError(f"Head {head!r} not found in {self.heads}")
        self.head, self.head_index = head, self.heads.index(head)
        return self

    def configure_inference(self, *, chunk_edges=None, radial_chunk=32768):
        from mace_ictc.ascendc.runtime import is_enabled
        from mace_ictc.ascend.runtime import require_npu_fp32
        if not is_enabled():
            raise ValueError("Enable the Ascend C backend before configuring foundation inference")
        require_npu_fp32(*self.parameters())
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError("Foundation inference requires explicitly frozen parameters")
        for layer in self.convolutions:
            layer.configure(chunk_edges=chunk_edges, radial_chunk=radial_chunk)
        plans = [product.configure() for product in self.products]
        self.generated = True
        return dict(backend="ascend", selected="generated", frozen_inference=True,
            message="channel-contiguous ICTC TP", products=plans,
            chunk_edges=chunk_edges, radial_chunk=radial_chunk,
            head=self.head, heads=self.heads, precision="FP32; HF32 disabled")

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.pop("_species_cache", None)
        return state

    def species_indices(self, atomic_numbers):
        import weakref
        cached = getattr(self, "_species_cache", None)
        if cached is not None and cached[0]() is atomic_numbers and cached[1] == atomic_numbers._version:
            return cached[2]
        if atomic_numbers.numel() and (int(atomic_numbers.min()) < 0 or int(atomic_numbers.max()) >= self._atomic_number_to_index.numel()):
            raise ValueError("Atomic number is outside this checkpoint's element map")
        indices = self._atomic_number_to_index[atomic_numbers]
        if indices.numel() and bool((indices < 0).any()):
            raise ValueError("An element is absent from the foundation checkpoint")
        self._species_cache = weakref.ref(atomic_numbers), atomic_numbers._version, indices
        return indices

    def atomic_reference_energies(self, atomic_numbers):
        """Return selected-head E0 per atom, to be added once by the caller."""
        return self.atomic_energy_table[self.species_indices(atomic_numbers), self.head_index]

    @staticmethod
    def _readout(module, features, node_heads):
        # MACE's mask_head allocates with the process default dtype. Keep the
        # identical head mask in the feature dtype, including CPU reference use.
        if module.__class__.__name__ == "NonLinearReadoutBlock":
            value = module.non_linearity(module.linear_1(features))
            heads = getattr(module, "num_heads", 1)
            if heads > 1:
                mask = F.one_hot(node_heads, heads).to(value).unsqueeze(-1)
                value = (value.reshape(value.shape[0], heads, value.shape[1]//heads)
                         * mask).reshape_as(value)
            return module.linear_2(value)
        return module(features, node_heads)

    def forward(self, positions, atomic_numbers, batch, edge_src, edge_dst,
                edge_shifts, cell, *, precomputed_edge_vec=None, layer_state_callback=None):
        """Return per-node interaction energies; the caller adds selected-head E0.

        The graph and total node count are dynamic. A deployment engine can
        request owned rows and supply projected-feature exchange between layers.
        """
        species = self.species_indices(atomic_numbers)
        if precomputed_edge_vec is None:
            shifts = torch.bmm(edge_shifts.unsqueeze(1), cell[batch[edge_src]]).squeeze(1)
            vectors = positions[edge_dst] - positions[edge_src] + shifts
        else:
            vectors = precomputed_edge_vec
        owned = getattr(self, "_ascend_owned_nodes", species.numel())
        result = self.forward_vectors(vectors, species, edge_src, edge_dst,
                                     owned=owned, exchange=layer_state_callback)
        value = result["interaction_energy"]
        if owned != species.numel() and not getattr(self, "_ascend_compact_outputs", False):
            value = torch.cat((value, value.new_zeros(species.numel()-owned)))
        return value

    def forward_vectors(self, vectors, species, src, dst, *, owned=None, exchange=None):
        total = species.numel()
        owned = total if owned is None else owned
        attrs = F.one_hot(species, len(self.native.atomic_numbers)).to(vectors)
        lengths = vectors.norm(dim=-1, keepdim=True)
        from ictc_o3 import direction_harmonics
        harmonics = direction_harmonics(vectors / lengths.clamp_min(1e-12), self.edge_lmax,
            policy="generated" if self.generated else "reference")
        radial, cutoff = self.native.radial_embedding(lengths, attrs,
            torch.stack((src, dst)), self.native.atomic_numbers)
        if cutoff is None:
            cutoff = torch.ones_like(lengths)
        node_heads = torch.full((owned,), self.head_index, device=vectors.device, dtype=torch.long)
        if hasattr(self.native, "pair_repulsion"):
            pair = self.native.pair_repulsion_fn(lengths, attrs,
                torch.stack((src, dst)), self.native.atomic_numbers)[:owned]
        else:
            pair = vectors.new_zeros(owned)
        features = self.embedding_table[species[:owned]]
        energies = [pair]
        for index, (native, convolution, product, native_product) in enumerate(zip(
                self.native.interactions, self.convolutions, self.products, self.native.products)):
            sc = native.skip_tp(features)
            projected = native.linear_up(features)
            residual = native.linear_res(projected)
            del features
            if index == 0:
                projected = torch.cat((projected, self.first_projected_table[species[owned:]]), 0)
            elif exchange is not None:
                projected = exchange(index-1, torch.cat((projected,
                    projected.new_zeros(total - owned, projected.shape[-1])), 0))
            elif owned != total:
                raise ValueError("Distributed hidden features require an exchange callback")
            message, density = convolution(projected, harmonics, radial, cutoff, species, src, dst, owned)
            del projected
            message = message / (density * native.beta + native.alpha) + residual
            del density, residual
            message = native.reshape(native.linear_2(native.equivariant_nonlin(message)))
            contracted = product(message, attrs[:owned], species[:owned])
            del message
            features = native_product.linear(contracted)
            del contracted
            if native_product.use_sc:
                features = features + sc
            del sc
            if len(self.native.readouts) != 1:
                energies.append(self._readout(self.native.readouts[index], features, node_heads)[
                    torch.arange(owned, device=vectors.device), node_heads])
        if len(self.native.readouts) == 1:
            energies.append(self._readout(self.native.readouts[0], features, node_heads)[
                torch.arange(owned, device=vectors.device), node_heads])
        interaction = self.native.scale_shift(torch.stack(energies).sum(0), node_heads)
        e0 = self.native.atomic_energies_fn(attrs[:owned])[
            torch.arange(owned, device=vectors.device), node_heads]
        return dict(interaction_energy=interaction, node_energy=e0 + interaction)
