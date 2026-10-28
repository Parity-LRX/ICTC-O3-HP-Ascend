# SPDX-License-Identifier: MIT
"""Dynamic Ascend model construction, configuration and force evaluation."""
from contextlib import nullcontext
import torch


def create_model(*, channels=64, lmax=3, edge_lmax=None, num_interactions=2,
                 correlation=3, use_reduced_cg=False,
                 atomic_numbers=(1, 6, 8), cutoff=5.0, avg_num_neighbors=50.0,
                 device="npu", dtype=torch.float32, electrostatics=None):
    """Create a parameterized Cartesian bridge-U model, without a runtime plan.

    New TP base coefficients are fixed by construction; radial/linear/SC weights
    are trainable. This factory creates initialized weights, not a trained
    potential. Existing checkpoints should be loaded through from_checkpoint.
    """
    if str(device).startswith("npu"):
        import torch_npu
        if dtype != torch.float32:
            raise ValueError("The Ascend factory requires FP32; use CPU for FP64 reference calculations")
    if not isinstance(num_interactions, int) or isinstance(num_interactions, bool) or num_interactions < 1:
        raise ValueError("num_interactions must be a positive integer")
    from ictc_o3.operators import _degree, _positive_integer
    _positive_integer(channels, "channels")
    _positive_integer(correlation, "correlation")
    if correlation not in (1,2,3):
        raise ValueError("HP correlation currently supports 1, 2, 3; cor4/5 experiments are separate")
    if not isinstance(use_reduced_cg, bool):
        raise TypeError("use_reduced_cg must be a boolean")
    if use_reduced_cg:
        raise ValueError("The independent HP factory uses native full CG; cuEq reduced-basis generation is not enabled")
    _degree(lmax, "lmax")
    edge_lmax = lmax if edge_lmax is None else edge_lmax
    _degree(edge_lmax, "edge_lmax")
    if edge_lmax < lmax:
        raise ValueError("edge_lmax must be at least lmax")
    from collections.abc import Mapping
    long_range_options = {}
    if electrostatics is not None:
        if not isinstance(electrostatics, Mapping):
            raise TypeError("electrostatics must be a configuration mapping or None")
        allowed = {'mode', 'boundary', 'reciprocal_backend', 'kmax', 'mesh_size',
                   'source_channels', 'hidden_dim', 'filter_hidden_dim', 'neutralize',
                   'include_k0', 'energy_partition', 'green_mode', 'assignment',
                   'slab_padding_factor', 'mesh_fft_full_ewald', 'mesh_fft_reciprocal_only',
                   'max_multipole_l', 'multipole_gate_init', 'smearing_sigma'}
        unknown = set(electrostatics) - allowed
        if unknown:
            raise ValueError(f"Unknown electrostatics options: {sorted(unknown)}")
        options = dict(mode='reciprocal-spectral-v1', boundary='periodic',
                       reciprocal_backend='mesh_fft', mesh_size=32, assignment='pcs',
                       mesh_fft_full_ewald=True, mesh_fft_reciprocal_only=True)
        options.update(electrostatics)
        if 'smearing_sigma' not in options:
            options['smearing_sigma'] = (1.0 if options['reciprocal_backend'] == 'mesh_fft'
                                        and options['mesh_fft_full_ewald'] else None)
        long_range_options = {'long_range_' + key: value for key, value in options.items()}
    from mace_ictc.models.factory import build_baseline_model
    from mace_ictc.utils.config import ModelConfig
    import e3nn
    previous_dtype = torch.get_default_dtype()
    previous_optimization = e3nn.get_optimization_defaults()
    try:
        # Scripted e3nn size(x)[0] becomes an int and bakes N during make_fx.
        # Build symbolic-friendly FX submodules, restoring caller defaults below.
        e3nn.set_optimization_defaults(jit_script_fx=False)
        config = ModelConfig(dtype=dtype, internal_compute_dtype=dtype, channel_in=channels,
            irreps_output_conv_channels=channels, lmax=lmax, max_radius=cutoff,
            max_radius_main=cutoff, function_type="bessel")
        return build_baseline_model(
            config, avg_num_neighbors=avg_num_neighbors, num_interaction=num_interactions,
            route="baseline", product_backend="ictd-bridge-u", correlation=correlation,
            use_reduced_cg=use_reduced_cg, freeze_conv_tp_weight=True, radial_sqrt_num_basis=False,
            edge_lmax=edge_lmax, attn_heads=0, atomic_numbers=list(atomic_numbers),
            ictd_save_tp_mode="fully-connected", invariant_channels=32,
            device=torch.device(device), dtype=config.dtype,
            **long_range_options,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
        e3nn.set_optimization_defaults(**previous_optimization)


def energy_and_forces(model, positions, atomic_numbers, batch, edge_src, edge_dst,
                      edge_shifts, cell, *, training=False, owned_mask=None):
    """Evaluate a prebuilt graph through the ordinary model forward.

    Returns energy [B], forces [N,3], atomic_energy [N]. Shifts are lattice
    coordinates and cell is [B,3,3]. No neighbor construction, E0 addition or
    optimizer is performed. training=True retains force-loss mixed derivatives;
    inference returns detached results. Owned masks select energy contributions,
    while force derivatives are taken for all supplied local/ghost coordinates.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("Full forces require autograd; use no_grad instead of inference_mode")
    if training and not getattr(model, "supports_training", True):
        raise ValueError("This foundation adapter supports frozen inference; full fine-tuning is not implemented")
    if positions.ndim != 2 or positions.shape[1] != 3 or cell.ndim != 3 or cell.shape[1:] != (3, 3):
        raise ValueError("Expected positions [N,3] and cell [B,3,3]")
    n = positions.shape[0]
    if atomic_numbers.shape != (n,) or batch.shape != (n,):
        raise ValueError("Atomic numbers and graph batch must have shape [N]")
    if owned_mask is not None and (owned_mask.shape != (n,) or owned_mask.requires_grad):
        raise ValueError("owned_mask must be a fixed [N] tensor")
    scope = nullcontext()
    if positions.device.type == "npu":
        from mace_ictc.ascend.runtime import graph_queue_scope
        scope = graph_queue_scope()
    with scope, torch.enable_grad():
        pos = positions if positions.requires_grad else positions.detach().requires_grad_(True)
        output = model(pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
        atomic = (output[0] if isinstance(output, tuple) else output).reshape(n)
        selected = atomic if owned_mask is None else atomic * owned_mask.to(atomic)
        energy = selected.new_zeros(cell.shape[0]).index_add(0, batch, selected)
        # Keep the coordinate dependency for zero-edge/empty graphs as well.
        force = -torch.autograd.grad(energy.sum() + pos.sum()*0.0, pos,
                                     create_graph=training, retain_graph=training)[0]
        result = {"energy": energy, "forces": force, "atomic_energy": atomic}
        return result if training else {key: value.detach() for key, value in result.items()}



def check_model(model, *, training=False, capture=False):
    """Inspect Ascend coverage without modifying parameters or precision."""
    if next(model.parameters()).device.type == "npu":
        from mace_ictc.ascend.runtime import check_model as check_ascend
        return check_ascend(model, training=training, capture=capture)
    return {"supported": False, "backend": "ascend",
            "reasons": ["Generated operators require NPU FP32; CPU is a reference backend"]}


def configure_model(model, *, training=False, policy="auto", layout="packed",
                    capture=False, message_chunk_edges=None, electrostatics_backend="auto"):
    """Select dynamic Ascend operators or the differentiable CPU reference."""
    if layout not in ("packed", "fused_pack"):
        raise ValueError("layout must be packed or fused_pack")
    if next(model.parameters()).device.type == "npu":
        from mace_ictc.ascend.runtime import configure_model as configure_ascend
        return configure_ascend(model, training=training, policy=policy, capture=capture,
                                edge_chunk_size=message_chunk_edges,
                                electrostatics_backend=electrostatics_backend)
    from .electrostatics import configure_electrostatics
    configure_electrostatics(model, backend=electrostatics_backend)
    from mace_ictc.models.operator_plans import configure_mace_runtime_plan
    return configure_mace_runtime_plan(model, policy=policy, training=training, capture=capture)
