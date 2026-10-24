# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Whole-network entrypoints; precision and trainability stay caller-controlled."""
import torch


def create_model(*, channels=64, lmax=3, edge_lmax=None, num_interactions=2,
                 correlation=3, use_reduced_cg=False,
                 atomic_numbers=(1, 6, 8), cutoff=5.0, avg_num_neighbors=50.0,
                 device="npu", dtype=torch.float32):
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
    from .operators import _degree, _positive_integer
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
    if positions.ndim != 2 or positions.shape[1] != 3 or cell.ndim != 3 or cell.shape[1:] != (3, 3):
        raise ValueError("Expected positions [N,3] and cell [B,3,3]")
    n = positions.shape[0]
    if atomic_numbers.shape != (n,) or batch.shape != (n,):
        raise ValueError("Atomic numbers and graph batch must have shape [N]")
    if owned_mask is not None and (owned_mask.shape != (n,) or owned_mask.requires_grad):
        raise ValueError("owned_mask must be a fixed [N] tensor")
    with torch.enable_grad():
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


def _layout(value, *, optional=False):
    if not (optional and value is None) and value not in ("packed", "fused_pack"):
        raise ValueError("HP layout must be 'packed' or 'fused_pack'")


def check_model(model: torch.nn.Module, *, training=False, capture=False) -> dict:
    """Read-only coverage check; no freezing, casting, planning or compilation."""
    if next(model.parameters()).device.type == "npu":
        from mace_ictc.ascend.runtime import check_model as check_ascend
        return check_ascend(model, training=training, capture=capture)
    from mace_ictc.models.operator_plans import generated_plan_unavailable
    reasons = generated_plan_unavailable(model, training=training, capture=capture)
    return {"supported": not reasons, "reasons": reasons}


def configure_model(model: torch.nn.Module, *, training=False, policy="auto",
                    layout="packed", gate_cache_bytes=0, capture=False,
                    wide_channel_policy="auto", high_l_policy="auto",
                    reference_buffer_policy="auto",message_schedule="native",
                    graph_reduction="auto",topology_backend="native",
                    csr_accumulator_limit=None,csr_slice_search=False,
                    geometry_schedule="auto",message_chunk_edges=None,
                    species_schedule="grouped",
                    training_gate_schedule="materialized",mixed_channel_tile=None,
                    radial_checkpoint=False,inference_message_layout="native") -> dict:
    """Configure every interaction for eager training or full-force inference.

    auto selects qualified independent HP operators or reports why the existing
    route is retained. generated requires coverage; reference restores the
    reference recipe. No dtype/trainability is changed. TF32 is disabled globally.
    Wide channels default to shared species GEMMs and dense-graph CSR forward;
    off restores the original HP schedules. The inference gate-cache budget is
    bytes per interaction; training requires zero.
    training_gate_schedule defaults to materialized. cached retains forward
    gates but pages first/mixed cotangents; recompute also rebuilds gates. Both
    are opt-in eager L<=3 schedules with live radial parameters. Coordinate
    Hessians and higher derivatives retain the differentiable formula.
    A tuple/list may specify one gate schedule per interaction. radial_checkpoint
    optionally recomputes radial MLP activations with non-reentrant checkpointing.
    mixed_channel_tile optionally changes only the generated mixed-adjoint
    channel tiling (0, 32 or 64); None preserves the established schedule.
    inference_message_layout="degree" keeps TP outputs in degree-major angular
    rows through the following linear and coordinate adjoint. It is opt-in,
    eager L<=3 inference with layout="fused_pack"; frozen gate/TP weights use
    the fast route. No weights are frozen by this configuration call.
    High-L auto uses bounded sparse cubic SC and mode-specific TP schedules for
    L4/5. high_l_policy=off restores the preceding high-L HP schedules.
    reference_buffer_policy=auto keeps dense high-L cubic U on the host after
    a validated GPU plan exists; device retains conventional device buffers.
    Parameter devices/dtypes stay unchanged. Parallel wrappers/groups keep
    device buffers, and reference/export paths restore them before use.
    """
    if next(model.parameters()).device.type == "npu":
        from mace_ictc.ascend.runtime import configure_model as configure_ascend
        if (gate_cache_bytes or message_schedule != "native" or graph_reduction != "auto"
                or topology_backend != "native" or geometry_schedule != "auto"
                or csr_accumulator_limit is not None or csr_slice_search
                or training_gate_schedule != "materialized" or mixed_channel_tile is not None
                or radial_checkpoint or inference_message_layout != "native"
                or species_schedule != "grouped" or wide_channel_policy != "auto"
                or high_l_policy != "auto" or reference_buffer_policy != "auto"):
            raise ValueError("CUDA scheduling overrides are not implemented by the Ascend backend")
        return configure_ascend(model, training=training, policy=policy, capture=capture,
                                edge_chunk_size=message_chunk_edges)
    from mace_ictc.models.fp32_policy import strict_fp32
    strict_fp32()
    _layout(layout)
    if message_schedule=='compact':
        raise ValueError('The experimental STF-coordinate schedule is retired; use native or roles in the original ICTC basis')
    if message_schedule not in ('native','roles'):
        raise ValueError('message_schedule must be native or roles')
    if graph_reduction not in ('auto','atomic','csr'):
        raise ValueError('graph_reduction must be auto, atomic or csr')
    if topology_backend not in ('native','device'):
        raise ValueError('topology_backend must be native or device')
    if csr_accumulator_limit is not None and (not isinstance(csr_accumulator_limit,int) or isinstance(csr_accumulator_limit,bool) or csr_accumulator_limit<1):
        raise ValueError('csr_accumulator_limit must be a positive integer or None')
    if not isinstance(csr_slice_search,bool):raise ValueError('csr_slice_search must be a bool')
    if geometry_schedule not in ('auto','native','streamed'):
        raise ValueError('geometry_schedule must be auto, native or streamed')
    if species_schedule not in ('grouped','indexed'):
        raise ValueError('species_schedule must be grouped or indexed')
    if inference_message_layout not in ('native','degree'):
        raise ValueError('inference_message_layout must be native or degree')
    if inference_message_layout=='degree' and policy!='reference':
        if training or capture:
            raise ValueError('Degree message layout currently requires eager inference')
        if layout!='fused_pack':
            raise ValueError('Degree message layout requires layout=fused_pack')
    gate_schedules=(training_gate_schedule,) if isinstance(training_gate_schedule,str) else training_gate_schedule
    if (not isinstance(gate_schedules,(tuple,list)) or not gate_schedules
            or any(not isinstance(value,str) or value not in ('materialized','cached','recompute') for value in gate_schedules)):
        raise ValueError('training_gate_schedule must contain materialized, cached or recompute')
    paged_gates=any(value!='materialized' for value in gate_schedules)
    if not isinstance(radial_checkpoint,bool):raise ValueError('radial_checkpoint must be a bool')
    if not training and policy!='reference' and (paged_gates or radial_checkpoint):
        raise ValueError('A training gate schedule requires training=True')
    if mixed_channel_tile is not None:
        if isinstance(mixed_channel_tile,bool) or not isinstance(mixed_channel_tile,int) or mixed_channel_tile not in (0,32,64):
            raise ValueError('mixed_channel_tile must be None, 0, 32 or 64')
        if not training and policy!='reference':
            raise ValueError('A mixed-adjoint channel tile requires training=True')
    if message_chunk_edges is not None and (not isinstance(message_chunk_edges,int) or isinstance(message_chunk_edges,bool) or message_chunk_edges<1):
        raise ValueError('message_chunk_edges must be a positive integer or None')
    from mace_ictc.models.operator_plans import mace_backbone_core
    core=mace_backbone_core(model)
    interactions=tuple(layer for layer in getattr(core,'interactions',()) if hasattr(layer,'tp'))
    per_layer=not isinstance(training_gate_schedule,str)
    if per_layer and len(gate_schedules)!=len(interactions):
        raise ValueError('A per-interaction training_gate_schedule must match the interaction count')
    if radial_checkpoint and capture and policy!='reference':raise ValueError('radial_checkpoint requires eager execution')
    if (policy!='reference' and inference_message_layout=='degree' and hasattr(core,'lmax')
            and hasattr(core,'ictd_fix_edge_lmax') and max(core.lmax,core.ictd_fix_edge_lmax)>3):
        raise ValueError('Degree message layout currently requires hidden/edge L<=3')
    if (policy!='reference' and (paged_gates or radial_checkpoint or mixed_channel_tile is not None) and hasattr(core,'lmax')
            and hasattr(core,'ictd_fix_edge_lmax') and max(core.lmax,core.ictd_fix_edge_lmax)>3):
        raise ValueError('Explicit training gate/mixed schedules currently require hidden/edge L<=3')
    if (policy!='reference' and message_schedule!='native' and hasattr(core,'lmax')
            and hasattr(core,'ictd_fix_edge_lmax') and max(core.lmax,core.ictd_fix_edge_lmax)>3):
        raise ValueError('Requested low-L message schedule requires hidden/edge L<=3')
    from mace_ictc.models.operator_plans import configure_mace_runtime_plan
    info = configure_mace_runtime_plan(
        model, policy=policy, training=training, capture=capture,
        gate_cache_bytes=gate_cache_bytes, generated_layout=layout,
        wide_channel_policy=wide_channel_policy,
        high_l_policy=high_l_policy,
        reference_buffer_policy=reference_buffer_policy,
    )
    selected=info['selected']=='generated'
    from mace_ictc.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    indexed_count=0
    for module in core.modules():
        if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
            module._species_circuit=(selected and species_schedule=='indexed'
                and max(module.lmax,module.target_lmax)<=3 and module._factorized_circuit
                and not module._partitioned_circuit)
            indexed_count+=int(module._species_circuit)
    info['species_schedule']=species_schedule if selected else 'reference'
    info['counts']['indexed_species_circuits']=indexed_count
    for index,layer in enumerate(interactions):
        degree_layout=selected and not training and inference_message_layout=='degree'
        layer.tp._inference_degree_layout=degree_layout
        if hasattr(layer,'message_linear'):layer.message_linear._inference_degree_layout=degree_layout
        schedule=gate_schedules[index] if per_layer else gate_schedules[0]
        layer.tp._training_gate_schedule=schedule if selected and training else 'materialized'
        layer.tp._radial_checkpoint=bool(selected and training and radial_checkpoint)
        layer.tp._generated_second_channel_tile=mixed_channel_tile if selected and training else None
        layer.tp._generated_specialized_adjoint=selected and message_schedule!='native'
        layer.tp._generated_second_roles=selected and message_schedule!='native'
        layer.tp._generated_compact_cartesian=False
        cached=getattr(layer.tp,'_bond_projection_adjoint_plan',None)
        if cached is not None:cached[1].pop('compact_cartesian_state',None)
        if selected and graph_reduction!='auto':
            layer.tp._generated_csr_forward=graph_reduction=='csr'
        layer.tp._generated_topology=topology_backend if selected else 'native'
        layer.tp._csr_slice_search=selected and csr_slice_search
        if selected and geometry_schedule!='auto':
            layer.tp._generated_geometry_streaming=geometry_schedule=='streamed'
        if selected and message_chunk_edges is not None:
            layer.tp._message_chunk_edges=message_chunk_edges
        if selected and csr_accumulator_limit is not None:
            layer.tp._csr_accumulator_limit=csr_accumulator_limit
        # Reconfiguration must not retain rejected experimental source adjoints.
        layer.tp._generated_csr_fused_adjoint=False
    info['message_schedule']=message_schedule if selected else 'reference'
    info['training_gate_schedule']=training_gate_schedule if selected and training else 'materialized'
    info['mixed_channel_tile']=mixed_channel_tile if selected and training else None
    info['radial_checkpoint']=bool(selected and training and radial_checkpoint)
    info['inference_message_layout']=inference_message_layout if selected and not training else 'native'
    info['counts']['degree_packed_inference']=sum(bool(getattr(layer.tp,'_inference_degree_layout',False))
                                                for layer in interactions)
    info['counts']['paged_training_gates']=sum(
        getattr(layer.tp,'_training_gate_schedule','materialized')!='materialized'
        for layer in getattr(core,'interactions',()) if hasattr(layer,'tp'))
    info.update(graph_reduction=graph_reduction if selected else 'reference',
                topology_backend=topology_backend if selected else 'native',
                csr_accumulator_limit=csr_accumulator_limit,csr_slice_search=selected and csr_slice_search)
    info.update(geometry_schedule=geometry_schedule if selected else 'reference',message_chunk_edges=message_chunk_edges)
    if selected:
        info['counts']['generated_csr_forward']=sum(bool(getattr(layer.tp,'_generated_csr_forward',False))
            for layer in getattr(core,'interactions',()) if hasattr(layer,'tp'))
    return {**info, "layout": layout if info["selected"] == "generated" else None}


def prepare_for_export(model: torch.nn.Module, *, policy="auto", layout=None, dynamic=True,
                       reference_buffer_policy="auto") -> dict:
    """Prepare a frozen inference model before FX/AOTInductor export.

    None preserves an explicit layout, or uses packed on a fresh model.
    checkpoint preserves the checkpoint route. This API prepares, but does not
    export, a package; existing .pt2 binaries must be re-exported to change kernels.
    Qualified high-L cor3 keeps canonical U on CPU after plan preparation;
    reference_buffer_policy="device" retains conventional device buffers.
    """
    from mace_ictc.models.fp32_policy import strict_fp32
    strict_fp32()
    _layout(layout, optional=True)
    if dynamic and policy != "checkpoint":
        from mace_ictc.models.operator_plans import generated_plan_unavailable,mace_backbone_core
        core = mace_backbone_core(model)
        if not generated_plan_unavailable(core):
            scripted = [name for name, child in core.named_modules() if isinstance(child, torch.jit.ScriptModule)]
            if scripted:
                raise ValueError("Dynamic AOTI requires FX e3nn submodules: use create_model, or set "
                    "e3nn.set_optimization_defaults(jit_script_fx=False) before rebuilding/loading the model. "
                    f"Scripted modules remain: {scripted[:3]}. Use dynamic=False only for an intentionally static-N package.")
    from mace_ictc.interfaces.deployment_triton import configure_mace_deployment_plan
    return configure_mace_deployment_plan(model, policy=policy, generated_layout=layout,
                                         reference_buffer_policy=reference_buffer_policy)
