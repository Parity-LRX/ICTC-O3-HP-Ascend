"""Persistent eager Ascend engine for the plain LAMMPS mff/torch pair style.

MPI owns host topology and ownership metadata. Model force buffers remain on the
NPU during HCCL exchange and owner accumulation. No ML-IAP interface is used.
"""
from __future__ import annotations

from datetime import timedelta
import inspect
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_npu


def load_checkpoint(path, device):
    """Rebuild a trusted FP32 short-range ICTC checkpoint without deployment tracing."""
    import e3nn
    from mace_ictc.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
    from mace_ictc.utils.checkpoint_metadata import get_checkpoint_e3_state_dict
    from mace_ictc.utils.config import ModelConfig

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a MACE-ICTC checkpoint dictionary')
    meta = dict(checkpoint.get('model_hyperparameters', {}))
    mode = checkpoint.get('tensor_product_mode', meta.get('tensor_product_mode'))
    if mode != 'pure-cartesian-ictd-fix':
        raise ValueError('Ascend LAMMPS requires a pure-cartesian-ictd-fix checkpoint')
    dtype = str(checkpoint.get('dtype', meta.get('dtype', ''))).replace('torch.', '')
    if dtype not in ('float32', 'float'):
        raise ValueError('Ascend LAMMPS requires FP32 weights; precision is not changed automatically')
    for key in ('channel_in', 'lmax', 'num_interaction', 'avg_num_neighbors',
                'radial_sqrt_num_basis', 'polynomial_cutoff_p', 'ictd_fix_product_backend'):
        if key not in meta and key not in checkpoint:
            raise ValueError(f'Missing model metadata: {key}; re-save with the training configuration')
    values = {**meta, **checkpoint}
    if values.get('angular_basis', 'ictd') != 'ictd':
        raise ValueError('Ascend LAMMPS currently requires the ICTC angular basis')
    if (values.get('long_range_mode', 'none') != 'none'
            or values.get('long_range_dispersion_mode', 'none') != 'none'
            or values.get('long_range_dispersion', False)):
        raise ValueError('This Ascend pair style supports short-range checkpoints only')
    if values.get('zbl_enabled', False):
        raise ValueError('A separate ZBL energy contribution is not implemented by this pair style')
    if values.get('feature_spectral_mode', 'none') != 'none' or values.get('num_fidelity_levels', 0):
        raise ValueError('Spectral and multi-fidelity checkpoints require a separate deployment implementation')
    keys = checkpoint.get('atomic_energy_keys')
    offsets = checkpoint.get('atomic_energy_values')
    if keys is None or offsets is None:
        raise ValueError('Checkpoint must contain explicit atomic_energy_keys and atomic_energy_values')
    keys = torch.as_tensor(keys, dtype=torch.int64).tolist()
    offsets = torch.as_tensor(offsets, dtype=torch.float32)
    if len(keys) != offsets.numel() or len(set(keys)) != len(keys):
        raise ValueError('Invalid checkpoint atomic energy mapping')
    radius = float(checkpoint.get('max_radius', values.get('max_radius', 0)))
    if radius <= 0:
        raise ValueError('Checkpoint cutoff must be positive')
    state, state_source = get_checkpoint_e3_state_dict(checkpoint)
    mapping = state.get('atomic_number_to_index')
    species = (torch.where(mapping >= 0)[0].tolist() if mapping is not None else keys)
    if not set(species).issubset(keys):
        raise ValueError('Atomic reference energies must cover every trained element')
    cfg_keys = inspect.signature(ModelConfig).parameters
    config_values = {k: v for k, v in values.items() if k in cfg_keys}
    config_values.update(dtype=torch.float32, internal_compute_dtype=torch.float32, max_radius=radius)
    previous_dtype, previous_options = torch.get_default_dtype(), e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(jit_script_fx=False)
        cfg = ModelConfig(**config_values)
        args = dict(max_embed_radius=radius, main_max_radius=cfg.max_radius_main,
                    main_number_of_basis=cfg.number_of_basis_main, hidden_dim_conv=cfg.channel_in,
                    hidden_dim_sh=cfg.get_hidden_dim_sh(), hidden_dim=cfg.emb_number_main_2,
                    channel_in2=cfg.channel_in2, embedding_dim=cfg.embedding_dim,
                    max_atomvalue=cfg.max_atomvalue, atomic_numbers=species,
                    output_size=cfg.output_size, embed_size=cfg.embed_size,
                    main_hidden_sizes3=cfg.main_hidden_sizes3, num_layers=cfg.num_layers,
                    function_type_main=cfg.function_type, lmax=cfg.lmax,
                    internal_compute_dtype=torch.float32, device='cpu')
        parameters = inspect.signature(PureCartesianICTDFix).parameters
        args.update({k: v for k, v in values.items() if k in parameters})
        args.update(atomic_numbers=species, device='cpu', internal_compute_dtype=torch.float32)
        model = PureCartesianICTDFix(**args).to(dtype=torch.float32)
        if 'mace_first_layer_sc0' in state:
            model.install_mace_first_layer_sc0(state['mace_first_layer_sc0'])
        for name, parameter in model.named_parameters():
            if name in state and state[name].dtype != torch.float32:
                raise ValueError(f'Checkpoint parameter {name} is not FP32')
        model.load_state_dict(state, strict=True)
    finally:
        torch.set_default_dtype(previous_dtype)
        e3nn.set_optimization_defaults(**previous_options)
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('All model parameters must remain FP32')
    e0 = torch.zeros(max(keys) + 1, device=device, dtype=torch.float32)
    e0[torch.tensor(keys, device=device)] = offsets.to(device)
    return model, e0, state_source


def _routing_plan(source, destination, outputs, device, cache, vector):
    """Share fixed coefficients/UB metadata while keeping live routing separate."""
    key = str(device), vector
    template = cache.get(key)
    if template is None:
        template = dict(sizes=(3, 1, 1, 3) if vector else (1, 1, 1, 1),
                        entries=[(i, 0, 0, i, 1.) for i in range(3)] if vector else [(0, 0, 0, 0, 1.)],
                        _ascendc_metadata={}, _ascendc_tiling={})
        cache[key] = template
    return dict(template, _ascendc_graph=dict(src=source, dst=destination, mask=None,
                                            destination_nodes=outputs))


class _VectorSum:
    """Gather and coalesce force rows with the existing Ascend C graph map."""
    def __init__(self, source, destination, outputs, device, cache=None):
        self.rows = source.numel()
        self.outputs = outputs
        self.plan = _routing_plan(source, destination, outputs, device,
                                  {} if cache is None else cache, True)
        self.right = torch.ones(self.rows, 1, 1, device=device, dtype=torch.float32)
        self.weight = torch.ones(1, 1, 1, device=device, dtype=torch.float32)

    def __call__(self, value):
        if not self.rows:
            return value.new_zeros(self.outputs, 3)
        from mace_ictc.ascendc.product import raw
        return raw(value.contiguous().reshape(-1, 1, 3), self.right, self.weight,
                   None, self.plan, True).reshape(self.outputs, 3)


class _RowSum:
    """Gather/sum arbitrary feature widths through native channel tiling."""
    def __init__(self, source, destination, outputs, width, device, cache=None):
        self.rows, self.outputs, self.width = source.numel(), outputs, width
        self.plan = _routing_plan(source, destination, outputs, device,
                                  {} if cache is None else cache, False)
        self.right = torch.ones(self.rows, 1, 1, device=device, dtype=torch.float32)
        self.weight = torch.ones(1, width, 1, device=device, dtype=torch.float32)

    def __call__(self, value):
        if not self.rows:return value.new_zeros(self.outputs, self.width)
        from mace_ictc.ascendc.product import raw
        return raw(value.contiguous().reshape(-1, self.width, 1), self.right,
                   self.weight, None, self.plan, True).reshape(self.outputs, self.width)


class _FeatureHalo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, anchor, engine, slot):
        ctx.engine, ctx.generation, ctx.slot = engine, engine.generation, slot
        return engine._feature_forward(value, slot)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        if ctx.generation != ctx.engine.generation:
            raise RuntimeError('Feature-halo topology changed before its adjoint completed')
        return ctx.engine._feature_reverse(gradient, ctx.slot), gradient.new_zeros(()), None, None


class AscendLAMMPSEngine:
    def __init__(self, checkpoint, cutoff, device_index, rank, world_size, master, port):
        if os.environ.get('MFF_ASCEND_EXECUTION', 'eager') != 'eager':
            raise ValueError('The Ascend LAMMPS backend supports dynamic eager execution only')
        self.execution = 'eager'
        torch.set_num_threads(int(os.environ.get('MFF_TORCH_THREADS', '2')))
        if not 0 <= device_index < torch.npu.device_count():
            raise ValueError(f'Local rank {device_index} has no visible NPU; use one MPI rank per allocated NPU')
        self.device = torch.device(f'npu:{device_index}')
        torch.npu.set_device(self.device)
        from mace_ictc.ascendc import enable
        from mace_ictc.hp import configure_model
        enable(mode=os.environ.get('MFF_ASCEND_MODE', 'performance'))
        self.model, self.e0, self.state_source = load_checkpoint(checkpoint, self.device)
        self.cutoff = float(self.model.max_radius)
        self.depth = int(self.model.num_interaction)
        self.halo_mode = os.environ.get('MFF_ASCEND_HALO',
                                       'layerwise' if world_size > 1 else 'wide')
        if self.halo_mode not in ('wide', 'layerwise'):
            raise ValueError('MFF_ASCEND_HALO must be wide or layerwise')
        owner_linear = os.environ.get('MFF_ASCEND_OWNER_LINEAR', '0')
        if owner_linear not in ('0', '1'):
            raise ValueError('MFF_ASCEND_OWNER_LINEAR must be zero or one')
        self.owner_linear = owner_linear == '1'
        gather = os.environ.get('MFF_ASCEND_FEATURE_GATHER', 'native')
        if gather not in ('native', 'index_select'):
            raise ValueError('MFF_ASCEND_FEATURE_GATHER must be native or index_select')
        self.feature_gather = gather
        prune = os.environ.get('MFF_ASCEND_PRUNE_SKIN', '0')
        if prune not in ('0', '1'):
            raise ValueError('MFF_ASCEND_PRUNE_SKIN must be zero or one')
        self.prune_skin = prune == '1'
        if self.prune_skin and getattr(self.model, 'assume_edges_within_radius', False):
            raise ValueError('Skin pruning requires the model cutoff mask to remain enabled')
        compact = os.environ.get('MFF_ASCEND_COMPACT_OUTPUTS', '0')
        if compact not in ('0', '1'):
            raise ValueError('MFF_ASCEND_COMPACT_OUTPUTS must be zero or one')
        self.compact_outputs = compact == '1' and self.halo_mode == 'layerwise' and world_size > 1
        overlap = os.environ.get('MFF_ASCEND_OVERLAP_FEATURES', '0')
        if overlap not in ('0', '1'):
            raise ValueError('MFF_ASCEND_OVERLAP_FEATURES must be zero or one')
        if overlap == '1' and not self.owner_linear:
            raise ValueError('Feature overlap requires MFF_ASCEND_OWNER_LINEAR=1')
        self.overlap_features = (overlap == '1' and self.halo_mode == 'layerwise'
                                 and world_size > 1 and self.depth > 1)
        self._feature_stream = torch.npu.Stream(device=self.device) if self.overlap_features else None
        self._feature_pending = set()
        owner_output = os.environ.get('MFF_ASCEND_OWNER_OUTPUT', '0')
        if owner_output not in ('0', '1'):
            raise ValueError('MFF_ASCEND_OWNER_OUTPUT must be zero or one')
        self.owner_output = owner_output == '1' and self.halo_mode == 'layerwise' and world_size > 1
        self.halo_depth = 1 if self.halo_mode == 'layerwise' else self.depth
        if abs(float(cutoff) - self.cutoff) > 1e-6:
            raise ValueError(f'Pair cutoff {cutoff} does not match checkpoint cutoff {self.cutoff}')
        self.configuration = configure_model(self.model, policy='generated', training=False)
        self.rank, self.world_size = rank, world_size
        self.group = None
        if world_size > 1:
            if dist.is_initialized():
                raise RuntimeError('A pre-existing distributed process group is not supported by this engine')
            dist.init_process_group('hccl', init_method=f'tcp://{master}:{port}',
                                    rank=rank, world_size=world_size,
                                    timeout=timedelta(seconds=int(os.environ.get('MFF_HCCL_TIMEOUT', '300'))))
            self.group = dist.group.WORLD
            dist.barrier(group=self.group, device_ids=[device_index])
        self.statistics = dict(calls=0, topology_updates=0, model_seconds=0.,
                               hccl_seconds=0., transfer_seconds=0., sent_bytes=0,
                               received_bytes=0, p2p_operations=0, max_owned_nodes=0,
                               max_model_nodes=0, max_edges=0, max_ghost_rows=0,
                               max_packed_rows=0, timed_calls=0,
                               force_p2p_operations=0, feature_forward_p2p_operations=0,
                               force_sent_bytes=0, force_received_bytes=0,
                               feature_reverse_p2p_operations=0, feature_sent_bytes=0,
                               feature_received_bytes=0, feature_forward_seconds=0.,
                               feature_reverse_seconds=0., edge_filter_seconds=0.,
                               edge_filter_updates=0, edge_candidates=0, edge_retained=0,
                               max_active_edges=0)
        self.initial_allocated_bytes = torch.npu.memory_allocated(self.device)
        torch.npu.reset_peak_memory_stats(self.device)
        self.profile = os.environ.get('MFF_ASCEND_PROFILE', '0') == '1'
        self.timing_skip = int(os.environ.get('MFF_ASCEND_PROFILE_SKIP', '0'))
        if self.timing_skip < 0:raise ValueError('MFF_ASCEND_PROFILE_SKIP must be nonnegative')
        self.generation = None
        self._routing_templates = {}
        self._owned_hooks = []
        self._owned_methods = []
        self._owned_metadata_views = {}
        self._owned_destination_counts = []
        if self.owner_output:
            for layer in self.model.interactions:
                module = layer.tp
                self._owned_destination_counts.append((module,
                    '_ascend_destination_nodes' in module.__dict__,
                    module.__dict__.get('_ascend_destination_nodes')))
        if self.halo_mode == 'layerwise' and world_size > 1:
            import weakref
            reference = weakref.ref(self)
            def owned_inputs(module, args, kwargs):
                engine = reference()
                return engine._owned_inputs(args, kwargs)
            def owned_output(module, args, output):
                engine = reference()
                if engine.compact_outputs and module not in engine.model.products:return output
                return engine._pad_owned(output)
            # Products and energy readouts act independently on each node.
            # Ghost products are replaced by owner features before the next
            # interaction; ghost readouts never contribute to owned energy.
            modules = list(self.model.products) + list(self.model.layer_energy_readouts)
            modules.append(self.model.last_layer_energy_readout)
            modules.extend(layer.message_linear for layer in self.model.interactions)
            for module in modules:
                self._owned_hooks.append(module.register_forward_pre_hook(owned_inputs, with_kwargs=True))
                self._owned_hooks.append(module.register_forward_hook(owned_output))
            for index, layer in enumerate(self.model.interactions):
                direct = [layer.self_connection]
                if self.owner_linear or self.compact_outputs:direct.append(layer.message_selector)
                for module in direct:
                    if module is None:continue
                    # These paths call forward_type_idx directly, bypassing
                    # Module hooks. Restore any existing instance override.
                    local = 'forward_type_idx' in module.__dict__
                    previous = module.__dict__.get('forward_type_idx')
                    original = module.forward_type_idx
                    def owned_direct(*args, _original=original, **kwargs):
                        engine = reference()
                        args, kwargs = engine._owned_inputs(args, kwargs)
                        output = _original(*args, **kwargs)
                        return output if engine.compact_outputs else engine._pad_owned(output)
                    self._owned_methods.append((module, local, previous))
                    module.forward_type_idx = owned_direct
                if self.owner_linear and index:
                    self._owned_hooks.append(layer.linear_up.register_forward_pre_hook(
                        owned_inputs, with_kwargs=True))
                    def exchange_linear(module, args, output, _slot=index-1):
                        engine = reference()
                        output = engine._pad_owned(output)
                        return engine._exchange_state(output, _slot)
                    self._owned_hooks.append(layer.linear_up.register_forward_hook(exchange_linear))
                    if self.overlap_features:
                        def join_features(module, args, output, _slot=index-1):
                            reference()._join_features(_slot)
                        self._owned_hooks.append(layer.fc.register_forward_hook(join_features))

    def _pad_owned(self, output):
        if self.nodes == self.nlocal:return output
        return torch.cat((output, output.new_zeros(
            (self.nodes - self.nlocal,) + output.shape[1:])))

    def _owned_inputs(self, args, kwargs):
        if self.nodes == self.nlocal:return args, kwargs
        def owned(value):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == self.nodes:
                if value.dtype in (torch.int32, torch.int64):
                    # Stable species views retain validation and native bank
                    # selection caches. Never cache floating learned features.
                    key = id(value), value._version, tuple(value.shape)
                    cached = self._owned_metadata_views.get(key)
                    if cached is not None and cached[0]() is value:return cached[1]
                    import weakref
                    view = value[:self.nlocal]
                    validated = getattr(value, '_ictc_ascend_valid_species', None)
                    if validated is not None and validated[1] == value._version:
                        view._ictc_ascend_valid_species = validated[0], view._version
                    self._owned_metadata_views[key] = weakref.ref(value), view
                    return view
                return value[:self.nlocal]
            if isinstance(value, dict):return {key:owned(item) for key,item in value.items()}
            if isinstance(value, tuple):return tuple(owned(item) for item in value)
            if isinstance(value, list):return [owned(item) for item in value]
            return value
        return tuple(owned(item) for item in args), {key:owned(item) for key,item in kwargs.items()}

    @staticmethod
    def _host(buffer, dtype, shape):
        return torch.from_numpy(np.frombuffer(buffer, dtype=dtype).reshape(shape))

    def topology(self, nlocal, nodes, atomic_numbers, src, dst, shifts,
                 ghost_nodes, ghost_slots, receive_indices, send_counts, receive_counts, generation):
        if self._feature_pending:
            raise RuntimeError('Topology changed during an unfinished feature exchange')
        device = self.device
        self.nlocal, self.nodes = int(nlocal), int(nodes)
        if self.owner_output:
            destination = np.frombuffer(dst, dtype=np.int64)
            if destination.size and (destination.min() < 0 or destination.max() >= nlocal):
                raise ValueError('Layerwise graph destinations must be owned atoms')
            for module, _, _ in self._owned_destination_counts:
                module._ascend_destination_nodes = self.nlocal
        self.atomic_numbers = self._host(atomic_numbers, np.int64, (nodes,)).to(device)
        species = set(self.model.atomic_numbers)
        if not set(np.frombuffer(atomic_numbers, dtype=np.int64)).issubset(species):
            raise ValueError('LAMMPS atom types include elements absent from the checkpoint')
        self.src = self._host(src, np.int64, (-1,)).to(device)
        self.dst = self._host(dst, np.int64, (-1,)).to(device)
        self.shifts = self._host(shifts, np.float32, (-1, 3)).to(device)
        if self.prune_skin:
            # The CPU integrator already owns these buffers. Keep immutable
            # copies across calls; LAMMPS may replace its vectors on a rebuild.
            self._skin_edges = (np.frombuffer(src, np.int64).copy(),
                                np.frombuffer(dst, np.int64).copy(),
                                np.frombuffer(shifts, np.float32).reshape(-1, 3).copy())
            self._skin_has_shifts = bool(np.any(self._skin_edges[2]))
            self._active_edges = None
        self.batch = torch.zeros(nodes, dtype=torch.int64, device=device)
        ghosts = self._host(ghost_nodes, np.int64, (-1,)).to(device)
        slots = self._host(ghost_slots, np.int64, (-1,)).to(device)
        receivers = self._host(receive_indices, np.int64, (-1,)).to(device)
        self.send_counts = np.frombuffer(send_counts, dtype=np.int64).tolist()
        self.receive_counts = np.frombuffer(receive_counts, dtype=np.int64).tolist()
        self.send_offsets, self.receive_offsets = [0], [0]
        for n in self.send_counts:self.send_offsets.append(self.send_offsets[-1] + n)
        for n in self.receive_counts:self.receive_offsets.append(self.receive_offsets[-1] + n)
        self.pack = _VectorSum(ghosts, slots, self.send_offsets[-1], device, self._routing_templates)
        self.accumulate = _VectorSum(torch.arange(receivers.numel(), device=device), receivers, nlocal, device, self._routing_templates)
        self.receive_buffer = torch.empty(receivers.numel(), 3, device=device, dtype=torch.float32)
        self.ghost_nodes, self.ghost_slots, self.receive_indices = ghosts, slots, receivers
        self.feature_plans = {}
        self._owned_metadata_views.clear()
        self.generation = generation
        self.statistics['topology_updates'] += 1
        for key, value in (('max_owned_nodes', nlocal), ('max_model_nodes', nodes),
                           ('max_edges', self.src.numel()), ('max_ghost_rows', ghosts.numel()),
                           ('max_packed_rows', self.send_offsets[-1])):
            self.statistics[key] = max(self.statistics[key], int(value))

    def _filter_skin(self, positions, cell):
        """Keep every edge that can be inside the FP32 device cutoff."""
        started = time.perf_counter()
        source, destination, shifts = self._skin_edges
        coordinates = np.frombuffer(positions, np.float32).reshape(-1, 3).astype(np.float64)
        left, right = coordinates[source], coordinates[destination]
        delta = right - left
        bound = np.abs(right) + np.abs(left)
        if self._skin_has_shifts:
            box = np.frombuffer(cell, np.float32).reshape(3, 3).astype(np.float64)
            delta += shifts.astype(np.float64) @ box
            bound += np.abs(shifts).astype(np.float64) @ np.abs(box)
        # Bound subtract/dot/norm roundoff conservatively, including periodic
        # cancellation at large absolute coordinates. The device still applies
        # its original cutoff mask to the retained edges.
        margin = 32 * np.finfo(np.float32).eps * (bound.sum(axis=1) + self.cutoff)
        distance2 = np.einsum('ij,ij->i', delta, delta)
        retained = (distance2 <= (self.cutoff + margin)**2)
        retained |= ~np.isfinite(distance2) | ~np.isfinite(margin)
        retained |= bound.max(axis=1, initial=0.) > np.finfo(np.float32).max / 16
        if self._active_edges is None or not np.array_equal(retained, self._active_edges):
            self.src = torch.from_numpy(source[retained].copy()).to(self.device)
            self.dst = torch.from_numpy(destination[retained].copy()).to(self.device)
            self.shifts = torch.from_numpy(shifts[retained].copy()).to(self.device)
            self._active_edges = retained
            self.statistics['edge_filter_updates'] += 1
        count = self.src.numel()
        self.statistics['edge_candidates'] += source.size
        self.statistics['edge_retained'] += count
        self.statistics['max_active_edges'] = max(self.statistics['max_active_edges'], count)
        self.statistics['edge_filter_seconds'] += time.perf_counter() - started

    def _reverse(self, forces):
        if not self.pack.rows and not self.receive_buffer.numel():
            return forces[:self.nlocal]
        send = self.pack(forces)
        receive = self.receive_buffer
        operations = []
        for peer in range(self.world_size):
            a, b = self.send_offsets[peer:peer + 2]
            c, d = self.receive_offsets[peer:peer + 2]
            if peer == self.rank:
                if b - a != d - c:
                    raise RuntimeError('Inconsistent self-ghost ownership schedule')
                if b > a:receive[c:d].copy_(send[a:b])
                continue
            if b > a:
                operations.append(dist.P2POp(dist.isend, send[a:b], peer, self.group))
            if d > c:
                operations.append(dist.P2POp(dist.irecv, receive[c:d], peer, self.group))
        if operations:
            if self.group is None:
                raise RuntimeError('Model ghost exchange requires an HCCL process group')
            self._wait_p2p(operations)
        self.statistics['p2p_operations'] += len(operations)
        self.statistics['force_p2p_operations'] += len(operations)
        for prefix, counts in (('sent', self.send_counts), ('received', self.receive_counts)):
            count = 12 * sum(n for p, n in enumerate(counts) if p != self.rank)
            self.statistics[f'{prefix}_bytes'] += count
            self.statistics[f'force_{prefix}_bytes'] += count
        return forces[:self.nlocal] + self.accumulate(receive)

    def _wait_p2p(self, operations):
        for request in dist.batch_isend_irecv(operations):request.wait()

    def _feature_plan(self, width, slot=0):
        key = int(slot), int(width)
        if key not in self.feature_plans:
            device = self.device
            ghost_rows = self.ghost_nodes.numel()
            received_rows = self.receive_indices.numel()
            ghost_order = torch.arange(ghost_rows, device=device)
            received_order = torch.arange(received_rows, device=device)
            self.feature_plans[key] = dict(
                owner_gather=_RowSum(self.receive_indices, received_order, received_rows, width, device, self._routing_templates),
                ghost_gather=_RowSum(self.ghost_slots, ghost_order, ghost_rows, width, device, self._routing_templates),
                ghost_pack=_RowSum(self.ghost_nodes, self.ghost_slots, self.send_offsets[-1], width, device, self._routing_templates),
                owner_sum=_RowSum(received_order, self.receive_indices, self.nlocal, width, device, self._routing_templates),
                forward_buffer=torch.empty(self.send_offsets[-1], width, device=device, dtype=torch.float32),
                reverse_buffer=torch.empty(received_rows, width, device=device, dtype=torch.float32))
        return self.feature_plans[key]

    def _feature_exchange(self, send, receive, forward):
        send_offsets = self.receive_offsets if forward else self.send_offsets
        receive_offsets = self.send_offsets if forward else self.receive_offsets
        operations = []
        sent_rows = received_rows = 0
        for peer in range(self.world_size):
            a, b = send_offsets[peer:peer + 2]
            c, d = receive_offsets[peer:peer + 2]
            if peer == self.rank:
                if b - a != d - c:raise RuntimeError('Inconsistent self feature-halo schedule')
                if b > a:receive[c:d].copy_(send[a:b])
                continue
            if b > a:
                operations.append(dist.P2POp(dist.isend, send[a:b], peer, self.group))
                sent_rows += b - a
            if d > c:
                operations.append(dist.P2POp(dist.irecv, receive[c:d], peer, self.group))
                received_rows += d - c
        if operations:
            if self.group is None:raise RuntimeError('Feature-halo exchange requires HCCL')
            self._wait_p2p(operations)
        sent_bytes, received_bytes = 4 * send.shape[1] * sent_rows, 4 * send.shape[1] * received_rows
        self.statistics['p2p_operations'] += len(operations)
        key = 'feature_forward_p2p_operations' if forward else 'feature_reverse_p2p_operations'
        self.statistics[key] += len(operations)
        for prefix, value in (('sent', sent_bytes), ('received', received_bytes)):
            self.statistics[f'{prefix}_bytes'] += value
            self.statistics[f'feature_{prefix}_bytes'] += value

    def _feature_forward_sync(self, value, plan):
        if self.feature_gather == 'index_select':
            send = value.index_select(0, self.receive_indices)
        else:send = plan['owner_gather'](value)
        self._feature_exchange(send, plan['forward_buffer'], True)
        if self.feature_gather == 'index_select':
            ghost = plan['forward_buffer'].index_select(0, self.ghost_slots)
        else:ghost = plan['ghost_gather'](plan['forward_buffer'])
        return torch.cat((value[:self.nlocal], ghost))

    def _feature_forward(self, value, slot=0):
        if self.profile:torch.npu.synchronize()
        started = time.perf_counter()
        plan = self._feature_plan(value.shape[1], slot)
        if self.overlap_features:
            if slot in self._feature_pending:
                raise RuntimeError('A feature slot was reused before its exchange completed')
            main = torch.npu.current_stream(self.device)
            stream = self._feature_stream
            stream.wait_stream(main)
            value.record_stream(stream)
            with torch.npu.stream(stream):
                result = self._feature_forward_sync(value, plan)
            result.record_stream(main)
            self._feature_pending.add(slot)
        else:result = self._feature_forward_sync(value, plan)
        if self.profile:torch.npu.synchronize()
        if self.statistics['calls'] >= self.timing_skip:
            self.statistics['feature_forward_seconds'] += time.perf_counter() - started
        return result

    def _join_features(self, slot):
        if slot in self._feature_pending:
            # linear_up has returned only views of the exchanged features so far.
            # Its independent radial network is now enqueued on the main stream.
            # Join before the tensor product first reads those feature values.
            torch.npu.current_stream(self.device).wait_stream(self._feature_stream)
            self._feature_pending.remove(slot)

    def _feature_reverse(self, gradient, slot=0):
        if self.profile:torch.npu.synchronize()
        started = time.perf_counter()
        plan = self._feature_plan(gradient.shape[1], slot)
        self._feature_exchange(plan['ghost_pack'](gradient), plan['reverse_buffer'], False)
        owned = gradient[:self.nlocal] + plan['owner_sum'](plan['reverse_buffer'])
        result = torch.cat((owned, gradient.new_zeros(self.nodes - self.nlocal, gradient.shape[1])))
        if self.profile:torch.npu.synchronize()
        if self.statistics['calls'] >= self.timing_skip:
            self.statistics['feature_reverse_seconds'] += time.perf_counter() - started
        return result

    def _layer_state(self, layer, value):
        if layer == self.depth - 1 or self.owner_linear:return value
        return self._exchange_state(value, layer)

    def _exchange_state(self, value, slot=0):
        if not self.ghost_nodes.numel() and not self.receive_indices.numel():return value
        width = value.numel() // self.nodes
        # Keep the exchange in the requested position/strain autograd graph even
        # on ranks whose local feature happens to be coordinate-independent.
        flat = value.reshape(self.nodes, width)
        return _FeatureHalo.apply(flat, self._halo_anchor, self, slot).reshape(value.shape)

    def _device_prediction(self, pos, box, want_virial):
        with torch.enable_grad():
            pos.requires_grad_(True)
            edge_vec = None
            if self.nodes:
                from mace_ictc.ascend.geometry import edge_vectors
                edge_vec = edge_vectors(pos, box, self.shifts, self.batch, self.src, self.dst)
                if edge_vec is None:
                    edge_cells = box[self.batch[self.src]]
                    shifts = torch.einsum('ni,nij->nj', self.shifts, edge_cells)
                    edge_vec = pos[self.dst] - pos[self.src] + shifts
                callback = None
                if self.halo_mode == 'layerwise' and self.world_size > 1 and self.depth > 1:
                    self._halo_anchor = pos.sum()*0
                    callback = self._layer_state
                prediction = self.model(pos, self.atomic_numbers, self.batch, self.src, self.dst, self.shifts, box,
                                        precomputed_edge_vec=edge_vec, layer_state_callback=callback)
                if self._feature_pending:
                    raise RuntimeError('The model did not join its pending feature exchanges')
                energy_nodes = self.nlocal if self.compact_outputs else self.nodes
                atomic = (prediction[0] if isinstance(prediction, tuple) else prediction).reshape(energy_nodes)
                atomic = atomic + self.e0[self.atomic_numbers[:energy_nodes]]
                energy = atomic[:self.nlocal].sum()
            else:
                atomic = pos.new_empty(0)
                energy = pos.sum() * 0
            variables = (pos, edge_vec) if want_virial and self.nodes else (pos,)
            connected = energy + pos.sum()*0
            if want_virial and self.nodes:connected = connected + edge_vec.sum()*0
            gradients = torch.autograd.grad(connected,
                                            variables, allow_unused=True)
            self._halo_anchor = None
        forces = -gradients[0].detach()
        if want_virial and self.nodes and self.src.numel():
            # Homogeneous strain transforms every periodic edge as r -> r T.
            # Its derivative is r.T @ dE/dr. Using short edge displacements
            # avoids cancellation between large absolute-position/cell terms.
            tensor = -(edge_vec.detach().T @ gradients[1].detach())
            tensor = (tensor + tensor.T) * .5
            virial = torch.stack((tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                  tensor[0, 1], tensor[0, 2], tensor[1, 2]))
        else:virial = pos.new_zeros(6)
        return forces, atomic, energy, virial

    def compute(self, positions, cell, want_virial):
        if self.generation is None:raise RuntimeError('Model topology has not been supplied')
        if self.profile:torch.npu.synchronize()
        start = time.perf_counter()
        if self.prune_skin:self._filter_skin(positions, cell)
        pos = self._host(positions, np.float32, (self.nodes, 3)).to(self.device)
        box = self._host(cell, np.float32, (1, 3, 3)).to(self.device)
        if self.profile:torch.npu.synchronize()
        transferred = time.perf_counter()
        forces, atomic, energy, virial = self._device_prediction(pos, box, bool(want_virial))
        if self.profile:torch.npu.synchronize()
        evaluated = time.perf_counter()
        with torch.no_grad():
            owned_forces = self._reverse(forces)
        if self.profile:torch.npu.synchronize()
        communicated = time.perf_counter()
        # One device-to-host transfer returns all outputs consumed by the CPU
        # integrator. No ghost force payload crosses this boundary.
        with torch.no_grad():
            packed = torch.cat((owned_forces.reshape(-1), atomic[:self.nlocal],
                                energy.reshape(1), virial)).cpu().numpy()
        n = self.nlocal
        result = (float(packed[4*n]), packed[:3*n].reshape(n, 3),
                  packed[3*n:4*n], packed[4*n+1:])
        stopped = time.perf_counter()
        if self.statistics['calls'] >= self.timing_skip:
            self.statistics['timed_calls'] += 1
            self.statistics['model_seconds'] += evaluated - transferred
            self.statistics['hccl_seconds'] += communicated - evaluated
            self.statistics['transfer_seconds'] += transferred - start + stopped - communicated
        self.statistics['calls'] += 1
        return result

    def close(self):
        for module, local, previous in self._owned_destination_counts:
            if local:module._ascend_destination_nodes = previous
            elif '_ascend_destination_nodes' in module.__dict__:del module._ascend_destination_nodes
        self._owned_destination_counts.clear()
        for handle in self._owned_hooks:handle.remove()
        self._owned_hooks.clear()
        for module, local, previous in self._owned_methods:
            if local:module.forward_type_idx = previous
            else:del module.forward_type_idx
        self._owned_methods.clear()
        self._owned_metadata_views.clear()
        destination = os.environ.get('MFF_ASCEND_STATS')
        if destination:
            import json
            from pathlib import Path
            path = Path(destination)
            path.mkdir(parents=True, exist_ok=True)
            (path/f'ascend_rank{self.rank}.json').write_text(json.dumps(
                dict(rank=self.rank, world_size=self.world_size, state_source=self.state_source,
                     execution=self.execution,
                     p2p_mode=getattr(self, 'p2p_mode', 'coalesced'),
                     halo_mode=self.halo_mode, halo_depth=self.halo_depth, interaction_depth=self.depth,
                     owner_linear=self.owner_linear,
                     owner_output=self.owner_output,
                     feature_gather=self.feature_gather, overlap_features=self.overlap_features,
                     compact_outputs=self.compact_outputs, prune_skin=self.prune_skin,
                     device_index=self.device.index, synchronized_timings=self.profile,
                     timing_skip=self.timing_skip,
                     initial_allocated_bytes=self.initial_allocated_bytes,
                     peak_allocated_bytes=torch.npu.max_memory_allocated(self.device),
                     peak_reserved_bytes=torch.npu.max_memory_reserved(self.device),
                     host_statistics=getattr(self, 'host_statistics', {}),
                     statistics=self.statistics), indent=2))
        if self.group is not None:
            dist.destroy_process_group(self.group)
            self.group = None
