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


class _VectorSum:
    """Gather and coalesce force rows with the existing Ascend C graph map."""
    def __init__(self, source, destination, outputs, device):
        self.rows = source.numel()
        self.outputs = outputs
        self.plan = dict(sizes=(3, 1, 1, 3), entries=[(i, 0, 0, i, 1.) for i in range(3)],
                         _ascendc_graph=dict(src=source, dst=destination, mask=None,
                                            destination_nodes=outputs))
        self.right = torch.ones(self.rows, 1, 1, device=device, dtype=torch.float32)
        self.weight = torch.ones(1, 1, 1, device=device, dtype=torch.float32)

    def __call__(self, value):
        if not self.rows:
            return value.new_zeros(self.outputs, 3)
        from mace_ictc.ascendc.product import raw
        return raw(value.contiguous().reshape(-1, 1, 3), self.right, self.weight,
                   None, self.plan, True).reshape(self.outputs, 3)


class AscendLAMMPSEngine:
    def __init__(self, checkpoint, cutoff, device_index, rank, world_size, master, port):
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
                               max_packed_rows=0, timed_calls=0)
        self.initial_allocated_bytes = torch.npu.memory_allocated(self.device)
        torch.npu.reset_peak_memory_stats(self.device)
        self.profile = os.environ.get('MFF_ASCEND_PROFILE', '0') == '1'
        self.timing_skip = int(os.environ.get('MFF_ASCEND_PROFILE_SKIP', '0'))
        if self.timing_skip < 0:raise ValueError('MFF_ASCEND_PROFILE_SKIP must be nonnegative')
        self.generation = None

    @staticmethod
    def _host(buffer, dtype, shape):
        return torch.from_numpy(np.frombuffer(buffer, dtype=dtype).reshape(shape))

    def topology(self, nlocal, nodes, atomic_numbers, src, dst, shifts,
                 ghost_nodes, ghost_slots, receive_indices, send_counts, receive_counts, generation):
        device = self.device
        self.nlocal, self.nodes = int(nlocal), int(nodes)
        self.atomic_numbers = self._host(atomic_numbers, np.int64, (nodes,)).to(device)
        species = set(self.model.atomic_numbers)
        if not set(np.frombuffer(atomic_numbers, dtype=np.int64)).issubset(species):
            raise ValueError('LAMMPS atom types include elements absent from the checkpoint')
        self.src = self._host(src, np.int64, (-1,)).to(device)
        self.dst = self._host(dst, np.int64, (-1,)).to(device)
        self.shifts = self._host(shifts, np.float32, (-1, 3)).to(device)
        self.batch = torch.zeros(nodes, dtype=torch.int64, device=device)
        ghosts = self._host(ghost_nodes, np.int64, (-1,)).to(device)
        slots = self._host(ghost_slots, np.int64, (-1,)).to(device)
        receivers = self._host(receive_indices, np.int64, (-1,)).to(device)
        self.send_counts = np.frombuffer(send_counts, dtype=np.int64).tolist()
        self.receive_counts = np.frombuffer(receive_counts, dtype=np.int64).tolist()
        self.send_offsets, self.receive_offsets = [0], [0]
        for n in self.send_counts:self.send_offsets.append(self.send_offsets[-1] + n)
        for n in self.receive_counts:self.receive_offsets.append(self.receive_offsets[-1] + n)
        self.pack = _VectorSum(ghosts, slots, self.send_offsets[-1], device)
        self.accumulate = _VectorSum(torch.arange(receivers.numel(), device=device), receivers, nlocal, device)
        self.receive_buffer = torch.empty(receivers.numel(), 3, device=device, dtype=torch.float32)
        self.generation = generation
        self.statistics['topology_updates'] += 1
        for key, value in (('max_owned_nodes', nlocal), ('max_model_nodes', nodes),
                           ('max_edges', self.src.numel()), ('max_ghost_rows', ghosts.numel()),
                           ('max_packed_rows', self.send_offsets[-1])):
            self.statistics[key] = max(self.statistics[key], int(value))

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
            for request in dist.batch_isend_irecv(operations):request.wait()
        self.statistics['p2p_operations'] += len(operations)
        self.statistics['sent_bytes'] += 12 * sum(n for p, n in enumerate(self.send_counts) if p != self.rank)
        self.statistics['received_bytes'] += 12 * sum(n for p, n in enumerate(self.receive_counts) if p != self.rank)
        return forces[:self.nlocal] + self.accumulate(receive)

    def compute(self, positions, cell, want_virial):
        if self.generation is None:raise RuntimeError('Model topology has not been supplied')
        if self.profile:torch.npu.synchronize()
        start = time.perf_counter()
        pos = self._host(positions, np.float32, (self.nodes, 3)).to(self.device)
        box = self._host(cell, np.float32, (1, 3, 3)).to(self.device)
        if self.profile:torch.npu.synchronize()
        transferred = time.perf_counter()
        with torch.enable_grad():
            pos.requires_grad_(True)
            strain = torch.zeros(3, 3, device=self.device, dtype=pos.dtype, requires_grad=bool(want_virial))
            if want_virial:
                transform = torch.eye(3, device=self.device, dtype=pos.dtype) + strain
                x, h = pos @ transform, box @ transform
            else:x, h = pos, box
            if self.nodes:
                prediction = self.model(x, self.atomic_numbers, self.batch, self.src, self.dst, self.shifts, h)
                atomic = (prediction[0] if isinstance(prediction, tuple) else prediction).reshape(self.nodes)
                atomic = atomic + self.e0[self.atomic_numbers]
                energy = atomic[:self.nlocal].sum()
            else:
                atomic = pos.new_empty(0)
                energy = pos.sum() * 0
            variables = (pos, strain) if want_virial else (pos,)
            gradients = torch.autograd.grad(energy + pos.sum()*0 + strain.sum()*0,
                                            variables, allow_unused=True)
        forces = -gradients[0].detach()
        if want_virial:
            tensor = -gradients[1].detach()
            tensor = (tensor + tensor.T) * .5
            virial = torch.stack((tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                  tensor[0, 1], tensor[0, 2], tensor[1, 2]))
        else:virial = pos.new_zeros(6)
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
        destination = os.environ.get('MFF_ASCEND_STATS')
        if destination:
            import json
            from pathlib import Path
            path = Path(destination)
            path.mkdir(parents=True, exist_ok=True)
            (path/f'ascend_rank{self.rank}.json').write_text(json.dumps(
                dict(rank=self.rank, world_size=self.world_size, state_source=self.state_source,
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
