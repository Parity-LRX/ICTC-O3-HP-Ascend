"""Captured short-range LAMMPS execution with collective topology invalidation."""
import os
import time
import numpy as np
import torch
import torch_npu

from .ascend_lammps import AscendLAMMPSEngine


class CapturedLAMMPSEngine(AscendLAMMPSEngine):
    """Replay device EF/EFV and HCCL tasks while LAMMPS retains host control."""
    _communication_keys = ('p2p_operations', 'force_p2p_operations',
        'feature_forward_p2p_operations', 'feature_reverse_p2p_operations',
        'sent_bytes', 'received_bytes', 'force_sent_bytes', 'force_received_bytes',
        'feature_sent_bytes', 'feature_received_bytes')

    def __init__(self, *args, **kwargs):
        if os.environ.get('MFF_ASCEND_MODE', 'performance') != 'performance':
            raise ValueError('ACLGraph deployment currently requires MFF_ASCEND_MODE=performance')
        if os.environ.get('TASK_QUEUE_ENABLE', '1') not in ('0', '1'):
            raise ValueError('ACLGraph requires TASK_QUEUE_ENABLE=1 or 0 with this TorchNPU version')
        if os.environ.get('MFF_ASCEND_PROFILE', '0') == '1':
            raise ValueError('Synchronized stage profiling requires MFF_ASCEND_EXECUTION=eager')
        super().__init__(*args, **kwargs)
        self.execution = 'graph'
        self._graphs = {}
        self._topology_key = None
        self._topology_committed = False
        self._capture_stream = torch.npu.Stream(device=self.device)
        self.p2p_mode = 'coalesced'
        cap = int(os.environ.get('MFF_ASCEND_GRAPH_CHUNK_CAP', '131072'))
        scratch = int(os.environ.get('MFF_ASCEND_GRAPH_SCRATCH_MIB', '3072')) * 1024**2
        minimum = int(os.environ.get('MFF_ASCEND_GRAPH_GROUP_MIN_EDGES', '65536'))
        if cap < 1:raise ValueError('MFF_ASCEND_GRAPH_CHUNK_CAP must be positive')
        if scratch < 1:raise ValueError('MFF_ASCEND_GRAPH_SCRATCH_MIB must be positive')
        if minimum < 0:raise ValueError('MFF_ASCEND_GRAPH_GROUP_MIN_EDGES must be nonnegative')
        for layer in self.model.interactions:
            layer.tp._ascend_edge_chunk_cap = cap
            layer.tp._ascend_edge_scratch_bytes = scratch
            layer.tp._ascend_grouped_min_edges = minimum
        self.statistics.update(graph_captures=0, graph_replays=0, graph_warmups=0,
            graph_topology_reuses=0, graph_global_invalidations=0, graph_seconds=0.,
            graph_ef_captures=0, graph_efv_captures=0, graph_ef_replays=0, graph_efv_replays=0)

    def _clear_graphs(self):
        if self._graphs:
            torch.npu.synchronize(self.device)
            for entry in self._graphs.values():entry['graph'].reset()
            self._graphs.clear()

    def topology(self, nlocal, nodes, *buffers):
        self._topology_committed = False
        generation = buffers[-1]
        # Compare exact host metadata, including owner schedules. Geometry and
        # learned floating features are live inputs, never part of this cache.
        key = (int(nlocal), int(nodes), *(bytes(value) for value in buffers[:-1]))
        if key == self._topology_key:
            self.generation = generation
            self.statistics['topology_updates'] += 1
            return False
        self._clear_graphs()
        super().topology(nlocal, nodes, *buffers)
        self._topology_key = key
        return True

    def topology_commit(self, changed):
        """LAMMPS calls this with the MPI maximum of local change flags."""
        if changed:
            self._clear_graphs()
            self.statistics['graph_global_invalidations'] += 1
        else:self.statistics['graph_topology_reuses'] += 1
        self._topology_committed = True

    def _packed_prediction(self, positions, cell, want_virial):
        forces, atomic, energy, virial = self._device_prediction(positions, cell, want_virial)
        with torch.no_grad():
            owned = self._reverse(forces)
            return torch.cat((owned.reshape(-1), atomic[:self.nlocal], energy.reshape(1), virial))

    def _capture(self, positions, cell, want_virial):
        pos = self._host(positions, np.float32, (self.nodes, 3)).to(self.device).requires_grad_(True)
        box = self._host(cell, np.float32, (1, 3, 3)).to(self.device)
        stream = self._capture_stream
        stream.wait_stream(torch.npu.current_stream(self.device))
        with torch.npu.stream(stream):
            for _ in range(3):
                self._packed_prediction(pos, box, want_virial)
                torch.npu.synchronize(self.device)
        self.statistics['graph_warmups'] += 3
        torch.npu.current_stream(self.device).wait_stream(stream)
        before = dict(self.statistics)
        graph = torch_npu.npu.NPUGraph()
        # A uniform EFV program keeps kernel and workspace addresses stable
        # when the caller alternates force-only and virial requests.
        with torch_npu.npu.graph(graph, stream=stream):
            packed = self._packed_prediction(pos, box, want_virial)
        deltas = {key:self.statistics[key] - before[key] for key in self._communication_keys}
        self.statistics.update(before)  # capture records tasks; replay executes them
        self.statistics['graph_captures'] += 1
        self.statistics['graph_efv_captures' if want_virial else 'graph_ef_captures'] += 1
        return dict(pos=pos, box=box, graph=graph, packed=packed, deltas=deltas)

    def compute(self, positions, cell, want_virial):
        if self.generation is None:raise RuntimeError('Model topology has not been supplied')
        if not self._topology_committed:
            raise RuntimeError('ACLGraph MD requires the MPI topology-commit pair style; rebuild the Ascend LAMMPS interface')
        requested_virial = bool(want_virial)
        mode = True
        started = time.perf_counter()
        if mode not in self._graphs:self._graphs[mode] = self._capture(positions, cell, mode)
        entry = self._graphs[mode]
        with torch.no_grad():
            entry['pos'].copy_(self._host(positions, np.float32, (self.nodes, 3)))
            entry['box'].copy_(self._host(cell, np.float32, (1, 3, 3)))
        uploaded = time.perf_counter()
        entry['graph'].replay()
        packed = entry['packed'].cpu().numpy()
        stopped = time.perf_counter()
        for key, delta in entry['deltas'].items():self.statistics[key] += delta
        self.statistics['graph_replays'] += 1
        self.statistics['graph_efv_replays' if requested_virial else 'graph_ef_replays'] += 1
        if self.statistics['calls'] >= self.timing_skip:
            self.statistics['timed_calls'] += 1
            self.statistics['graph_seconds'] += stopped - uploaded
            self.statistics['transfer_seconds'] += uploaded - started
        self.statistics['calls'] += 1
        n = self.nlocal
        virial = packed[4*n+1:] if requested_virial else np.zeros(6, dtype=np.float32)
        return (float(packed[4*n]), packed[:3*n].reshape(n, 3), packed[3*n:4*n], virial)

    def close(self):
        self._clear_graphs()
        return super().close()
