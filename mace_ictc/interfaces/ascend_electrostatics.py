# SPDX-License-Identifier: MIT
"""Owned-source reciprocal electrostatics with differentiable HCCL sums."""
import torch
import torch.distributed as dist


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        out = value.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(out)
        return out

    @staticmethod
    def backward(ctx, gradient):
        # Owned energies sum to the physical energy. Their mesh cotangents must
        # also be summed before differentiating local learned moments.
        return _Sum.apply(gradient)


class NativeReciprocal:
    def __init__(self, model):
        from mace_ictc.models.long_range import LatentReciprocalLongRange
        from ictc_o3.electrostatics.functional import MeshLongRangeKernel3D
        module = model.long_range_module
        if not isinstance(module, LatentReciprocalLongRange) or not isinstance(module.kernel, MeshLongRangeKernel3D):
            raise ValueError('Ascend LAMMPS electrostatics requires reciprocal-spectral-v1 with mesh_fft')
        if module.kernel.full_ewald and not module.kernel.reciprocal_only:
            raise ValueError('Ascend LAMMPS requires reciprocal-only electrostatics')
        self.module = module
        self.rank = int(getattr(model, 'long_range_max_multipole_l', 0))
        self.channels = module.source_channels
        self.width = self.channels * (13 if self.rank == 2 else 4 if self.rank == 1 else 1)
        if self.rank not in (0, 1, 2):
            raise ValueError('Ascend reciprocal sources support monopoles, dipoles and quadrupoles')
        self.calls = 0
        self._ownership = None
        self._counts = None
        self.description = dict(backend='ascendc-reciprocal', mesh=module.kernel.mesh_shape,
            assignment=module.kernel.assignment, smearing_sigma=module.kernel.smearing_sigma,
            boundary=module.kernel.boundary, source_channels=self.channels,
            counting='owned atoms only', fft='NPU real FFT, replicated grid',
            transport='HCCL amplitude/potential sums and their reverse sums')

    def set_ownership(self, owned, size, device):
        """Refresh global atom count once per topology rebuild, on all ranks."""
        counts = torch.tensor([owned], dtype=torch.int64, device=device)
        if size > 1:dist.all_reduce(counts)
        self._counts = counts.to(torch.float32).clamp_min(1)
        self._ownership = int(owned), int(size)

    def __call__(self, pos, batch, cell, source, *, owned, size):
        from ictc_o3.electrostatics.functional import _prepare_frac_for_boundary_batched
        from ictc_o3.backends.ascend.reciprocal_fft import scalar_potential, multipole_energy
        if cell.shape != (1, 3, 3) or source.ndim != 2 or source.shape[1] != self.width or source.shape[0] < owned:
            raise ValueError('Expected one cell and packed owned reciprocal sources')
        kernel = self.module.kernel
        source = source[:owned]
        q = source[:, :self.channels] + pos[:owned].sum()*0.
        dipole = source[:, self.channels:4*self.channels].reshape(owned, self.channels, 3) if self.rank > 0 else None
        quadrupole = source[:, 4*self.channels:].reshape(owned, self.channels, 3, 3) if self.rank > 1 else None
        reduce = _Sum.apply if size > 1 else None
        if self._counts is None:
            counts = torch.tensor([owned], dtype=torch.int64, device=pos.device)
            if size > 1:dist.all_reduce(counts)
            counts = counts.to(q.dtype).clamp_min(1)
        else:
            if self._ownership != (owned, size):
                raise RuntimeError('Reciprocal ownership changed without set_ownership on all ranks')
            counts = self._counts
        if self.module.neutralize:
            total = q.sum(0, keepdim=True)
            if reduce is not None:total = reduce(total)
            q = q - total / counts[:, None]
        position, graph = pos[:owned], batch[:owned]
        if self.rank:
            energy = multipole_energy(kernel, position, graph, cell, q, dipole, quadrupole,
                reduce_amplitudes=reduce, graph_counts=counts, compact_quadrupole=True)
        else:
            frac = _prepare_frac_for_boundary_batched(position, graph, cell,
                boundary=kernel.boundary, slab_padding_factor=kernel.slab_padding_factor)
            mesh = kernel._spread(frac, graph, q, 1)
            potential, _, _, _ = scalar_potential(kernel, mesh, cell, reduce_spectrum=reduce)
            energy = .5 * (q * kernel._gather(frac, graph, potential)).sum(-1, keepdim=True)
            if kernel.energy_partition == 'uniform':
                total = energy.sum()
                if reduce is not None:total = reduce(total)
                energy = (total/counts).expand(owned, 1)
        if self.module.energy_scale is not None:energy = self.module.energy_scale*energy
        self.calls += 1
        return torch.cat((energy, energy.new_zeros(pos.shape[0]-owned, 1)))
