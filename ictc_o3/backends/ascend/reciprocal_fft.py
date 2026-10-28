# SPDX-License-Identifier: MIT
"""Real-FFT reciprocal algebra, preserving both members of every k pair.

At Nyquist indices in a triclinic cell, the folded mirror vector need not
equal -k. Both Green weights and both multipole amplitudes are evaluated;
blindly doubling the retained half would change the original functional.
"""
import math
import os
import torch
from .real_fft import rfftn, irfftn, _scale

_LATTICES = {}


def _lattice(kernel, cell, dtype, y_slice):
    """Cache grid indices and deconvolution, never live cell/Green values."""
    from ictc_o3.electrostatics.functional import _fft_integer_frequencies, _assignment_window_1d
    mx, my, mz = kernel.mesh_shape
    section = None if y_slice is None else (y_slice.start, y_slice.stop, y_slice.step)
    key = kernel.mesh_shape, kernel.assignment, kernel.assignment_window_floor, str(cell.device), dtype, section
    if key in _LATTICES:return _LATTICES[key]
    fx = _fft_integer_frequencies(mx, device=cell.device, dtype=dtype)
    fy = _fft_integer_frequencies(my, device=cell.device, dtype=dtype)
    if y_slice is not None:fy = fy[y_slice]
    z = torch.arange(mz//2+1, device=cell.device, dtype=dtype)
    if mz%2==0:z = torch.where(z==mz//2, -z, z)
    positive = torch.stack(torch.meshgrid(fx, fy, z, indexing='ij'), -1).reshape(-1, 3)
    nyquist = positive.new_tensor([-(n//2) if n%2==0 else float('inf') for n in (mx,my,mz)])
    negative = torch.where(positive==nyquist, positive, -positive)
    if y_slice is None:
        window = kernel._build_assignment_window(device=cell.device, dtype=dtype)[..., :mz//2+1]
    else:
        windows = [_assignment_window_1d(mesh_size=n, assignment=kernel.assignment, device=cell.device, dtype=dtype)
                   for n in (mx,my,mz)]
        window = windows[0][:,None,None]*windows[1][y_slice][None,:,None]*windows[2][:mz//2+1][None,None,:]
    deconv = window.reshape(1,-1).clamp_min(kernel.assignment_window_floor).square().reciprocal()
    index = torch.arange(mz//2+1, device=cell.device)
    interior = (index>0)&((index<mz//2) if mz%2==0 else torch.ones_like(index,dtype=torch.bool))
    interior = interior[None,None,:].expand(mx,fy.numel(),-1).reshape(1,-1)
    if len(_LATTICES)>=8:_LATTICES.pop(next(iter(_LATTICES)))
    _LATTICES[key] = positive, negative, interior, deconv
    return _LATTICES[key]


def _real_weighted_components(weights, values, *, conjugate=False):
    """Contract short Cartesian axes without tiny complex batched GEMMs.

    Weights are real and values are [graph, frequency, source, component].
    Component views keep workspace bounded by one source field, independent
    of the number of Cartesian fields. Ordinary autograd retains coordinate,
    cell and higher derivative support.
    """
    pair = torch.view_as_real(values)
    real = weights[..., 0, None] * pair[..., 0, 0]
    imag = weights[..., 0, None] * pair[..., 0, 1]
    for component in range(1, values.shape[-1]):
        weight = weights[..., component, None]
        value = pair[..., component, :]
        real = real + weight * value[..., 0]
        imag = imag + weight * value[..., 1]
    return real, -imag if conjugate else imag


def geometry(kernel, cell, dtype, *, exclude_k0, y_slice=None):
    from ictc_o3.electrostatics.functional import _effective_cell_for_boundary
    from ictc_o3.electrostatics.functional import _inverse_3x3
    from ictc_o3.electrostatics.functional import _det_3x3
    from ictc_o3.electrostatics.functional import _safe_vector_norm
    mx, my, mz = kernel.mesh_shape
    effective = _effective_cell_for_boundary(cell, boundary=kernel.boundary,
                    slab_padding_factor=kernel.slab_padding_factor, dtype=dtype)
    positive, negative, interior, deconv = _lattice(kernel, cell, dtype, y_slice)
    inverse = _inverse_3x3(effective).transpose(-1, -2)
    kp = 2. * math.pi * torch.einsum('kd,bdh->bkh', positive, inverse)
    kn = 2. * math.pi * torch.einsum('kd,bdh->bkh', negative, inverse)
    volume = _det_3x3(effective).abs().clamp_min(kernel.k_norm_floor)
    real_cutoff = alpha = None
    if kernel.full_ewald:
        sigma = getattr(kernel, 'smearing_sigma', None)
        if sigma is not None and kernel.reciprocal_only:
            alpha = cell.new_full((cell.shape[0],), 1./(math.sqrt(2.)*sigma), dtype=dtype)
        else:
            real_cutoff = kernel._estimate_real_cutoff_batched(cell.to(dtype))
            alpha = kernel._estimate_ewald_alpha(real_cutoff)
    def spectral(k):
        norms = _safe_vector_norm(k, dim=-1, floor=kernel.k_norm_floor)
        value = kernel.green_kernel(norms) / volume[:, None] * deconv
        if alpha is not None:
            value = value * torch.exp(-norms.square() / (4. * alpha[:, None].square()))
        if exclude_k0:
            value = torch.where(norms > kernel.k_norm_floor, value, torch.zeros_like(value))
        return value
    return kp, kn, spectral(kp), spectral(kn), interior, alpha, volume, real_cutoff


def scalar_potential(kernel, mesh, cell, *, reduce_spectrum=None):
    mx, my, mz = kernel.mesh_shape
    _, _, sp, sn, _, alpha, volume, cutoff = geometry(
        kernel, cell, mesh.dtype, exclude_k0=kernel.full_ewald or not kernel.include_k0)
    field = mesh.permute(0, 4, 1, 2, 3).contiguous()
    transformed = rfftn(field)
    if reduce_spectrum is not None:
        transformed = torch.view_as_complex(reduce_spectrum(torch.view_as_real(transformed).contiguous()))
    # The original returns the real part of a full inverse FFT. Symmetrizing
    # the folded Green kernel reproduces that real linear map exactly.
    weights = (.5 * (sp + sn)).reshape(cell.shape[0], 1, mx, my, mz//2+1)
    filtered = irfftn(_scale(transformed, weights), (mx,my,mz))
    return (filtered * float(mx*my*mz)).permute(0,2,3,4,1), alpha, volume, cutoff


def pack_moments(source, dipole, quadrupole, *, compact_quadrupole=False):
    """Pack Cartesian fields, optionally using the six exact Q combinations."""
    n, channels = source.shape
    fields = [source]
    if dipole is not None:
        fields.append(dipole.reshape(n, channels * 3))
    if quadrupole is not None:
        if compact_quadrupole:
            q = quadrupole
            compact = torch.stack((q[...,0,0],q[...,1,1],q[...,2,2],
                q[...,0,1]+q[...,1,0],q[...,0,2]+q[...,2,0],q[...,1,2]+q[...,2,1]),dim=-1)
            fields.append(compact.reshape(n, channels * 6))
        else:
            fields.append(quadrupole.reshape(n, channels * 9))
    return torch.cat(fields, dim=1) if len(fields)>1 else source


def multipole_energy(kernel, pos, batch, cell, source, dipole, quadrupole,
                     *, reduce_amplitudes=None, graph_counts=None, compact_quadrupole=False):
    from ictc_o3.electrostatics.functional import _prepare_frac_for_boundary_batched
    frac = _prepare_frac_for_boundary_batched(pos, batch, cell,
                boundary=kernel.boundary, slab_padding_factor=kernel.slab_padding_factor)
    channels = source.shape[1]
    packed = pack_moments(source,dipole,quadrupole,compact_quadrupole=compact_quadrupole)
    # Assignment and its coordinate/mixed adjoints coalesce source channels.
    # A bounded grid transpose is cheaper than strided plane access per lane.
    mesh = kernel._spread(frac, batch, packed, cell.shape[0]).permute(0,4,1,2,3).contiguous()
    transformed = rfftn(mesh).flatten(2).transpose(1,2)
    graph_energy = multipole_spectral_energy(kernel, transformed, cell, channels,
        has_dipole=dipole is not None, has_quadrupole=quadrupole is not None,
        reduce_amplitudes=reduce_amplitudes,compact_quadrupole=compact_quadrupole)
    counts = kernel._graph_counts(batch, cell, dtype=source.dtype) if graph_counts is None else graph_counts
    return (graph_energy.index_select(0,batch)/counts.index_select(0,batch))[:,None]


def multipole_spectral_energy(kernel, transformed, cell, channels, *,
                             has_dipole, has_quadrupole,
                             reduce_amplitudes=None, y_slice=None, compact_quadrupole=False):
    """Energy of a complete spectrum or a disjoint slab along its y axis."""
    kp, kn, sp, sn, interior, _, _, _ = geometry(kernel, cell,
        transformed.real.dtype, exclude_k0=True, y_slice=y_slice)
    policy = os.environ.get('ICTC_ASCEND_SPECTRAL_FUSION', '1')
    if policy not in ('0', '1'):
        raise ValueError('ICTC_ASCEND_SPECTRAL_FUSION must be zero or one')
    if policy == '1':
        from .spectral_contraction import amplitudes
        amplitude = amplitudes(transformed, kp, kn, channels,
            has_dipole=has_dipole, has_quadrupole=has_quadrupole,
            compact_quadrupole=compact_quadrupole)
        if reduce_amplitudes is not None:amplitude = reduce_amplitudes(amplitude)
        density = amplitude.reshape(*amplitude.shape[:-1], 2, 2).square().sum(-1)
        return .5*((sp[..., None]*density[..., 0]) + ((sn*interior)[..., None]*density[..., 1])).sum((1, 2))
    def amplitude(spectrum, k, *, conjugate=False):
        scalar = spectrum[:, :, :channels]
        real, imag = torch.view_as_real(scalar).unbind(-1)
        if conjugate:imag = -imag
        if has_dipole:
            mu = spectrum[:, :, channels:4*channels].reshape(cell.shape[0],-1,channels,3)
            value_real, value_imag = _real_weighted_components(k, mu, conjugate=conjugate)
            real, imag = real - value_imag, imag + value_real
        if has_quadrupole:
            offset = channels * (4 if has_dipole else 1)
            if compact_quadrupole:
                quad = spectrum[:, :, offset:offset+6*channels].reshape(cell.shape[0],-1,channels,6)
                x, y, z = k.unbind(-1)
                weights = torch.stack((x*x,y*y,z*z,x*y,x*z,y*z),dim=-1)
                value_real, value_imag = _real_weighted_components(weights, quad, conjugate=conjugate)
            else:
                quad = spectrum[:, :, offset:offset+9*channels].reshape(cell.shape[0],-1,channels,3,3)
                weights = (k[..., :, None]*k[..., None, :]).flatten(-2)
                value_real, value_imag = _real_weighted_components(weights, quad.flatten(-2), conjugate=conjugate)
            real, imag = real - .5 * value_real, imag - .5 * value_imag
        if reduce_amplitudes is not None:
            return torch.stack((real, imag), dim=-1)
        return real.square()+imag.square()
    positive = amplitude(transformed, kp)
    negative = amplitude(transformed, kn, conjugate=True)
    if reduce_amplitudes is not None:
        # Communicate two folded complex source amplitudes, rather than every
        # Cartesian moment grid. Both Nyquist partners are still explicit.
        positive, negative = reduce_amplitudes(torch.stack((positive, negative))).unbind(0)
        positive = positive.square().sum(-1)
        negative = negative.square().sum(-1)
    density = sp[:, :, None] * positive
    density = density + (sn * interior)[:, :, None] * negative
    return .5 * density.sum(dim=(1,2))
