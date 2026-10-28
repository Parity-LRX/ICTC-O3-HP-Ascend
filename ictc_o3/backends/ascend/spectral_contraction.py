# SPDX-License-Identifier: MIT
"""Fused real multipole amplitudes using native multilinear-map adjoints."""
import torch

_PLANS = {}


def _coefficients(k, dipole, quadrupole, compact):
    x, y, z = k.unbind(-1)
    fields = [torch.ones_like(x)]
    if dipole:fields.extend((x, y, z))
    if quadrupole:
        if compact:fields.extend((x*x, y*y, z*z, x*y, x*z, y*z))
        else:fields.extend((k[..., :, None]*k[..., None, :]).flatten(-2).unbind(-1))
    return torch.stack(fields, -1)


def amplitudes(spectrum, kp, kn, channels, *, has_dipole, has_quadrupole, compact_quadrupole):
    from .sparse_product import sparse_product
    graphs, frequencies = spectrum.shape[:2]
    fields = [spectrum[..., :channels].unsqueeze(-1)]
    if has_dipole:
        fields.append(spectrum[..., channels:4*channels].reshape(graphs, frequencies, channels, 3))
    if has_quadrupole:
        begin = channels*(4 if has_dipole else 1)
        width = 6 if compact_quadrupole else 9
        fields.append(spectrum[..., begin:begin+width*channels].reshape(graphs, frequencies, channels, width))
    packed = torch.cat(fields, -1) if len(fields)>1 else fields[0]
    count = packed.shape[-1]
    left = torch.view_as_real(packed).reshape(graphs*frequencies, channels, 2*count).contiguous()
    positive = _coefficients(kp, has_dipole, has_quadrupole, compact_quadrupole)
    negative = _coefficients(kn, has_dipole, has_quadrupole, compact_quadrupole)
    right = torch.cat((positive, negative), -1).reshape(graphs*frequencies, 1, 2*count).contiguous()
    key = has_dipole, has_quadrupole, compact_quadrupole, channels, str(left.device)
    if key not in _PLANS:
        entries = [(0, 0, 0, 0, 1.), (1, 0, 0, 1, 1.),
                   (0, count, 0, 2, 1.), (1, count, 0, 3, -1.)]
        for index in range(1, count):
            is_dipole = has_dipole and index<=3
            if is_dipole:
                entries.extend(((2*index+1, index, 0, 0, -1.), (2*index, index, 0, 1, 1.),
                                (2*index+1, count+index, 0, 2, 1.), (2*index, count+index, 0, 3, 1.)))
            else:
                entries.extend(((2*index, index, 0, 0, -.5), (2*index+1, index, 0, 1, -.5),
                                (2*index, count+index, 0, 2, -.5), (2*index+1, count+index, 0, 3, .5)))
        plan = dict(sizes=(2*count, 2*count, 1, 4), entries=entries,
                    _ascendc_metadata={}, _ascendc_tiling={}, _ascendc_native=True)
        _PLANS[key] = plan, left.new_ones(1, channels, 1)
    plan, ones = _PLANS[key]
    return sparse_product(left, right, ones, plan).reshape(graphs, frequencies, channels, 4)
