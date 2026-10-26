"""Monomial indexing, Laplacians and radial lifts for the ICTC harmonic basis."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch


def _counts_list(L: int) -> List[Tuple[int, int, int]]:
    """All (a,b,c) with a+b+c=L in a deterministic order."""
    out = []
    for a in range(L + 1):
        for b in range(L - a + 1):
            c = L - a - b
            out.append((a, b, c))
    return out


def _build_laplacian_matrix(L: int, device=None, dtype=None) -> torch.Tensor:
    """
    Laplacian operator Δ = ∂_x^2 + ∂_y^2 + ∂_z^2 on homogeneous polynomials:
      Sym^L -> Sym^{L-2}
    in the monomial coefficient basis indexed by (a,b,c), a+b+c=L.

    This is the standard way to characterize harmonic polynomials (kernel of Δ),
    which correspond to STF tensors of rank L.
    """
    assert L >= 2
    src = _counts_list(L)
    dst = _counts_list(L - 2)
    j_of: Dict[Tuple[int, int, int], int] = {t: i for i, t in enumerate(dst)}
    Dsrc = len(src)
    Ddst = len(dst)
    D = torch.zeros(Ddst, Dsrc, device=device, dtype=dtype)

    for col, (a, b, c) in enumerate(src):
        if a >= 2:
            row = j_of[(a - 2, b, c)]
            D[row, col] += float(a * (a - 1))
        if b >= 2:
            row = j_of[(a, b - 2, c)]
            D[row, col] += float(b * (b - 1))
        if c >= 2:
            row = j_of[(a, b, c - 2)]
            D[row, col] += float(c * (c - 1))
    return D


def _build_r2k_lift(L_src: int, k: int, device=None, dtype=None) -> torch.Tensor:
    """
    Lift map induced by multiplication with r^{2k} = (x^2 + y^2 + z^2)^k:
      Sym^{L_src} -> Sym^{L_dst},  L_dst = L_src + 2k
    in the (a,b,c) monomial-count basis.
    """
    assert k >= 0
    if k == 0:
        src = _counts_list(L_src)
        return torch.eye(len(src), device=device, dtype=dtype)
    L_dst = L_src + 2 * k
    src = _counts_list(L_src)
    dst = _counts_list(L_dst)
    j_of: Dict[Tuple[int, int, int], int] = {t: i for i, t in enumerate(dst)}
    M = torch.zeros(len(dst), len(src), device=device, dtype=dtype)
    # (x^2+y^2+z^2)^k = sum_{u+v+w=k} k!/(u!v!w!) x^{2u} y^{2v} z^{2w}
    coeffs: List[Tuple[int, int, int, float]] = []
    for u in range(k + 1):
        for v in range(k - u + 1):
            w = k - u - v
            coef = math.factorial(k) / (math.factorial(u) * math.factorial(v) * math.factorial(w))
            coeffs.append((u, v, w, float(coef)))
    for col, (a, b, c) in enumerate(src):
        for u, v, w, coef in coeffs:
            row = j_of[(a + 2 * u, b + 2 * v, c + 2 * w)]
            M[row, col] += coef
    return M
