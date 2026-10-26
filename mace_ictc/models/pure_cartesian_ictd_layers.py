"""Shared equivariant readouts and physical-tensor basis transforms."""

from __future__ import annotations

import itertools
import torch
import torch.nn as nn
import math
from functools import lru_cache

from mace_ictc.models.ictd_irreps import (
    apply_channel_adapter_per_l,
    build_harmonic_projectors,
    build_harmonic_reconstructors,
    sym_dim,
)
from mace_ictc.models.ictd_fast import _counts_list


def _irreps_total_dim(channels: int, lmax: int) -> int:
    return channels * (lmax + 1) ** 2


def resolve_save_multiple_mix_channels(
    channels: int,
    num_interaction: int,
    requested_mix_channels: int | None,
) -> int:
    channels = int(channels)
    num_interaction = int(num_interaction)
    if requested_mix_channels is not None:
        return int(requested_mix_channels)
    # Default bottleneck width scales with interaction depth while remaining
    # narrower than the full concatenated input.
    return max(1, math.ceil(channels * num_interaction / 2))


def _split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    widths = [channels * (2 * l + 1) for l in range(lmax + 1)]
    # One split joins cotangents once, instead of creating and adding a
    # full-width zero-padded gradient for every independently sliced degree.
    # Preserve front-slicing when a caller requests only the lower degrees.
    blocks = x.split([*widths, x.shape[-1] - sum(widths)], dim=-1)
    return {l: blocks[l].view(*x.shape[:-1], channels, 2 * l + 1)
            for l in range(lmax + 1)}


def _merge_irreps(blocks: dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    parts = []
    for l in range(lmax + 1):
        parts.append(blocks[l].reshape(*blocks[l].shape[:-2], channels * (2 * l + 1)))
    return torch.cat(parts, dim=-1)


class SO3BlockRMSNorm(nn.Module):
    """Per-l block RMS norm for flattened SO(3) features."""

    def __init__(self, channels: int, lmax: int, eps: float = 1e-8):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(self.lmax + 1, self.channels, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            blk = blocks[l]
            rms = blk.pow(2).mean(dim=(-1, -2), keepdim=True).add(self.eps).sqrt()
            gain = self.gain[l].view(*([1] * (blk.ndim - 2)), self.channels, 1).to(dtype=blk.dtype)
            out_blocks[l] = blk * (gain / rms)
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class EquivariantScalarReadoutSO3(nn.Module):
    """
    Strictly equivariant linear readout to 0e from a flattened SO(3) feature.

    A linear SO(3)-equivariant map to 0e can only act on the l=0 block, so this
    head extracts the scalar block and mixes channels linearly to produce one
    scalar per node.
    """

    def __init__(self, channels: int, lmax: int, output_init_std: float = 0.003):
        """
        output_init_std: small value (default 0.003) so initial energy predictions ≈ 0,
        keeping energy loss small early in training so that force gradients dominate
        early learning dynamics (standard MLIP training practice).
        """
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.readout = nn.Linear(self.channels, 1, bias=False)
        nn.init.normal_(self.readout.weight, mean=0.0, std=1.0 / (float(self.channels) ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = _split_irreps(x, self.channels, self.lmax)
        scalar_block = blocks[0].squeeze(-1)  # (..., C)
        return self.readout(scalar_block)


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


@lru_cache(maxsize=None)
def _sym_rank_linear_indices_and_coefs(L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute:
      - linear indices into a flattened (..., 3,3,...,3) rank-L tensor (last axis flattened)
        for a canonical representative of each (a,b,c) monomial in Sym^L.
      - multinomial coefficients multinomial(L; a,b,c) that map the symmetric tensor entry
        to the polynomial coefficient t_{abc}.

    Ordering of (a,b,c) follows `mace_ictc.models.ictd_fast._counts_list(L)`,
    which is the canonical order expected by ICTC projectors in `ictd_irreps`.

    Returns:
      lin_idx: (Dsym(L),) int64
      coefs:   (Dsym(L),) float64
    """
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    counts = _counts_list(L)
    D = len(counts)
    if L == 0:
        return torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.float64)

    # Flattening convention for a rank-L tensor shaped (3,)*L:
    # linear index = sum_{k=0..L-1} i_k * 3^{L-1-k}
    pow3 = [3 ** (L - 1 - k) for k in range(L)]
    lin = torch.empty(D, dtype=torch.int64)
    coefs = torch.empty(D, dtype=torch.float64)
    for d, (a, b, c) in enumerate(counts):
        idx_seq = ([0] * a) + ([1] * b) + ([2] * c)  # canonical representative
        if len(idx_seq) != L:
            raise RuntimeError("Internal error: bad exponent tuple length")
        li = 0
        for ik, pk in zip(idx_seq, pow3):
            li += int(ik) * int(pk)
        lin[d] = li
        coefs[d] = float(math.factorial(L) / (math.factorial(a) * math.factorial(b) * math.factorial(c)))
    return lin.contiguous(), coefs.contiguous()


@lru_cache(maxsize=None)
def _sym_rank_linear_permutation_indices(L: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    For each canonical (a,b,c) monomial in Sym^L, return the flattened linear
    indices of all unique cartesian permutations and the matching multinomial
    coefficient.  This is the inverse lookup for `_sym_rank_linear_indices_and_coefs`.
    """
    counts = _counts_list(L)
    if L == 0:
        return [torch.zeros(1, dtype=torch.int64)], torch.ones(1, dtype=torch.float64)

    pow3 = [3 ** (L - 1 - k) for k in range(L)]
    perm_indices: list[torch.Tensor] = []
    coefs = []
    for (a, b, c) in counts:
        idx_seq = ([0] * a) + ([1] * b) + ([2] * c)
        uniq = sorted(set(itertools.permutations(idx_seq, L)))
        lin = []
        for tup in uniq:
            li = 0
            for ik, pk in zip(tup, pow3):
                li += int(ik) * int(pk)
            lin.append(li)
        perm_indices.append(torch.tensor(lin, dtype=torch.int64))
        coefs.append(float(math.factorial(L) / (math.factorial(a) * math.factorial(b) * math.factorial(c))))
    return perm_indices, torch.tensor(coefs, dtype=torch.float64)


@lru_cache(maxsize=None)
def _rank2_antisymmetric_basis() -> torch.Tensor:
    """
    Basis that maps the axial-vector components [yz, zx, xy] to a flattened 3x3
    antisymmetric matrix in row-major order.
    """
    basis = torch.zeros(3, 9, dtype=torch.float64)
    # yz
    basis[0, 5] = 1.0
    basis[0, 7] = -1.0
    # zx
    basis[1, 6] = 1.0
    basis[1, 2] = -1.0
    # xy
    basis[2, 1] = 1.0
    basis[2, 3] = -1.0
    return basis


class PhysicalTensorICTDEmbedding(nn.Module):
    """
    Embed a (symmetric) Cartesian physical tensor of rank L into ICTC-irreps blocks.

    Supported inputs (per-sample, with optional channel/multiplicity dimension):
      - cartesian symmetric tensor: (..., Cin, 3, 3, ..., 3) with L trailing 3's
        (assumes symmetry across index permutations; a single canonical entry is used)
      - monomial coefficients (Sym^L): (..., Cin, Dsym(L)) where Dsym(L) = (L+2 choose 2)

    Output:
      - dict l -> (..., Cout, 2l+1) for all l=0..lmax_out (missing l are zero)
      - or a flattened (..., Cout*(lmax_out+1)^2) vector compatible with this model's
        internal irreps layout (concatenated l-blocks).

    Internals:
      Uses the trace-chain projectors from `ictd_irreps.build_harmonic_projectors` to decompose
      Sym^L into harmonic irreps blocks l=L, L-2, ..., (0 or 1). This matches how ICTC builds
      Cartesian tensor products: polynomial multiplication in Sym-space followed by trace-chain
      projection into irreps.
    """

    def __init__(
        self,
        *,
        rank: int,
        lmax_out: int,
        channels_in: int = 1,
        channels_out: int | None = None,
        input_repr: str = "cartesian",
        include_trace_chain: bool = True,
        rank2_mode: str = "symmetric",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rank = int(rank)
        self.lmax_out = int(lmax_out)
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out) if channels_out is not None else int(channels_in)
        self.input_repr = str(input_repr).strip().lower()
        self.include_trace_chain = bool(include_trace_chain)
        self.rank2_mode = str(rank2_mode).strip().lower()
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)

        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
        if self.lmax_out < 0:
            raise ValueError(f"lmax_out must be >= 0, got {self.lmax_out}")
        if self.channels_in <= 0:
            raise ValueError(f"channels_in must be positive, got {self.channels_in}")
        if self.channels_out <= 0:
            raise ValueError(f"channels_out must be positive, got {self.channels_out}")
        if self.input_repr not in ("cartesian", "monomial"):
            raise ValueError(f"input_repr must be 'cartesian' or 'monomial', got {self.input_repr!r}")
        if self.rank2_mode not in ("symmetric", "full"):
            raise ValueError(f"rank2_mode must be 'symmetric' or 'full', got {self.rank2_mode!r}")
        if self.rank != 2 and self.rank2_mode != "symmetric":
            raise ValueError("rank2_mode='full' is only supported for rank=2")

        # Precompute Sym^L lookup for cartesian -> monomial coefficients
        lin_idx, coefs = _sym_rank_linear_indices_and_coefs(self.rank)
        self.register_buffer("_sym_lin_idx_cpu", lin_idx, persistent=False)
        self.register_buffer("_sym_multinom_cpu", coefs, persistent=False)

        # Per-l channel mixing (Cin -> Cout). Only needed for l present in trace chain and <= lmax_out.
        self._adapters = nn.ModuleDict()
        for l in range(self.lmax_out + 1):
            if self.channels_in == self.channels_out:
                self._adapters[str(l)] = nn.Identity()
            else:
                self._adapters[str(l)] = nn.Linear(self.channels_in, self.channels_out, bias=False)

        # Cache projectors per (device, dtype) to avoid repeated .to() allocations.
        # key: (device_str, dtype_str) -> dict l -> P (2l+1, Dsym(rank)) in internal_compute_dtype
        self._proj_cache: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

        self.output_dim = _irreps_total_dim(self.channels_out, self.lmax_out)

    def _get_projectors(self, device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
        # Use internal_compute_dtype for stability; output casting happens later.
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached

        proj = build_harmonic_projectors(Lmax=self.rank)  # cached CPU float64
        D = sym_dim(self.rank)

        P_by_l: dict[int, torch.Tensor] = {}
        if self.include_trace_chain:
            ls = list(range(self.rank, -1, -2))
        else:
            ls = [self.rank]
        if self.rank == 2 and self.rank2_mode == "full" and 1 <= self.lmax_out:
            ls = sorted(set(ls + [1]))
        for l in ls:
            if l > self.lmax_out:
                continue
            if (self.rank, l) not in proj.P:
                continue
            P = proj.P[(self.rank, l)]  # (2l+1, D) CPU float64
            if P.shape[-1] != D:
                raise RuntimeError("ICTC projector has unexpected Sym dimension")
            P_by_l[l] = P.to(device=device, dtype=compute_dtype).contiguous()

        self._proj_cache[key] = P_by_l
        return P_by_l

    def _ensure_channel_dim_monomial(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (..., D) or (..., Cin, D) where channel dim is -2.
        if x.dim() >= 2 and x.shape[-2] == self.channels_in:
            return x
        return x.unsqueeze(-2)

    def _ensure_channel_dim_cartesian(self, T: torch.Tensor) -> torch.Tensor:
        # Accept (..., Cin, 3,3,...,3) where channel dim is -(rank+1),
        # or (..., 3,3,...,3) when channels_in==1 (we insert the Cin dim).
        if self.rank == 0:
            # Scalar: accept (..., Cin) or (...) when Cin==1.
            if T.dim() >= 1 and T.shape[-1] == self.channels_in:
                return T
            if self.channels_in == 1:
                return T.unsqueeze(-1)
            raise ValueError(f"rank=0 cartesian tensor must have trailing channels_in={self.channels_in}; got shape={tuple(T.shape)}")

        if T.dim() >= (self.rank + 1) and T.shape[-(self.rank + 1)] == self.channels_in:
            return T
        if self.channels_in == 1:
            return T.unsqueeze(-(self.rank + 1))
        raise ValueError(f"Expected channels_in={self.channels_in} at dim -{self.rank + 1}, got shape={tuple(T.shape)}")

    def _to_monomial_coeffs(self, T: torch.Tensor) -> torch.Tensor:
        """
        Convert a symmetric cartesian tensor (..., Cin, 3,3,...,3) to monomial coefficients
        (..., Cin, Dsym(rank)) consistent with ICTC ordering.
        """
        if self.rank == 0:
            Tch = self._ensure_channel_dim_cartesian(T)  # (..., Cin)
            return Tch.unsqueeze(-1)  # (..., Cin, 1)

        # Expect L trailing 3's (channel dim may be absent when channels_in==1)
        if T.dim() < self.rank:
            raise ValueError(f"cartesian input must have at least rank={self.rank} dims; got shape={tuple(T.shape)}")
        if tuple(T.shape[-self.rank:]) != (3,) * self.rank:
            raise ValueError(f"cartesian input last {self.rank} dims must be all 3; got shape={tuple(T.shape)}")

        T = self._ensure_channel_dim_cartesian(T)

        # Flatten the 3^L cartesian indices and gather one canonical representative per (a,b,c).
        lin_idx = self._sym_lin_idx_cpu.to(device=T.device)
        coefs = self._sym_multinom_cpu.to(device=T.device, dtype=T.dtype)
        flat = T.reshape(*T.shape[:-self.rank], 3 ** self.rank)  # (..., Cin, 3^L)
        vals = flat.index_select(-1, lin_idx)  # (..., Cin, Dsym)
        return vals * coefs

    def forward(self, tensor: torch.Tensor, *, return_blocks: bool = False) -> torch.Tensor | dict[int, torch.Tensor]:
        """
        Args:
          tensor:
            - if input_repr="cartesian": (..., Cin, 3,3,...,3) (rank trailing dims)
            - if input_repr="monomial": (..., Cin, Dsym(rank))
          return_blocks: if True, return dict l->(..., Cout, 2l+1); else return flattened (..., Cout*(lmax_out+1)^2)
        """
        dtype = tensor.dtype
        device = tensor.device

        if self.input_repr == "cartesian":
            tensor_for_trace = tensor
            if self.rank == 2 and self.rank2_mode == "full":
                if tensor.shape[-1:] == (9,):
                    tensor_for_trace = tensor.reshape(*tensor.shape[:-1], 3, 3)
                tensor_for_trace = 0.5 * (tensor_for_trace + tensor_for_trace.transpose(-1, -2))
            t = self._to_monomial_coeffs(tensor_for_trace)  # (..., Cin, Dsym)
        else:
            t = self._ensure_channel_dim_monomial(tensor)
            D = sym_dim(self.rank)
            if t.shape[-1] != D:
                raise ValueError(f"monomial input last dim must be Dsym(rank)={D}, got {t.shape[-1]}")
            if t.shape[-2] != self.channels_in:
                raise ValueError(f"monomial input channel dim must be channels_in={self.channels_in}, got {t.shape[-2]}")

        # Project Sym^L -> harmonic blocks using trace-chain projectors
        P_by_l = self._get_projectors(device=device, dtype=dtype)
        compute_dtype = self.internal_compute_dtype
        t_comp = t.to(dtype=compute_dtype) if t.dtype != compute_dtype else t

        out_blocks: dict[int, torch.Tensor] = {}
        batch_shape = t.shape[:-2]
        antisym_block = None
        if self.rank == 2 and self.rank2_mode == "full":
            if self.input_repr != "cartesian":
                raise ValueError("rank2_mode='full' currently requires input_repr='cartesian'")
            T = tensor.to(dtype=compute_dtype) if tensor.dtype != compute_dtype else tensor
            if T.shape[-2:] == (9,):
                T = T.reshape(*T.shape[:-1], 3, 3)
            elif T.shape[-2:] == (6,):
                xx, yy, zz, xy, xz, yz = T.unbind(dim=-1)
                row0 = torch.stack((xx, xy, xz), dim=-1)
                row1 = torch.stack((xy, yy, yz), dim=-1)
                row2 = torch.stack((xz, yz, zz), dim=-1)
                T = torch.stack((row0, row1, row2), dim=-2)
            T = self._ensure_channel_dim_cartesian(T)
            anti = 0.5 * (T - T.transpose(-1, -2))
            antisym_block = torch.stack(
                (anti[..., 1, 2], anti[..., 2, 0], anti[..., 0, 1]),
                dim=-1,
            )
        for l in range(self.lmax_out + 1):
            if l in P_by_l:
                P = P_by_l[l]  # (2l+1, Dsym) compute_dtype
                # (..., Cin, D) x (2l+1, D) -> (..., Cin, 2l+1)
                c = torch.einsum("...cd,md->...cm", t_comp, P)
                c = c.to(dtype=dtype) if c.dtype != dtype else c
                c = apply_channel_adapter_per_l(c, self._adapters[str(l)])  # (..., Cout, 2l+1)
                out_blocks[l] = c
            elif l == 1 and antisym_block is not None:
                c = antisym_block.to(dtype=dtype) if antisym_block.dtype != dtype else antisym_block
                c = apply_channel_adapter_per_l(c, self._adapters[str(l)])
                out_blocks[l] = c
            else:
                out_blocks[l] = torch.zeros(*batch_shape, self.channels_out, 2 * l + 1, device=device, dtype=dtype)

        if return_blocks:
            return out_blocks
        return _merge_irreps(out_blocks, self.channels_out, self.lmax_out)


class PhysicalTensorICTDRecovery(nn.Module):
    """
    Recover a symmetric Cartesian tensor from ICTC irreps blocks.

    This is the inverse companion of :class:`PhysicalTensorICTDEmbedding` at the
    representation level:
      irreps blocks -> Sym^L monomial coefficients -> symmetric Cartesian tensor.

    Notes
    -----
    - Exact recovery requires that the provided blocks contain the complete trace
      chain for the target rank (e.g. l=0 and l=2 for a general symmetric rank-2 tensor).
    - If only the highest-l block is given (e.g. l=2), the recovered Cartesian
      tensor is the corresponding traceless symmetric component.
    """

    def __init__(
        self,
        *,
        rank: int,
        channels_in: int = 1,
        lmax_in: int | None = None,
        include_trace_chain: bool = True,
        rank2_mode: str = "symmetric",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rank = int(rank)
        self.channels_in = int(channels_in)
        self.lmax_in = int(lmax_in) if lmax_in is not None else int(rank)
        self.include_trace_chain = bool(include_trace_chain)
        self.rank2_mode = str(rank2_mode).strip().lower()
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)

        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
        if self.channels_in <= 0:
            raise ValueError(f"channels_in must be positive, got {self.channels_in}")
        if self.lmax_in < 0:
            raise ValueError(f"lmax_in must be >= 0, got {self.lmax_in}")
        if self.rank2_mode not in ("symmetric", "full"):
            raise ValueError(f"rank2_mode must be 'symmetric' or 'full', got {self.rank2_mode!r}")
        if self.rank != 2 and self.rank2_mode != "symmetric":
            raise ValueError("rank2_mode='full' is only supported for rank=2")

        self._recon_cache: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        self._antisym_cache: dict[tuple[str, str], torch.Tensor] = {}

    def _get_reconstructors(self, device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._recon_cache.get(key)
        if cached is not None:
            return cached

        recon = build_harmonic_reconstructors(Lmax=self.rank)
        if self.include_trace_chain:
            ls = list(range(self.rank, -1, -2))
        else:
            ls = [self.rank]

        out: dict[int, torch.Tensor] = {}
        for l in ls:
            if l > self.lmax_in:
                continue
            V = recon.V[(self.rank, l)]
            out[l] = V.to(device=device, dtype=compute_dtype).contiguous()
        self._recon_cache[key] = out
        return out

    def _ensure_channel_dim_blocks(self, blk: torch.Tensor, l: int) -> torch.Tensor:
        if blk.shape[-1] != 2 * l + 1:
            raise ValueError(f"irreps block for l={l} must end with 2l+1={2*l+1}, got shape={tuple(blk.shape)}")
        if blk.dim() >= 2 and blk.shape[-2] == self.channels_in:
            return blk
        if self.channels_in == 1:
            return blk.unsqueeze(-2)
        raise ValueError(
            f"Expected channels_in={self.channels_in} at dim -2 for l={l} block, got shape={tuple(blk.shape)}"
        )

    def _get_rank2_antisymmetric_basis(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._antisym_cache.get(key)
        if cached is not None:
            return cached
        basis = _rank2_antisymmetric_basis().to(device=device, dtype=compute_dtype).contiguous()
        self._antisym_cache[key] = basis
        return basis

    def _to_monomial_coeffs(self, blocks: dict[int, torch.Tensor]) -> torch.Tensor:
        device = next(iter(blocks.values())).device
        dtype = next(iter(blocks.values())).dtype
        compute_dtype = self.internal_compute_dtype
        recons = self._get_reconstructors(device=device, dtype=dtype)
        D = sym_dim(self.rank)

        sample_block = next(iter(blocks.values()))
        batch_shape = self._ensure_channel_dim_blocks(sample_block, next(iter(blocks.keys()))).shape[:-2]
        t = torch.zeros(*batch_shape, self.channels_in, D, device=device, dtype=compute_dtype)

        for l, V in recons.items():
            blk = blocks.get(l)
            if blk is None:
                continue
            blk = self._ensure_channel_dim_blocks(blk, l)
            blk_comp = blk.to(dtype=compute_dtype) if blk.dtype != compute_dtype else blk
            t = t + torch.einsum("...cm,dm->...cd", blk_comp, V)
        return t.to(dtype=dtype) if t.dtype != dtype else t

    def _monomial_to_cartesian(self, t: torch.Tensor, *, squeeze_channel: bool) -> torch.Tensor:
        if self.rank == 0:
            out = t[..., 0]
            return out.squeeze(-1) if squeeze_channel and self.channels_in == 1 else out

        perm_indices, coefs = _sym_rank_linear_permutation_indices(self.rank)
        coefs = coefs.to(device=t.device, dtype=t.dtype)
        flat = torch.zeros(*t.shape[:-1], 3 ** self.rank, device=t.device, dtype=t.dtype)
        for d, idxs_cpu in enumerate(perm_indices):
            idxs = idxs_cpu.to(device=t.device)
            vals = (t[..., d] / coefs[d]).unsqueeze(-1).expand(*t.shape[:-1], idxs.numel())
            idxs = idxs.view(*([1] * (flat.dim() - 1)), idxs.numel()).expand(*t.shape[:-1], idxs.numel())
            flat.scatter_(-1, idxs, vals)
        if self.rank == 2 and self.rank2_mode == "full":
            return flat
        out = flat.reshape(*t.shape[:-1], *([3] * self.rank))
        if squeeze_channel and self.channels_in == 1:
            out = out.squeeze(-self.rank - 1)
        return out

    def forward(
        self,
        blocks: dict[int, torch.Tensor] | torch.Tensor,
        *,
        input_repr: str = "blocks",
        squeeze_channel: bool | None = None,
        return_monomial: bool = False,
    ) -> torch.Tensor:
        """
        Args:
          blocks:
            - if input_repr="blocks": dict l->(..., Cin, 2l+1) or dict l->(..., 2l+1) for Cin=1
            - if input_repr="flat": (..., Cin*(lmax_in+1)^2)
          squeeze_channel:
            If True and channels_in==1, drop the channel dimension in the returned
            Cartesian tensor to match common `(B,3)` / `(B,3,3)` label shapes.
        """
        if squeeze_channel is None:
            squeeze_channel = (self.channels_in == 1)

        if input_repr == "flat":
            if not torch.is_tensor(blocks):
                raise TypeError("input_repr='flat' expects a tensor input")
            blocks_dict = _split_irreps(blocks, self.channels_in, self.lmax_in)
        elif input_repr == "blocks":
            if torch.is_tensor(blocks):
                raise TypeError("input_repr='blocks' expects a dict[int, Tensor] input")
            blocks_dict = blocks
        else:
            raise ValueError(f"input_repr must be 'blocks' or 'flat', got {input_repr!r}")

        t = self._to_monomial_coeffs(blocks_dict)
        if return_monomial:
            return t.squeeze(-2) if squeeze_channel and self.channels_in == 1 else t
        if self.rank == 2 and self.rank2_mode == "full":
            device = t.device
            dtype = t.dtype
            flat = self._monomial_to_cartesian(t, squeeze_channel=False)
            out = flat.reshape(*t.shape[:-2], self.channels_in, 3, 3)
            blk1 = blocks_dict.get(1)
            if blk1 is not None:
                blk1 = self._ensure_channel_dim_blocks(blk1, 1)
                blk1_comp = blk1.to(dtype=self.internal_compute_dtype) if blk1.dtype != self.internal_compute_dtype else blk1
                basis = self._get_rank2_antisymmetric_basis(device=device, dtype=dtype)
                anti_flat = torch.einsum("...cm,mn->...cn", blk1_comp, basis)
                anti = anti_flat.reshape(*blk1.shape[:-2], self.channels_in, 3, 3)
                anti = anti.to(dtype=dtype) if anti.dtype != dtype else anti
                out = out + anti
            if squeeze_channel and self.channels_in == 1:
                out = out.squeeze(-3)
            return out
        return self._monomial_to_cartesian(t, squeeze_channel=squeeze_channel)
