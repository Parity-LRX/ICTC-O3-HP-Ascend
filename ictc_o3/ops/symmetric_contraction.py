"""Species-conditioned symmetric contraction in the fixed ICTC basis."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from e3nn import o3
from ictc_o3.basis.ictd_irreps import direction_harmonics_all
from ictc_o3.basis.layout import _split_irreps
from ._mace_symmetric_contraction import MaceSymmetricContraction

def _hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{int(channels)}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(lmax) + 1)))


def _so3_flat_to_mace_features(x: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    blocks = _split_irreps(x, int(channels), int(lmax))
    return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)


def _merge_blocks_subset(blocks: Dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    return torch.cat([blocks[l].reshape(blocks[l].shape[0], int(channels) * (2 * l + 1)) for l in range(int(lmax) + 1)], dim=-1)


class SO3ToE3NNBasisBridge(nn.Module):
    """
    Fixed per-l orthogonal bridge between the ICTC SO3 basis and e3nn/MACE basis.

    `direction_harmonics_all` now matches e3nn component normalization in RMS, but
    each l block can still differ by an orthogonal basis convention. Native MACE
    contraction assumes the e3nn convention, while ICTC interaction emits ICTC
    convention features. The bridge uses a deterministic least-squares/SVD fit
    from sampled directions to construct Q_l such that:

        Y_ictd_l @ Q_l ~= Y_e3nn_l
    """

    def __init__(self, channels: int, lmax: int, num_samples: int = 8192):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260426)
        dirs = torch.randn(int(num_samples), 3, generator=generator, dtype=torch.float64)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y_ictd = direction_harmonics_all(dirs, self.lmax)
        y_e3nn = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(self.lmax),
            dirs,
            normalize=True,
            normalization="component",
        )
        offset = 0
        for l in range(self.lmax + 1):
            width = 2 * l + 1
            a = y_ictd[l].to(dtype=torch.float64)
            b = y_e3nn[:, offset : offset + width].to(dtype=torch.float64)
            offset += width
            u, _, vh = torch.linalg.svd(a.T @ b)
            q = (u @ vh).to(dtype=torch.get_default_dtype())
            self.register_buffer(f"q_{l}", q, persistent=True)

    def _q(self, l: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return getattr(self, f"q_{int(l)}").to(dtype=dtype, device=device)

    def ictd_flat_to_e3nn_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,mp->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def e3nn_flat_to_ictd_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,pm->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def ictd_flat_to_e3nn_features(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)

    def ictd_flat_to_e3nn_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))

    def e3nn_flat_to_ictd_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.e3nn_flat_to_ictd_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))


class ICTDBridgeUSymmetricContractionSO3(nn.Module):
    """
    Bridge-U symmetric contraction expressed directly in the ICTC basis.

    This is algebraically equivalent to:
      ICTC features -> e3nn basis -> MACE SymmetricContraction -> ICTC basis
    but the per-l basis change is folded into the stored U tensors once at
    initialization. The forward path therefore consumes and returns ICTC-basis
    flat SO3 features. This backend is the stable high-l bridge used when
    pure ICTC U generation is not numerically reliable.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self._fold_basis_change_into_u_tensors()

    def _input_q(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        blocks = [self.basis_bridge._q(l, dtype=dtype, device=device) for l in range(self.lmax + 1)]
        return torch.block_diag(*blocks)

    def _transform_u_tensor(self, u_tensor: torch.Tensor, output_l: int) -> torch.Tensor:
        dtype = u_tensor.dtype
        device = u_tensor.device
        q_in = self._input_q(dtype=dtype, device=device)
        nu = int(u_tensor.dim() - 1) if int(output_l) == 0 else int(u_tensor.dim() - 2)
        if int(output_l) == 0:
            if nu == 1:
                return torch.einsum("ai,ip->ap", q_in, u_tensor)
            if nu == 2:
                return torch.einsum("ai,bj,ijp->abp", q_in, q_in, u_tensor)
            if nu == 3:
                return torch.einsum("ai,bj,ck,ijkp->abcp", q_in, q_in, q_in, u_tensor)
        q_out = self.basis_bridge._q(output_l, dtype=dtype, device=device)
        if nu == 1:
            return torch.einsum("ro,ai,oip->rap", q_out, q_in, u_tensor)
        if nu == 2:
            return torch.einsum("ro,ai,bj,oijp->rabp", q_out, q_in, q_in, u_tensor)
        if nu == 3:
            return torch.einsum("ro,ai,bj,ck,oijkp->rabcp", q_out, q_in, q_in, q_in, u_tensor)
        # Preserve the established small-order expressions. Higher orders use
        # the same fixed basis map on each tensor leg, without a cubic cap or
        # an einsum alphabet limit. This only lowers immutable U coefficients.
        if int(output_l)==0:
            result=u_tensor
            first_axis=0
        else:
            result=torch.tensordot(q_out,u_tensor,dims=([1],[0]))
            first_axis=1
        for axis in range(first_axis,first_axis+nu):
            result=torch.tensordot(q_in,result,dims=([1],[axis])).movedim(0,axis)
        return result

    def _fold_basis_change_into_u_tensors(self) -> None:
        with torch.no_grad():
            for (mul, ir), contraction in zip(self.target_irreps, self.symmetric_contractions.contractions):
                del mul
                output_l = int(ir.l)
                for nu in range(1, int(contraction.correlation) + 1):
                    name = f"U_matrix_{nu}"
                    old = getattr(contraction, name)
                    new = self._transform_u_tensor(old, output_l)
                    old.copy_(new.to(dtype=old.dtype, device=old.device))
                if getattr(contraction, "_use_scalar_corr3_fast", False) and hasattr(
                    contraction, "refresh_scalar_corr3_fast_buffers"
                ):
                    contraction.refresh_scalar_corr3_fast_buffers()

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        if getattr(self, "_ascend_hp", False) and x.device.type == "npu":
            from ictc_o3.backends.ascend.contraction import factorized_contractions
            if node_type_idx is None:
                node_type_idx = node_attrs.argmax(-1)
            return factorized_contractions(self, x, node_type_idx)
        if node_type_idx is None:
            from .reference_u_residency import prepare_reference_u
            prepare_reference_u(self,x)
            return self.symmetric_contractions(x, node_attrs)
        from .reference_u_residency import prepare_reference_u
        prepare_reference_u(self,x)
        return torch.cat(
            [
                contraction(x, node_attrs, node_type_idx=node_type_idx)
                for contraction in self.symmetric_contractions.contractions
            ],
            dim=-1,
        )
