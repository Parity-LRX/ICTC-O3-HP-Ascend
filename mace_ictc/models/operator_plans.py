"""Model unwrapping and differentiable reference configuration."""
import torch


def mace_backbone_core(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap deployment/DDP wrappers without selecting an unrelated backbone."""
    from .pure_cartesian_ictd_fix import PureCartesianICTDFix
    while not isinstance(model, PureCartesianICTDFix):
        child = getattr(model, "module", None)
        if child is None:
            child = getattr(model, "model", None)
        if not isinstance(child, torch.nn.Module) or child is model:
            break
        model = child
    return model



def configure_mace_runtime_plan(model, *, policy="auto", training=False,
                                capture=False, generated_layout="packed"):
    if policy not in ("auto", "reference", "generated"):
        raise ValueError("policy must be auto, reference or generated")
    core = mace_backbone_core(model)
    if policy != "reference" and next(core.parameters()).device.type == "npu":
        from mace_ictc.ascend.runtime import configure_model
        return configure_model(core, policy=policy, training=training, capture=capture)
    if policy == "generated":
        raise ValueError("Generated operators require NPU FP32; CPU is a reference backend")
    from .reference_u_residency import restore_model_reference_u
    restore_model_reference_u(core)
    for module in core.modules():
        module._ascend_hp = False
    core._fused_direction_harmonics = False
    return {"requested": policy, "selected": "reference", "backend": "reference",
            "reasons": [], "counts": {"interactions": len(getattr(core, "interactions", ()))}}


def bond_first_projection(a, b, projector, angular_backend="torch"):
    """Apply a shared geometry matrix without broadcasting it over channels."""
    if angular_backend != "torch":
        raise ValueError("The Ascend library has no CUDA angular backend")
    magnetic = torch.matmul(b.squeeze(-2), projector).reshape(
        *b.shape[:-2], a.shape[-1], projector.shape[-1] // a.shape[-1])
    return torch.matmul(a, magnetic)
