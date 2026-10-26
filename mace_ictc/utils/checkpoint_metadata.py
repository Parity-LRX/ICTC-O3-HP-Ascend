"""Select raw, EMA or SWA weights from MACE-ICTC checkpoints."""
from __future__ import annotations

from typing import Any, Mapping


def get_checkpoint_e3_state_dict(
    checkpoint: Mapping[str, Any] | None,
    *,
    prefer_ema: bool = True,
) -> tuple[Mapping[str, Any], str]:
    """Return the model state_dict to use from a checkpoint.

    New MACE-ICTC checkpoints may carry ``default_state_source`` to choose raw,
    EMA, or SWA weights for deployment. Older checkpoints keep the historical
    behavior: prefer EMA when present, otherwise use raw weights. The second
    return value is a short source label: ``"raw"``, ``"ema"``, or ``"swa"``.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping to resolve e3trans weights.")

    raw_state = checkpoint.get("e3trans_state_dict")
    ema_state = checkpoint.get("e3trans_ema_state_dict")
    swa_state = checkpoint.get("e3trans_swa_state_dict")
    states = {
        "raw": raw_state,
        "ema": ema_state,
        "swa": swa_state,
    }

    default_source = str(checkpoint.get("default_state_source", "") or "").lower()
    if default_source in states and isinstance(states[default_source], Mapping):
        return states[default_source], default_source

    if prefer_ema and isinstance(ema_state, Mapping):
        return ema_state, "ema"

    if isinstance(raw_state, Mapping):
        return raw_state, "raw"

    if isinstance(swa_state, Mapping):
        return swa_state, "swa"

    if isinstance(ema_state, Mapping):
        return ema_state, "ema"

    raise KeyError(
        "Checkpoint does not contain e3trans_state_dict, e3trans_ema_state_dict, "
        "or e3trans_swa_state_dict."
    )
