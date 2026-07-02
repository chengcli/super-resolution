from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from topo_ra.models.topo_ra import TopoRA


def model_from_config(config: dict[str, Any]) -> TopoRA:
    model_cfg = config.get("model", {})
    return TopoRA(
        static_channels=int(model_cfg.get("static_channels", 5)),
        dynamic_channels=int(model_cfg.get("dynamic_channels", 6)),
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        depth=int(model_cfg.get("depth", 4)),
        latent_size=int(model_cfg.get("latent_size", 30)),
        refine_channels=int(model_cfg.get("refine_channels", 16)),
        refine_depth=int(model_cfg.get("refine_depth", 2)),
        z_channels=int(model_cfg.get("z_channels", model_cfg.get("refine_channels", 16))),
        z_depth=int(model_cfg.get("z_depth", 2)),
        z_basis_rank=int(model_cfg.get("z_basis_rank", 4)),
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a TopoRA training checkpoint")
    return checkpoint


def load_model_state(model: TopoRA, state: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    incompatible_unexpected = []
    for key, value in state.items():
        if key not in model_state:
            if key.startswith("z_head."):
                continue
            incompatible_unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            if key.startswith("z_head."):
                continue
            raise RuntimeError(
                f"Checkpoint is incompatible: shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)}, model={tuple(model_state[key].shape)}"
            )
        compatible_state[key] = value

    result = model.load_state_dict(compatible_state, strict=False)
    unexpected = [*incompatible_unexpected, *list(result.unexpected_keys)]
    allowed_missing_prefixes = ("z_head.",)
    missing = [key for key in result.missing_keys if not key.startswith(allowed_missing_prefixes)]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint is incompatible: missing={missing}, unexpected={unexpected}")
