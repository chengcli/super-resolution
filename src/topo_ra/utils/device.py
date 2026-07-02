from __future__ import annotations

from typing import Any

import torch


def resolve_device(config: dict[str, Any] | None = None) -> torch.device:
    """Pick the training device: explicit config ``device`` > CUDA > MPS > CPU."""
    preference = (config or {}).get("device")
    if preference:
        return torch.device(str(preference))
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")
