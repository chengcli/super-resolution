from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

Variables = dict[str, torch.Tensor]
MeshState = list[Variables]
PredictionMetadata = Mapping[str, Any]


class HighResPredictor(Protocol):
    """Predict a high-resolution Snapy state from a low-resolution state."""

    def predict(
        self, low_state: MeshState, *, metadata: PredictionMetadata
    ) -> MeshState:
        ...
