from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .types import HighResPredictor, MeshState, PredictionMetadata


def clone_state(state: MeshState) -> MeshState:
    return [{name: tensor.detach().clone() for name, tensor in block.items()} for block in state]


@dataclass(frozen=True)
class BilinearUpscaler:
    """Factor-two baseline predictor for Snapy block tensors."""

    nghost: int
    variable_names: tuple[str, ...] = ("hydro_w",)

    def __init__(
        self, nghost: int, variable_names: Iterable[str] = ("hydro_w",)
    ) -> None:
        object.__setattr__(self, "nghost", int(nghost))
        object.__setattr__(self, "variable_names", tuple(variable_names))

    def predict(
        self, low_state: MeshState, *, metadata: PredictionMetadata
    ) -> MeshState:
        try:
            from paddle.restart_resize import resize_spatial_tensor
        except ModuleNotFoundError:
            resize_spatial_tensor = _resize_spatial_tensor

        selected = set(self.variable_names)
        prediction: MeshState = []
        for block in low_state:
            predicted_block = {}
            for name, tensor in block.items():
                if name in selected and _is_spatial_float_tensor(tensor):
                    predicted_block[name] = resize_spatial_tensor(
                        tensor, mode="refine", nghost=self.nghost
                    )
                else:
                    predicted_block[name] = tensor.detach().clone()
            prediction.append(predicted_block)
        return prediction


def _is_spatial_float_tensor(tensor: torch.Tensor) -> bool:
    return tensor.ndim >= 3 and tensor.is_floating_point()


def _resize_spatial_tensor(
    tensor: torch.Tensor, *, mode: str, nghost: int
) -> torch.Tensor:
    if mode != "refine":
        raise ValueError(f"unsupported resize mode: {mode}")
    del nghost
    height, width = tensor.shape[-3:-1]
    channels_first = tensor.movedim(-1, -3).reshape(-1, 1, height, width)
    resized = torch.nn.functional.interpolate(
        channels_first,
        size=(height * 2, width * 2),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(*tensor.shape[:-3], tensor.shape[-1], height * 2, width * 2).movedim(-3, -1)
