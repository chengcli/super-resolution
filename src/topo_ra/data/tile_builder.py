from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from topo_ra.models.projection import average_to_coarse, coarse_grid_size

FINE_TILE_SIZE = 300
FINE_DX_M = 30
TILE_SIZE_M = 9000
SUPPORTED_COARSE_DX = (300, 500, 1000)


@dataclass(frozen=True)
class TileMetadata:
    tile_id: str
    time: str | None
    dx: int


def conservative_resample_2d(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2] >= out_hw[0] and x.shape[-1] >= out_hw[1]:
        return average_to_coarse(x, out_hw)
    flat = x.reshape(-1, 1, *x.shape[-2:])
    up = F.interpolate(flat, size=out_hw, mode="bilinear", align_corners=False)
    return up.reshape(*x.shape[:-2], *out_hw)


def build_model_input(
    static_30m: torch.Tensor,
    dynamic_fields: torch.Tensor,
    dx: int = 300,
    tile_id: str = "synthetic-0000",
    time: str | None = None,
) -> dict[str, Any]:
    """Build one model input dictionary for a fixed 9 km x 9 km tile.

    static_30m: [C_static, 300, 300]
    dynamic_fields: either [C_dynamic, 300, 300] fine fields or native
      [C_dynamic, Hc, Wc] fields matching `dx`.
    """
    if dx not in SUPPORTED_COARSE_DX:
        raise ValueError(f"dx must be one of {SUPPORTED_COARSE_DX}, got {dx}")
    if static_30m.ndim != 3 or static_30m.shape[-2:] != (FINE_TILE_SIZE, FINE_TILE_SIZE):
        raise ValueError("static_30m must have shape [C_static, 300, 300]")
    if dynamic_fields.ndim != 3:
        raise ValueError("dynamic_fields must have shape [C_dynamic, H, W]")

    grid = coarse_grid_size(dx)
    if dynamic_fields.shape[-2:] == (grid, grid):
        dynamic_native = dynamic_fields
    else:
        dynamic_native = conservative_resample_2d(dynamic_fields, (grid, grid))

    canonical_uv_100m = conservative_resample_2d(dynamic_native[:2], (9, 9))
    return {
        "static_30m": static_30m,
        "dynamic_native": dynamic_native,
        "canonical_uv_100m": canonical_uv_100m,
        "coarse_dx": torch.tensor(float(dx), dtype=torch.float32),
        "metadata": TileMetadata(tile_id=tile_id, time=time, dx=dx),
    }


def block_average_2d(fine: torch.Tensor, dx: int = 300) -> torch.Tensor:
    grid = coarse_grid_size(dx)
    return average_to_coarse(fine, (grid, grid))
