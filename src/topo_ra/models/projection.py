from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_DX_TO_GRID = {1000: 9, 500: 18, 300: 30}


def coarse_grid_size(dx: int | float) -> int:
    rounded = int(round(float(dx)))
    if rounded not in SUPPORTED_DX_TO_GRID:
        raise ValueError(f"Unsupported coarse dx={dx}; expected one of {sorted(SUPPORTED_DX_TO_GRID)}")
    return SUPPORTED_DX_TO_GRID[rounded]


def adaptive_avg_pool2d_compat(x: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    """``F.adaptive_avg_pool2d`` with a CPU fallback for non-divisible sizes on MPS.

    MPS only implements adaptive pooling when input sizes divide the output
    sizes; autograd still flows through the ``.cpu()`` round-trip.
    """
    if x.device.type == "mps" and (x.shape[-2] % output_hw[0] != 0 or x.shape[-1] % output_hw[1] != 0):
        return F.adaptive_avg_pool2d(x.cpu(), output_hw).to(x.device)
    return F.adaptive_avg_pool2d(x, output_hw)


def average_to_coarse(fine: torch.Tensor, coarse_hw: tuple[int, int]) -> torch.Tensor:
    """Average fine fields to a target coarse grid.

    For integer block sizes, this uses an exact reshape mean. For non-integer
    grids such as 18 x 18 from 300 x 300, it falls back to adaptive averaging.
    """
    if fine.ndim < 2:
        raise ValueError("fine tensor must have at least two spatial dimensions")
    height, width = fine.shape[-2:]
    coarse_h, coarse_w = coarse_hw
    if height % coarse_h == 0 and width % coarse_w == 0:
        bh = height // coarse_h
        bw = width // coarse_w
        view = fine.reshape(*fine.shape[:-2], coarse_h, bh, coarse_w, bw)
        return view.mean(dim=(-3, -1))
    flat = fine.reshape(-1, 1, height, width)
    pooled = adaptive_avg_pool2d_compat(flat, coarse_hw)
    return pooled.reshape(*fine.shape[:-2], coarse_h, coarse_w)


def expand_coarse_delta(delta: torch.Tensor, fine_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(delta, size=fine_hw, mode="nearest")


def project_to_coarse_consistency(pred: torch.Tensor, coarse_uv: torch.Tensor, dx: int | float = 300) -> torch.Tensor:
    """Adjust u/v predictions so fine-grid averages match native coarse u/v.

    Shapes:
      pred: [B, 3, 27, 300, 300]
      coarse_uv: [B, 2, Hc, Wc]
    """
    if pred.ndim != 5 or pred.shape[1] < 2:
        raise ValueError("pred must have shape [B, >=2, Z, H, W]")
    if coarse_uv.ndim != 4 or coarse_uv.shape[1] != 2:
        raise ValueError("coarse_uv must have shape [B, 2, Hc, Wc]")
    if pred.shape[0] != coarse_uv.shape[0]:
        raise ValueError("pred and coarse_uv batch sizes must match")

    expected_grid = coarse_grid_size(dx)
    if coarse_uv.shape[-2:] != (expected_grid, expected_grid):
        raise ValueError(f"coarse_uv shape does not match dx={dx}: got {tuple(coarse_uv.shape[-2:])}")

    corrected = pred.clone()
    batch, _, levels, fine_h, fine_w = pred.shape
    uv = corrected[:, :2].reshape(batch * 2 * levels, 1, fine_h, fine_w)
    avg = average_to_coarse(uv, coarse_uv.shape[-2:]).reshape(batch, 2, levels, *coarse_uv.shape[-2:])
    target = coarse_uv.unsqueeze(2).expand_as(avg)
    delta = (target - avg).reshape(batch * 2 * levels, 1, *coarse_uv.shape[-2:])
    correction = expand_coarse_delta(delta, (fine_h, fine_w)).reshape(batch, 2, levels, fine_h, fine_w)
    corrected[:, :2] = corrected[:, :2] + correction
    return corrected


class CoarseConsistencyProjection(nn.Module):
    def forward(self, pred: torch.Tensor, coarse_uv: torch.Tensor, dx: int | float = 300) -> torch.Tensor:
        return project_to_coarse_consistency(pred, coarse_uv, dx=dx)


def supported_dx_from_grid(height: int, width: int) -> int:
    if height != width:
        raise ValueError(f"Coarse grid must be square, got {height} x {width}")
    for dx, grid in SUPPORTED_DX_TO_GRID.items():
        if grid == height:
            return dx
    size_m = 9000.0 / float(height)
    if not math.isclose(size_m, round(size_m)):
        raise ValueError(f"Cannot infer supported dx from grid {height} x {width}")
    return int(round(size_m))
