from __future__ import annotations

import numpy as np
import torch

FU_XI_AGL_LEVELS = [
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
    100.0,
    106.5,
    114.95,
    125.94,
    140.22,
    158.78,
    182.91,
    214.29,
]


def interpolate_to_agl(
    values: np.ndarray | torch.Tensor,
    source_heights: np.ndarray | torch.Tensor,
    target_heights: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Interpolate values along the third-from-last vertical axis.

    Expected shape is `[..., Z, Y, X]`, which covers snapy arrays such as
    `[var, x3, x2, x1]` after mapping `x3` to height.
    """
    if target_heights is None:
        target_heights = np.asarray(FU_XI_AGL_LEVELS, dtype=np.float32)
    if isinstance(values, torch.Tensor):
        src = torch.as_tensor(source_heights, device=values.device, dtype=values.dtype)
        tgt = torch.as_tensor(target_heights, device=values.device, dtype=values.dtype)
        if values.shape[-3] != src.numel():
            raise ValueError(f"Vertical axis length {values.shape[-3]} does not match {src.numel()} source heights")
        tgt_clamped = tgt.clamp(float(src.min()), float(src.max()))
        flat = values.movedim(-3, 0).reshape(src.numel(), -1)
        idx = torch.searchsorted(src, tgt_clamped)
        idx = idx.clamp(1, src.numel() - 1)
        lo = idx - 1
        hi = idx
        weight = ((tgt_clamped - src[lo]) / (src[hi] - src[lo]).clamp_min(torch.finfo(values.dtype).eps)).view(-1, 1)
        interp = flat[lo] * (1.0 - weight) + flat[hi] * weight
        return interp.reshape(tgt.numel(), *values.movedim(-3, 0).shape[1:]).movedim(0, -3)

    arr = np.asarray(values)
    src_np = np.asarray(source_heights, dtype=np.float32)
    tgt_np = np.asarray(target_heights, dtype=np.float32)
    if arr.shape[-3] != src_np.size:
        raise ValueError(f"Vertical axis length {arr.shape[-3]} does not match {src_np.size} source heights")
    moved = np.moveaxis(arr, -3, 0)
    flat = moved.reshape(src_np.size, -1)
    out = np.empty((tgt_np.size, flat.shape[1]), dtype=arr.dtype)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(tgt_np, src_np, flat[:, col])
    shaped = out.reshape((tgt_np.size, *moved.shape[1:]))
    return np.moveaxis(shaped, 0, -3)


def extract_uv_100m(uv_values: np.ndarray | torch.Tensor, source_heights: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Extract u/v at 100 m AGL from values shaped `[2, Z, Y, X]`."""
    target = torch.tensor([100.0]) if isinstance(uv_values, torch.Tensor) else np.asarray([100.0], dtype=np.float32)
    out = interpolate_to_agl(uv_values, source_heights, target)
    return out[:, 0]


def heights_in_range(
    target_heights: np.ndarray | torch.Tensor,
    source_heights: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Mark which target heights fall inside the source's non-extrapolated span.

    ``interpolate_to_agl`` silently clamps queries outside
    ``[min(source_heights), max(source_heights)]`` to the nearest edge value, so
    it never raises for an out-of-range query. This helper gives callers an
    explicit ``valid_level_mask``-style signal so heights that would require
    extrapolation (e.g. anything above the public FuXi 214.29 m ceiling) can be
    gated or discarded rather than silently trusted as FuXi-distilled accuracy.

    Defaults ``source_heights`` to ``FU_XI_AGL_LEVELS`` since that is the only
    vertical range this repository currently supervises.
    """
    if source_heights is None:
        source_heights = FU_XI_AGL_LEVELS
    if isinstance(target_heights, torch.Tensor):
        src = torch.as_tensor(source_heights, device=target_heights.device, dtype=target_heights.dtype)
        return (target_heights >= src.min()) & (target_heights <= src.max())
    tgt = np.asarray(target_heights, dtype=np.float32)
    src = np.asarray(source_heights, dtype=np.float32)
    return (tgt >= src.min()) & (tgt <= src.max())
