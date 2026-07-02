from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from topo_ra.models.projection import average_to_coarse


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def _downsample_for_fft(x: torch.Tensor, max_hw: int = 64) -> torch.Tensor:
    height, width = x.shape[-2:]
    if max(height, width) <= max_hw:
        return x
    flat = x.reshape(-1, 1, height, width)
    scale = max_hw / float(max(height, width))
    out_hw = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    pooled = F.interpolate(flat, size=out_hw, mode="area")
    return pooled.reshape(*x.shape[:-2], *out_hw)


def fft_magnitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_small = _downsample_for_fft(pred)
    target_small = _downsample_for_fft(target)
    pred_mag = torch.fft.rfft2(pred_small, dim=(-2, -1)).abs()
    target_mag = torch.fft.rfft2(target_small, dim=(-2, -1)).abs()
    return F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))


def psd_power_loss(pred: torch.Tensor, target: torch.Tensor, level_idx: int | None = None) -> torch.Tensor:
    """Log-power spectral loss matching the radial-PSD diagnostic form.

    If ``level_idx`` is provided, the loss is computed only on that vertical
    level. For FuXi-CFD this is useful for directly optimizing the 100 m texture
    diagnostic while leaving the physical output units unchanged.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must match: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if level_idx is not None:
        if pred.ndim != 5:
            raise ValueError("level_idx requires tensors shaped [B, C, Z, H, W]")
        pred = pred[:, :, int(level_idx)]
        target = target[:, :, int(level_idx)]
    pred_small = _downsample_for_fft(torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0))
    target_small = _downsample_for_fft(torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0))
    pred_power = torch.fft.rfft2(pred_small, dim=(-2, -1)).abs().pow(2)
    target_power = torch.fft.rfft2(target_small, dim=(-2, -1)).abs().pow(2)
    return F.l1_loss(torch.log1p(pred_power), torch.log1p(target_power))


def coarse_consistency_loss(pred: torch.Tensor, coarse_uv: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 5:
        raise ValueError("pred must have shape [B, 3, Z, H, W]")
    batch, _, levels, height, width = pred.shape
    uv = pred[:, :2].reshape(batch * 2 * levels, 1, height, width)
    avg = average_to_coarse(uv, coarse_uv.shape[-2:]).reshape(batch, 2, levels, *coarse_uv.shape[-2:])
    target = coarse_uv.unsqueeze(2).expand_as(avg)
    return F.mse_loss(avg, target)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 match of horizontal finite-difference gradients at full resolution.

    Unlike ``fft_magnitude_loss`` (which downsamples to 64x64 and therefore cannot
    see the finest scales), this operates directly on the 30 m grid, so it
    penalizes missing sharp, terrain-locked structure. Pass normalized components
    for balanced weighting across u/v/w.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must match: {tuple(pred.shape)} vs {tuple(target.shape)}")
    dpx = pred[..., :, 1:] - pred[..., :, :-1]
    dtx = target[..., :, 1:] - target[..., :, :-1]
    dpy = pred[..., 1:, :] - pred[..., :-1, :]
    dty = target[..., 1:, :] - target[..., :-1, :]
    return (dpx - dtx).abs().mean() + (dpy - dty).abs().mean()


def speed_charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss on physical wind-speed magnitude."""
    pred_speed = torch.sqrt((pred[:, 0] ** 2 + pred[:, 1] ** 2 + pred[:, 2] ** 2).clamp_min(0.0))
    target_speed = torch.sqrt((target[:, 0] ** 2 + target[:, 1] ** 2 + target[:, 2] ** 2).clamp_min(0.0))
    return charbonnier_loss(pred_speed, target_speed, eps=eps)


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    dx = pred[..., :, 1:] - pred[..., :, :-1]
    dy = pred[..., 1:, :] - pred[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def divergence_proxy_loss(pred: torch.Tensor) -> torch.Tensor:
    """A light finite-difference proxy using horizontal u/v gradients."""
    u = pred[:, 0]
    v = pred[:, 1]
    dudx = u[..., :, 1:] - u[..., :, :-1]
    dvdy = v[..., 1:, :] - v[..., :-1, :]
    dudx = dudx[..., :-1, :]
    dvdy = dvdy[..., :, :-1]
    return (dudx + dvdy).abs().mean()


def offline_distillation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coarse_uv: torch.Tensor,
    lambda_fft: float = 0.05,
    lambda_coarse: float = 0.1,
    *,
    normalizer: Any | None = None,
    lambda_grad: float = 0.0,
    lambda_speed: float = 0.0,
    lambda_psd: float = 0.0,
    psd_level_idx: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Offline distillation objective.

    When ``normalizer`` is provided, the Charbonnier, FFT, and gradient terms are
    computed in per-variable standardized space so u/v/w contribute comparably.
    The coarse-consistency and speed terms stay in physical units. ``lambda_grad``
    enables the full-resolution gradient loss, ``lambda_speed`` directly
    optimizes physical wind-speed magnitude error, and ``lambda_psd`` matches
    log-power spectra in physical units.
    """
    if normalizer is not None:
        pred_n = normalizer.normalize(pred)
        target_n = normalizer.normalize(target)
    else:
        pred_n, target_n = pred, target

    charb = charbonnier_loss(pred_n, target_n)
    zero = pred.new_zeros(())
    fft = fft_magnitude_loss(pred_n, target_n) if lambda_fft != 0.0 else zero
    grad = gradient_loss(pred_n, target_n) if lambda_grad != 0.0 else zero
    coarse = coarse_consistency_loss(pred, coarse_uv) if lambda_coarse != 0.0 else zero
    speed = speed_charbonnier_loss(pred, target) if lambda_speed != 0.0 else zero
    psd = psd_power_loss(pred, target, level_idx=psd_level_idx) if lambda_psd != 0.0 else zero
    total = charb + lambda_fft * fft + lambda_grad * grad + lambda_coarse * coarse + lambda_speed * speed + lambda_psd * psd
    return total, {"charbonnier": charb, "fft": fft, "grad": grad, "coarse": coarse, "speed_loss": speed, "psd": psd}


def online_weak_loss(
    pred: torch.Tensor,
    coarse_uv: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.01,
    gamma: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    coarse = coarse_consistency_loss(pred, coarse_uv)
    smooth = smoothness_loss(pred)
    div = divergence_proxy_loss(pred)
    total = alpha * coarse + beta * smooth + gamma * div
    return total, {"live_coarse": coarse, "smoothness": smooth, "divergence": div}
