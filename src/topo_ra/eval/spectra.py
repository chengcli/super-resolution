from __future__ import annotations

import torch
from topo_ra.models.projection import adaptive_avg_pool2d_compat


def _finite_for_spectra(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _small(x: torch.Tensor, max_hw: int = 64) -> torch.Tensor:
    height, width = x.shape[-2:]
    if max(height, width) <= max_hw:
        return x
    flat = x.reshape(-1, 1, height, width)
    scale = max_hw / float(max(height, width))
    out_hw = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    # Area interpolation lowers to adaptive_avg_pool2d; use the MPS-safe wrapper.
    return adaptive_avg_pool2d_compat(flat, out_hw).reshape(*x.shape[:-2], *out_hw)


def radial_psd_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_s = _small(_finite_for_spectra(pred))
    target_s = _small(_finite_for_spectra(target))
    pred_power = torch.fft.rfft2(pred_s, dim=(-2, -1)).abs().pow(2)
    target_power = torch.fft.rfft2(target_s, dim=(-2, -1)).abs().pow(2)
    return (torch.log1p(pred_power) - torch.log1p(target_power)).abs().mean()


def high_frequency_energy_ratio(x: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
    x_s = _small(_finite_for_spectra(x))
    power = torch.fft.rfft2(x_s, dim=(-2, -1)).abs().pow(2)
    height, width_r = power.shape[-2:]
    fy = torch.fft.fftfreq(height, device=x.device).abs().view(height, 1)
    fx = torch.fft.rfftfreq((width_r - 1) * 2, device=x.device).abs().view(1, width_r)
    radius = torch.sqrt(fx**2 + fy**2)
    mask = radius >= cutoff * radius.max().clamp_min(1e-12)
    high = power[..., mask].sum()
    total = power.sum().clamp_min(torch.finfo(power.dtype).eps)
    return high / total
