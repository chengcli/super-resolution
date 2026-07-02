from __future__ import annotations

import torch

from topo_ra.eval.spectra import high_frequency_energy_ratio, radial_psd_error
from topo_ra.models.losses import divergence_proxy_loss
from topo_ra.models.projection import average_to_coarse


def _finite_for_metrics(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _rmse(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2))


def _mae(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x))


def wind_speed(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2).clamp_min(0.0))


def wind_direction_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dir = torch.atan2(pred[:, 1], pred[:, 0])
    target_dir = torch.atan2(target[:, 1], target[:, 0])
    diff = torch.atan2(torch.sin(pred_dir - target_dir), torch.cos(pred_dir - target_dir)).abs()
    return diff.mean() * (180.0 / torch.pi)


def profile_rmse_by_height(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = pred - target
    return torch.sqrt(torch.mean(err**2, dim=(0, 1, 3, 4))).mean()


def coarse_consistency(pred: torch.Tensor, coarse_uv: torch.Tensor | None) -> dict[str, torch.Tensor]:
    if coarse_uv is None:
        coarse_uv = average_to_coarse(pred[:, :2, 19], (30, 30)).detach()
    batch, _, levels, height, width = pred.shape
    uv = pred[:, :2].reshape(batch * 2 * levels, 1, height, width)
    avg = average_to_coarse(uv, coarse_uv.shape[-2:]).reshape(batch, 2, levels, *coarse_uv.shape[-2:])
    target = coarse_uv.unsqueeze(2).expand_as(avg)
    du = avg[:, 0] - target[:, 0]
    dv = avg[:, 1] - target[:, 1]
    speed_avg = torch.sqrt((avg[:, 0] ** 2 + avg[:, 1] ** 2).clamp_min(0.0))
    speed_target = torch.sqrt((target[:, 0] ** 2 + target[:, 1] ** 2).clamp_min(0.0))
    return {
        "coarse_consistency_u": _mae(du),
        "coarse_consistency_v": _mae(dv),
        "coarse_consistency_speed": _mae(speed_avg - speed_target),
    }


def tile_boundary_jump(pred: torch.Tensor) -> torch.Tensor:
    top = (pred[..., 0, :] - pred[..., 1, :]).abs().mean()
    bottom = (pred[..., -1, :] - pred[..., -2, :]).abs().mean()
    left = (pred[..., :, 0] - pred[..., :, 1]).abs().mean()
    right = (pred[..., :, -1] - pred[..., :, -2]).abs().mean()
    return (top + bottom + left + right) / 4.0


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor | None = None,
    coarse_uv: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    # Count non-finite values on the RAW prediction, before nan_to_num sanitizes
    # them away. These counts feed the online accept/reject safety gate, so they
    # must reflect the model's actual output rather than the cleaned tensor.
    nan_count = torch.isnan(pred).sum().to(torch.float32)
    inf_count = torch.isinf(pred).sum().to(torch.float32)
    nonfinite_count = nan_count + inf_count

    pred = _finite_for_metrics(pred)
    if target is None:
        target = pred.detach()
    else:
        target = _finite_for_metrics(target)
    err = pred - target
    speed_err = wind_speed(pred) - wind_speed(target)
    speed = wind_speed(pred)
    metrics = {
        "MAE_u": _mae(err[:, 0]),
        "MAE_v": _mae(err[:, 1]),
        "MAE_w": _mae(err[:, 2]),
        "RMSE_u": _rmse(err[:, 0]),
        "RMSE_v": _rmse(err[:, 1]),
        "RMSE_w": _rmse(err[:, 2]),
        "speed_RMSE": _rmse(speed_err),
        "wind_direction_MAE": wind_direction_mae(pred, target),
        "profile_RMSE_by_height": profile_rmse_by_height(pred, target),
        "divergence_proxy": divergence_proxy_loss(pred),
        "tile_boundary_jump": tile_boundary_jump(pred),
        "speed_p50": torch.quantile(speed.flatten(), 0.50),
        "speed_p90": torch.quantile(speed.flatten(), 0.90),
        "speed_p95": torch.quantile(speed.flatten(), 0.95),
        "speed_p99": torch.quantile(speed.flatten(), 0.99),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_count": nonfinite_count,
        "radial_PSD_error": radial_psd_error(pred, target),
        "high_frequency_energy_ratio": high_frequency_energy_ratio(pred),
    }
    metrics.update(coarse_consistency(pred, coarse_uv))
    return metrics


REQUIRED_METRICS = {
    "MAE_u",
    "MAE_v",
    "MAE_w",
    "RMSE_u",
    "RMSE_v",
    "RMSE_w",
    "speed_RMSE",
    "wind_direction_MAE",
    "profile_RMSE_by_height",
    "coarse_consistency_u",
    "coarse_consistency_v",
    "coarse_consistency_speed",
    "divergence_proxy",
    "tile_boundary_jump",
    "speed_p50",
    "speed_p90",
    "speed_p95",
    "speed_p99",
    "nan_count",
    "inf_count",
    "nonfinite_count",
    "radial_PSD_error",
    "high_frequency_energy_ratio",
}
