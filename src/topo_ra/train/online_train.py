from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch

from topo_ra.data.aloft_extension import extend_wind_from_top_level
from topo_ra.data.fuxi_case_dataset import FuXiCaseDataset
from topo_ra.data.replay_buffer import ReplayBuffer
from topo_ra.data.synthetic_dataset import make_synthetic_sample
from topo_ra.data.vertical_interp import FU_XI_AGL_LEVELS
from topo_ra.eval.metrics import compute_metrics
from topo_ra.models.losses import offline_distillation_loss, online_weak_loss
from topo_ra.models.projection import average_to_coarse
from topo_ra.models.topo_ra import TopoRA
from topo_ra.utils.checkpoint import load_checkpoint, load_model_state, model_from_config
from topo_ra.utils.config import load_config
from topo_ra.utils.device import resolve_device
from topo_ra.utils.io import append_csv_row, ensure_dir
from topo_ra.utils.seeding import seed_everything


VERTICAL_METRICS = ["requested_level_count", "valid_level_count", "valid_height_min_m", "valid_height_max_m"]
LIVE_METRICS = [
    *VERTICAL_METRICS,
    "target_available",
    "MAE_u",
    "MAE_v",
    "MAE_w",
    "speed_RMSE",
    "radial_PSD_error",
    "coarse_consistency_u",
    "coarse_consistency_v",
    "coarse_consistency_speed",
    "speed_p99",
    "divergence_proxy",
    "tile_boundary_jump",
    "nan_count",
    "inf_count",
    "nonfinite_count",
]

REPLAY_METRICS = [*VERTICAL_METRICS, "MAE_u", "MAE_v", "MAE_w", "speed_RMSE", "radial_PSD_error"]


def _sample_to_batch(sample: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
    return batch


def _metrics_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _print_before_after(title: str, keys: list[str], before: dict[str, float], after: dict[str, float]) -> None:
    print(title)
    print(f"{'metric':28s} {'before':>12s} {'after':>12s} {'delta':>12s}")
    for key in keys:
        b = before[key]
        a = after[key]
        print(f"{key:28s} {b:12.6g} {a:12.6g} {a - b:12.6g}")


def _candidate_accepted(
    before_live: dict[str, float],
    after_live: dict[str, float],
    before_replay: dict[str, float],
    after_replay: dict[str, float],
    cfg: dict[str, Any],
) -> bool:
    acceptance = cfg.get("acceptance", {})
    replay_tol = float(acceptance.get("replay_tolerance", 0.25))
    seam_tol = float(acceptance.get("seam_tolerance", 0.25))
    seam_abs_tol = float(acceptance.get("seam_abs_tolerance", 0.0))
    max_speed = float(acceptance.get("max_speed", 100.0))
    target_metric = acceptance.get("target_metric")
    if target_metric:
        if target_metric not in before_live:
            raise ValueError(f"acceptance.target_metric={target_metric!r} is not available in live metrics")
        if before_live.get("target_available", 0.0) != 1.0:
            raise ValueError("acceptance.target_metric requires live samples with target labels")
        target_tol = float(acceptance.get("target_tolerance", 0.0))
        live_ok = after_live[target_metric] <= before_live[target_metric] * (1.0 + target_tol) + 1e-6
    else:
        live_ok = (
            after_live["coarse_consistency_speed"]
            <= before_live["coarse_consistency_speed"] * (1.0 + 1e-3) + 1e-6
        )
    before_replay_mae = before_replay["MAE_u"] + before_replay["MAE_v"] + before_replay["MAE_w"]
    after_replay_mae = after_replay["MAE_u"] + after_replay["MAE_v"] + after_replay["MAE_w"]
    replay_ok = after_replay_mae <= before_replay_mae * (1.0 + replay_tol) + 1e-6
    # Reject any update whose prediction contains NaN or Inf. nonfinite_count is
    # measured on the raw prediction (see compute_metrics), so this gate is live.
    no_nonfinite = (
        after_live["nonfinite_count"] == 0.0 and after_replay["nonfinite_count"] == 0.0
    )
    speed_ok = after_live["speed_p99"] < max_speed
    seam_ok = (
        after_live["tile_boundary_jump"]
        <= before_live["tile_boundary_jump"] * (1.0 + seam_tol) + seam_abs_tol + 1e-6
    )
    return bool(live_ok and replay_ok and no_nonfinite and speed_ok and seam_ok)


def _z_out_from_config(config: dict[str, Any]) -> torch.Tensor | None:
    snapy_cfg = config.get("snapy", {})
    z_cfg = snapy_cfg.get("z_out", config.get("z_out"))
    if z_cfg is None:
        return None
    if isinstance(z_cfg, dict):
        if "values" in z_cfg:
            z_cfg = z_cfg["values"]
        else:
            start = float(z_cfg.get("start", z_cfg.get("min", 5.0)))
            stop = float(z_cfg.get("stop", z_cfg.get("max", 3000.0)))
            if "step" in z_cfg:
                step = float(z_cfg["step"])
                if step <= 0:
                    raise ValueError("snapy.z_out.step must be positive")
                return torch.arange(start, stop + step * 1e-6, step, dtype=torch.float32)
            count = int(z_cfg.get("count", z_cfg.get("nz", 0)))
            if count <= 0:
                raise ValueError("snapy.z_out must provide count/nz or step when values are omitted")
            return torch.linspace(start, stop, count, dtype=torch.float32)
    z_out = torch.as_tensor(z_cfg, dtype=torch.float32)
    if z_out.ndim != 1:
        raise ValueError("snapy.z_out must be a 1D list or a {start, stop, count} mapping")
    return z_out


def _batch_z_out(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    z_out = batch.get("z_out")
    if z_out is None:
        return None
    if z_out.ndim == 1:
        return z_out
    if z_out.ndim == 2 and z_out.shape[0] == 1:
        return z_out[0]
    raise ValueError("online z_out must be shared across the batch with shape [Z] or [1, Z]")


def _batch_valid_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    valid_mask = batch.get("valid_mask")
    if valid_mask is None:
        return None
    if valid_mask.ndim == 1:
        return valid_mask.to(dtype=torch.bool)
    if valid_mask.ndim == 2 and valid_mask.shape[0] == 1:
        return valid_mask[0].to(dtype=torch.bool)
    raise ValueError("online valid_mask must be shared across the batch with shape [Z] or [1, Z]")


def _batch_coarse_profile_heights(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    heights = batch.get("coarse_profile_heights")
    if heights is None:
        return None
    if heights.ndim == 1:
        return heights
    if heights.ndim == 2 and heights.shape[0] == 1:
        return heights[0]
    raise ValueError("coarse_profile_heights must be shared across the batch with shape [Zc] or [1, Zc]")


def _predict_for_batch(
    model: TopoRA,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    z_out = _batch_z_out(batch)
    if z_out is None:
        return (
            model(batch["static_30m"], batch["dynamic_native"], batch["canonical_uv_100m"], batch["coarse_dx"]),
            None,
        )
    pred, valid_mask = model.predict_at_heights(
        batch["static_30m"],
        batch["dynamic_native"],
        batch["canonical_uv_100m"],
        batch["coarse_dx"],
        z_out,
        coarse_profile=batch.get("coarse_profile"),
        coarse_profile_heights=_batch_coarse_profile_heights(batch),
    )
    sample_mask = _batch_valid_mask(batch)
    return pred, sample_mask.to(device=valid_mask.device) if sample_mask is not None else valid_mask


def _valid_vertical_view(
    pred: torch.Tensor,
    target: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if valid_mask is None:
        return pred, target
    mask = valid_mask.to(device=pred.device, dtype=torch.bool)
    if int(mask.sum().detach().cpu()) == 0:
        raise ValueError("No z_out levels fall inside the FuXi-supervised range")
    pred_valid = pred[:, :, mask]
    if target is None:
        return pred_valid, None
    if target.shape[2] == mask.numel():
        return pred_valid, target[:, :, mask]
    if target.shape[2] == pred_valid.shape[2]:
        return pred_valid, target
    raise ValueError(f"target vertical levels ({target.shape[2]}) do not match z_out ({mask.numel()}) or valid levels ({pred_valid.shape[2]})")


def _vertical_metric_tensors(
    pred: torch.Tensor,
    valid_mask: torch.Tensor | None,
    z_out: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if valid_mask is None:
        heights = torch.as_tensor(FU_XI_AGL_LEVELS, device=pred.device, dtype=pred.dtype)
        return {
            "requested_level_count": pred.new_tensor(float(pred.shape[2])),
            "valid_level_count": pred.new_tensor(float(pred.shape[2])),
            "valid_height_min_m": heights.min(),
            "valid_height_max_m": heights.max(),
        }

    mask = valid_mask.to(device=pred.device, dtype=torch.bool)
    z = torch.as_tensor(z_out, device=pred.device, dtype=pred.dtype)
    valid_z = z[mask]
    return {
        "requested_level_count": pred.new_tensor(float(mask.numel())),
        "valid_level_count": pred.new_tensor(float(mask.sum().detach().cpu())),
        "valid_height_min_m": valid_z.min(),
        "valid_height_max_m": valid_z.max(),
    }


def _metrics_for_prediction(
    pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None,
) -> dict[str, float]:
    target = batch.get("target")
    pred_eval, target_eval = _valid_vertical_view(pred, target, valid_mask)
    metrics = compute_metrics(pred_eval, target_eval, batch["dynamic_native"][:, :2])
    metrics["target_available"] = pred.new_tensor(1.0 if target is not None else 0.0)
    metrics.update(_vertical_metric_tensors(pred, valid_mask, _batch_z_out(batch)))
    return _metrics_float(metrics)


def _set_trainable_parameters(model: TopoRA, config: dict[str, Any]) -> list[torch.nn.Parameter]:
    training_cfg = config.get("training", {})
    if bool(training_cfg.get("z_head_only", False)):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.z_head.parameters():
            parameter.requires_grad_(True)
        return [parameter for parameter in model.z_head.parameters() if parameter.requires_grad]

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _supervised_online_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coarse_uv: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    loss, _ = offline_distillation_loss(
        pred,
        target,
        coarse_uv,
        lambda_fft=float(loss_cfg.get("lambda_fft", 0.0)),
        lambda_coarse=float(loss_cfg.get("lambda_coarse", 0.1)),
        lambda_grad=float(loss_cfg.get("lambda_grad", 0.0)),
        lambda_speed=float(loss_cfg.get("lambda_speed", 0.0)),
        lambda_psd=float(loss_cfg.get("lambda_psd", 0.0)),
        psd_level_idx=loss_cfg.get("psd_level_idx"),
    )
    return loss


def _make_online_sample(idx: int, dx: int, config: dict[str, Any], z_out: torch.Tensor | None) -> dict[str, torch.Tensor]:
    model_cfg = config.get("model", {})
    snapy_cfg = config.get("snapy", {})
    if snapy_cfg.get("truth_source") == "fuxi_top_extrapolated":
        case_offset = int(config.get("case_offset", snapy_cfg.get("case_offset", 0))) + idx
        ds = FuXiCaseDataset(
            root=config.get("data_root", snapy_cfg.get("data_root", "data/fuxi/dataset")),
            synthetic=False,
            length=1,
            case_offset=case_offset,
            static_channels=int(model_cfg.get("static_channels", 5)),
            dynamic_channels=int(model_cfg.get("dynamic_channels", 6)),
        )
        sample = ds[0]
        if z_out is not None:
            sample["z_out"] = z_out
            sample["target"] = extend_wind_from_top_level(
                sample["target"],
                z_out,
                alpha=float(snapy_cfg.get("profile_alpha", 0.12)),
                veer_max_deg=float(snapy_cfg.get("veer_max_deg", 10.0)),
                veer_reference_height_m=float(snapy_cfg.get("veer_reference_height_m", 3000.0)),
                w_decay_m=float(snapy_cfg.get("w_decay_m", 700.0)),
            ).float()
            sample["valid_mask"] = torch.ones_like(z_out, dtype=torch.bool)
            if bool(snapy_cfg.get("coarse_profile_from_target", False)):
                sample["coarse_profile"] = average_to_coarse(
                    sample["target"],
                    tuple(sample["dynamic_native"].shape[-2:]),
                ).float()
                sample["coarse_profile_heights"] = z_out.float()
        return sample

    return make_synthetic_sample(
        idx,
        dx=dx,
        static_channels=int(model_cfg.get("static_channels", 5)),
        dynamic_channels=int(model_cfg.get("dynamic_channels", 6)),
        z_out=z_out,
        target_mode=str(snapy_cfg.get("synthetic_truth", "fuxi_interp")),
    )


def guarded_online_update(
    model: TopoRA,
    optimizer: torch.optim.Optimizer,
    live: dict[str, torch.Tensor],
    replay_batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one guarded online update on a live batch, rolling back on rejection.

    Returns the acceptance decision plus before/after live and replay metrics.
    """
    loss_cfg = config.get("loss", {})
    training_cfg = config.get("training", {})
    gradient_steps_per_update = max(1, int(training_cfg.get("gradient_steps_per_update", 1)))
    weak_weight = float(loss_cfg.get("weak_weight", 1.0))
    live_supervised_weight = float(loss_cfg.get("live_supervised_weight", 0.0))
    replay_supervised_weight = float(loss_cfg.get("replay_supervised_weight", 1.0))

    model.eval()
    with torch.no_grad():
        before_live_pred, before_live_mask = _predict_for_batch(model, live)
        before_replay_pred, before_replay_mask = _predict_for_batch(model, replay_batch)
        before_live = _metrics_for_prediction(before_live_pred, live, before_live_mask)
        before_replay = _metrics_for_prediction(before_replay_pred, replay_batch, before_replay_mask)

    before_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    # Snapshot optimizer state too, so a rejected update leaves no trace in the
    # AdamW moment estimates (otherwise a bad gradient keeps biasing momentum
    # into the next tile even after the weights are rolled back). deepcopy is
    # required because optimizer.step() mutates the state tensors in place.
    before_optimizer_state = copy.deepcopy(optimizer.state_dict())
    model.train()
    for _ in range(gradient_steps_per_update):
        optimizer.zero_grad(set_to_none=True)
        live_pred, live_mask = _predict_for_batch(model, live)
        live_target = live.get("target")
        live_pred_loss, live_target_loss = _valid_vertical_view(live_pred, live_target, live_mask)
        weak, _ = online_weak_loss(
            live_pred_loss,
            live["dynamic_native"][:, :2],
            alpha=float(loss_cfg.get("alpha", 1.0)),
            beta=float(loss_cfg.get("beta", 0.01)),
            gamma=float(loss_cfg.get("gamma", 0.01)),
        )
        total_loss = weak_weight * weak
        if live_target_loss is not None and live_supervised_weight != 0.0:
            total_loss = total_loss + live_supervised_weight * _supervised_online_loss(
                live_pred_loss,
                live_target_loss,
                live["dynamic_native"][:, :2],
                loss_cfg,
            )

        replay_pred, replay_mask = _predict_for_batch(model, replay_batch)
        replay_pred_loss, replay_target_loss = _valid_vertical_view(replay_pred, replay_batch["target"], replay_mask)
        assert replay_target_loss is not None
        total_loss = total_loss + replay_supervised_weight * _supervised_online_loss(
            replay_pred_loss,
            replay_target_loss,
            replay_batch["dynamic_native"][:, :2],
            loss_cfg,
        )
        total_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        after_live_pred, after_live_mask = _predict_for_batch(model, live)
        after_replay_pred, after_replay_mask = _predict_for_batch(model, replay_batch)
        after_live = _metrics_for_prediction(after_live_pred, live, after_live_mask)
        after_replay = _metrics_for_prediction(after_replay_pred, replay_batch, after_replay_mask)

    accepted = _candidate_accepted(before_live, after_live, before_replay, after_replay, config)
    if not accepted:
        model.load_state_dict(before_state)
        optimizer.load_state_dict(before_optimizer_state)

    return {
        "accepted": accepted,
        "before_live": before_live,
        "after_live": after_live,
        "before_replay": before_replay,
        "after_replay": after_replay,
    }


def run_online_training(config: dict[str, Any]) -> Path:
    seed_everything(config.get("seed", 0))
    device = resolve_device(config)
    output_dir = ensure_dir(config.get("output_dir", "runs/online_fuxi_top_aloft_current"))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    model = model_from_config(config).to(device)
    init_from = config.get("init_from")
    if init_from:
        checkpoint = load_checkpoint(init_from, device)
        load_model_state(model, checkpoint["model"])
    trainable_parameters = _set_trainable_parameters(model, config)
    if not trainable_parameters:
        raise ValueError("No trainable parameters selected for online training")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(config.get("learning_rate", 2e-4)))
    z_out = _z_out_from_config(config)
    online_dx = int(config.get("coarse_dx", config.get("dx", 300)))

    replay = ReplayBuffer(capacity=32)
    for idx in range(int(config.get("replay_length", 2))):
        replay.add(_make_online_sample(idx, dx=online_dx, config=config, z_out=z_out))

    accepted_count = 0
    rejected_count = 0
    num_updates = int(config.get("num_updates", 1))

    for update_idx in range(1, num_updates + 1):
        live = _sample_to_batch(_make_online_sample(100 + update_idx, dx=online_dx, config=config, z_out=z_out), device)
        replay_batch = _sample_to_batch(replay.sample(1)[0], device)
        result = guarded_online_update(model, optimizer, live, replay_batch, config)
        accepted = result["accepted"]
        before_live, after_live = result["before_live"], result["after_live"]
        before_replay, after_replay = result["before_replay"], result["after_replay"]
        if accepted:
            accepted_count += 1
            checkpoint_path = checkpoint_dir / f"online_update_{update_idx:04d}.pt"
            torch.save({"model": model.state_dict(), "config": config, "update": update_idx}, checkpoint_path)
            replay.add(_make_online_sample(100 + update_idx, dx=online_dx, config=config, z_out=z_out))
        else:
            rejected_count += 1
            checkpoint_path = checkpoint_dir / "rejected_no_checkpoint.pt"

        print(f"[online update {update_idx:04d} | n_tiles=1 | coarse_dx={online_dx}]")
        print()
        _print_before_after("LIVE SNAPY METRICS", LIVE_METRICS, before_live, after_live)
        print()
        _print_before_after("TEACHER REPLAY VALIDATION", REPLAY_METRICS, before_replay, after_replay)
        print()
        print(f"decision: {'ACCEPT' if accepted else 'REJECT'}")
        print(f"checkpoint: {checkpoint_path}")

        row = {
            "update": update_idx,
            "accepted": int(accepted),
            "accepted_updates": accepted_count,
            "rejected_updates": rejected_count,
        }
        for key in LIVE_METRICS:
            row[f"live_before_{key}"] = before_live[key]
            row[f"live_after_{key}"] = after_live[key]
        for key in REPLAY_METRICS:
            row[f"replay_before_{key}"] = before_replay[key]
            row[f"replay_after_{key}"] = after_replay[key]
        append_csv_row(metrics_dir / "before_after.csv", row)

    torch.save({"model": model.state_dict(), "config": config}, checkpoint_dir / "last.pt")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TopoRA online training as a snapy sidecar.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_online_training(load_config(args.config))


if __name__ == "__main__":
    main()
