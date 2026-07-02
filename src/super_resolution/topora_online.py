"""Online-train TopoRA as a sidecar beside a live snapy two-resolution run.

Each accepted snapy step yields the low-resolution coarse state; the sidecar
converts it into a TopoRA live sample (snapy wind forcing over a local 30 m
terrain tile) and applies one guarded online update. Teacher replay samples
anchor every update so live adaptation cannot silently destroy distilled skill.

Requires the ``topo-ra`` package (install with ``pip install -e /path/to/TopoRA``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from topo_ra.data.fuxi_case_dataset import FuXiCaseDataset
from topo_ra.data.replay_buffer import ReplayBuffer
from topo_ra.data.snapy_reader import HYDRO_W_ORDER
from topo_ra.train.distill import _load_checkpoint, _load_model_state, _model_from_config
from topo_ra.train.online_train import (
    LIVE_METRICS,
    REPLAY_METRICS,
    _print_before_after,
    _sample_to_batch,
    _set_trainable_parameters,
    _z_out_from_config,
    guarded_online_update,
)
from topo_ra.utils.config import load_config
from topo_ra.utils.device import resolve_device
from topo_ra.utils.io import append_csv_row, ensure_dir
from topo_ra.utils.seeding import seed_everything

from .runner import TwoResolutionRunner


def block_uvw(hydro_w: torch.Tensor, nghost: int) -> torch.Tensor:
    """Extract the interior u/v/w columns from one snapy block ``hydro_w`` buffer.

    Args:
      hydro_w: ``[var, x3, x2, x1]`` primitive-variable block tensor, variables
        ordered per ``HYDRO_W_ORDER``. For shallow-water runs ``x1 == 1``.
      nghost: ghost cells to crop from each horizontal (x3/x2) side. The x1
        axis is only cropped when it carries more than one cell.
    Returns:
      ``[3, Zc, H, W]`` float32 tensor ordered u, v, w with ``Zc`` vertical
      (x1) levels and ``H/W`` the interior x3/x2 extents.
    """
    if hydro_w.ndim != 4:
        raise ValueError("hydro_w must have shape [var, x3, x2, x1]")
    mapping = {name: i for i, name in enumerate(HYDRO_W_ORDER[: hydro_w.shape[0]])}
    missing = [name for name in ("u", "v", "w") if name not in mapping]
    if missing:
        raise KeyError(f"snapy hydro_w is missing variables required for uvw: {missing}")
    interior = hydro_w
    if nghost > 0:
        interior = interior[:, nghost:-nghost, nghost:-nghost]
        if interior.shape[3] > 2 * nghost + 1:
            interior = interior[..., nghost:-nghost]
    uvw = torch.stack((interior[mapping["u"]], interior[mapping["v"]], interior[mapping["w"]]))
    return uvw.permute(0, 3, 1, 2).to(torch.float32)


def snapy_state_to_sample(
    low_state: list[dict[str, torch.Tensor]],
    *,
    nghost: int,
    static_30m: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Build a TopoRA online live sample from a snapy low-resolution MeshState.

    The snapy coarse wind becomes the dynamic forcing (``dynamic_native`` and
    ``canonical_uv_100m``); the caller supplies the 30 m static terrain tile.
    The live sample carries no target: online updates use the weak
    coarse-consistency loss plus teacher replay supervision.
    """
    snapy_cfg = config.get("snapy", {})
    block_index = int(snapy_cfg.get("block_index", 0))
    velocity_scale = float(snapy_cfg.get("velocity_scale", 1.0))
    coarse_cells = int(snapy_cfg.get("coarse_cells", 9))
    hydro_w = low_state[block_index]["hydro_w"]
    if hydro_w.ndim == 5:
        hydro_w = hydro_w[-1]
    uvw = block_uvw(hydro_w.detach().cpu(), nghost) * velocity_scale
    if not torch.isfinite(uvw).all():
        raise ValueError("snapy low-resolution state contains non-finite u/v/w values")
    coarse = F.adaptive_avg_pool2d(uvw, (coarse_cells, coarse_cells))

    reference_level = int(snapy_cfg.get("reference_level", -1))
    uv = coarse[:2, reference_level]
    dynamic_fields = [uv[0], uv[1], coarse[2, reference_level]]
    dynamic_channels = int(config.get("model", {}).get("dynamic_channels", 6))
    while len(dynamic_fields) < dynamic_channels:
        dynamic_fields.append(torch.zeros_like(uv[0]))
    sample = {
        "static_30m": static_30m.float(),
        "dynamic_native": torch.stack(dynamic_fields[:dynamic_channels]).float(),
        "canonical_uv_100m": uv.float(),
        "coarse_dx": torch.tensor(float(config.get("coarse_dx", 1000)), dtype=torch.float32),
    }
    if coarse.shape[1] >= 2:
        sample["coarse_profile"] = coarse.float()
        heights = snapy_cfg.get("profile_heights")
        if heights is not None:
            heights_tensor = torch.as_tensor(heights, dtype=torch.float32)
            if heights_tensor.numel() != coarse.shape[1]:
                raise ValueError("snapy.profile_heights length must match the snapy x1 level count")
        else:
            heights_tensor = torch.arange(coarse.shape[1], dtype=torch.float32)
        sample["coarse_profile_heights"] = heights_tensor
    return sample


def _build_runner(config: dict[str, Any]) -> TwoResolutionRunner:
    # Import snapy eagerly (instead of inside runner.run) so the float32
    # default-dtype reset in run_snapy_online happens after snapy's float64 switch.
    import snapy  # noqa: F401

    snapy_cfg = config.get("snapy", {})
    case_path = snapy_cfg.get("config")
    if not case_path:
        raise ValueError("snapy.config must point to a snapy shallow-water YAML case file")
    return TwoResolutionRunner(
        case_path,
        use_paddle_dist=bool(snapy_cfg.get("use_paddle_dist", False)),
        device=snapy_cfg.get("device", "cpu"),
    )


def _teacher_samples(config: dict[str, Any], count: int) -> list[dict[str, torch.Tensor]]:
    model_cfg = config.get("model", {})
    root = config.get("data_root")
    dataset = FuXiCaseDataset(
        root=root,
        synthetic=bool(config.get("synthetic", root is None)),
        length=max(1, count),
        static_channels=int(model_cfg.get("static_channels", 5)),
        dynamic_channels=int(model_cfg.get("dynamic_channels", 6)),
    )
    return [dataset[idx] for idx in range(len(dataset))]


def run_snapy_online(config: dict[str, Any], runner: Any | None = None) -> Path:
    seed_everything(config.get("seed", 0))
    if runner is None:
        runner = _build_runner(config)
    # Importing snapy switches the torch default dtype to float64; restore
    # float32 so TopoRA modules and samples are built at their trained precision.
    torch.set_default_dtype(torch.float32)
    device = resolve_device(config)
    output_dir = ensure_dir(config.get("output_dir", "runs/topora_online"))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")

    model = _model_from_config(config).to(device)
    init_from = config.get("init_from")
    if init_from:
        checkpoint = _load_checkpoint(init_from, device)
        _load_model_state(model, checkpoint["model"])
    trainable_parameters = _set_trainable_parameters(model, config)
    if not trainable_parameters:
        raise ValueError("No trainable parameters selected for online training")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(config.get("learning_rate", 2e-4)))
    z_out = _z_out_from_config(config)

    teacher = _teacher_samples(config, int(config.get("replay_length", 2)))
    replay = ReplayBuffer(capacity=32)
    replay.extend(teacher)
    static_tiles = [sample["static_30m"] for sample in teacher]

    num_updates = int(config.get("num_updates", 4))
    accepted_count = 0
    rejected_count = 0
    update_idx = 0
    print(f"snapy online sidecar: device={device}, num_updates={num_updates}")
    for step in runner.run(max_steps=num_updates):
        update_idx += 1
        static_30m = static_tiles[(update_idx - 1) % len(static_tiles)]
        live_sample = snapy_state_to_sample(
            step.low,
            nghost=runner.nghost,
            static_30m=static_30m,
            config=config,
        )
        if z_out is not None:
            live_sample["z_out"] = z_out
        live = _sample_to_batch(live_sample, device)
        replay_batch = _sample_to_batch(replay.sample(1)[0], device)
        result = guarded_online_update(model, optimizer, live, replay_batch, config)
        if result["accepted"]:
            accepted_count += 1
            checkpoint_path = checkpoint_dir / f"online_update_{update_idx:04d}.pt"
            torch.save({"model": model.state_dict(), "config": config, "update": update_idx}, checkpoint_path)
        else:
            rejected_count += 1
            checkpoint_path = checkpoint_dir / "rejected_no_checkpoint.pt"

        print(f"[snapy online update {update_idx:04d} | cycle={step.cycle} | t={step.time:.6g}]")
        print()
        _print_before_after("LIVE SNAPY METRICS", LIVE_METRICS, result["before_live"], result["after_live"])
        print()
        _print_before_after("TEACHER REPLAY VALIDATION", REPLAY_METRICS, result["before_replay"], result["after_replay"])
        print()
        print(f"decision: {'ACCEPT' if result['accepted'] else 'REJECT'}")
        print(f"checkpoint: {checkpoint_path}")

        row: dict[str, Any] = {
            "update": update_idx,
            "snapy_cycle": step.cycle,
            "snapy_time": step.time,
            "accepted": int(result["accepted"]),
            "accepted_updates": accepted_count,
            "rejected_updates": rejected_count,
        }
        for key in LIVE_METRICS:
            row[f"live_before_{key}"] = result["before_live"][key]
            row[f"live_after_{key}"] = result["after_live"][key]
        for key in REPLAY_METRICS:
            row[f"replay_before_{key}"] = result["before_replay"][key]
            row[f"replay_after_{key}"] = result["after_replay"][key]
        append_csv_row(metrics_dir / "before_after.csv", row)

    if update_idx == 0:
        raise RuntimeError("snapy runner produced no accepted steps; nothing to train on")
    torch.save({"model": model.state_dict(), "config": config}, checkpoint_dir / "last.pt")
    print(f"accepted {accepted_count} / rejected {rejected_count} updates")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TopoRA online training beside a live snapy case.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_snapy_online(load_config(args.config))


if __name__ == "__main__":
    main()
