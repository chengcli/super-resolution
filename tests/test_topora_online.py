from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from super_resolution.topora_online import block_uvw, run_snapy_online, snapy_state_to_sample


def _fake_hydro_w(nx3: int = 8, nx2: int = 8, nx1: int = 1, nghost: int = 1) -> torch.Tensor:
    hydro_w = torch.full((4, nx3 + 2 * nghost, nx2 + 2 * nghost, nx1), float("nan"), dtype=torch.float64)
    interior = (slice(None), slice(nghost, -nghost), slice(nghost, -nghost), slice(None))
    hydro_w[interior] = torch.zeros(4, nx3, nx2, nx1, dtype=torch.float64)
    hydro_w[0][interior[1:]] = 8000.0
    hydro_w[2][interior[1:]] = 5.0  # u
    hydro_w[3][interior[1:]] = -2.0  # v
    return hydro_w


def test_block_uvw_crops_ghosts_and_maps_variables():
    uvw = block_uvw(_fake_hydro_w(), nghost=1)

    assert uvw.shape == (3, 1, 8, 8)
    assert uvw.dtype == torch.float32
    assert torch.isfinite(uvw).all()
    assert torch.allclose(uvw[0], torch.full((1, 8, 8), 5.0))
    assert torch.allclose(uvw[1], torch.full((1, 8, 8), -2.0))
    assert torch.allclose(uvw[2], torch.zeros(1, 8, 8))


def test_snapy_state_to_sample_builds_online_inputs():
    state = [{"hydro_w": _fake_hydro_w()}]
    static_30m = torch.zeros(5, 300, 300)
    config = {"model": {"dynamic_channels": 6}, "snapy": {"velocity_scale": 0.5}, "coarse_dx": 1000}

    sample = snapy_state_to_sample(state, nghost=1, static_30m=static_30m, config=config)

    assert sample["dynamic_native"].shape == (6, 9, 9)
    assert sample["canonical_uv_100m"].shape == (2, 9, 9)
    assert torch.allclose(sample["canonical_uv_100m"][0], torch.full((9, 9), 2.5))
    assert float(sample["coarse_dx"]) == 1000.0
    assert "coarse_profile" not in sample  # single shallow-water level


def test_snapy_state_to_sample_emits_profile_for_multilevel_state():
    state = [{"hydro_w": _fake_hydro_w(nx1=3)}]
    config = {
        "model": {"dynamic_channels": 6},
        "snapy": {"profile_heights": [100.0, 500.0, 1000.0]},
    }

    sample = snapy_state_to_sample(state, nghost=1, static_30m=torch.zeros(5, 300, 300), config=config)

    assert sample["coarse_profile"].shape == (3, 3, 9, 9)
    assert torch.allclose(sample["coarse_profile_heights"], torch.tensor([100.0, 500.0, 1000.0]))


@dataclass
class _FakeStep:
    cycle: int
    time: float
    low: list


class _FakeRunner:
    nghost = 1

    def run(self, *, max_steps: int):
        for cycle in range(1, max_steps + 1):
            yield _FakeStep(cycle=cycle, time=0.1 * cycle, low=[{"hydro_w": _fake_hydro_w()}])


def test_run_snapy_online_executes_guarded_updates(tmp_path: Path):
    config = {
        "seed": 7,
        "device": "cpu",
        "output_dir": str(tmp_path / "topora_online"),
        "num_updates": 2,
        "learning_rate": 1e-4,
        "static_tile_count": 1,
        "coarse_dx": 1000,
        "model": {"static_channels": 5, "dynamic_channels": 6, "embed_dim": 8, "depth": 1, "latent_size": 9},
        "loss": {"alpha": 1.0, "beta": 0.01, "gamma": 0.01, "weak_weight": 0.5},
    }

    out = run_snapy_online(config, runner=_FakeRunner())

    with (out / "metrics" / "before_after.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["snapy_cycle"] == "1"
    assert float(rows[0]["live_before_target_available"]) == 0.0
    assert (out / "checkpoints" / "last.pt").exists()
