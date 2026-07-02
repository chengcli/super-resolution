from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

import super_resolution.topora_online as topora_online
from super_resolution.topora_online import (
    _LowResolutionSnapyRunner,
    _ProcessIsolatedTwoResolutionSnapyRunner,
    _TopoRADataParallel,
    block_uvw,
    run_snapy_online,
    snapy_state_to_sample,
)


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


def test_low_resolution_snapy_runner_only_builds_low_mesh(tmp_path: Path, monkeypatch):
    config = tmp_path / "case.yaml"
    config.write_text(
        """
geometry:
  type: gnomonic-equiangle
  cells: {nx1: 1, nx2: 4, nx3: 4, nghost: 1}
distribute:
  layout: cubed-sphere
dynamics:
  equation-of-state:
    type: shallow-water
""",
        encoding="utf-8",
    )
    requested_resolutions = []

    def _make_options(path, *, resolution, snapy_module):
        requested_resolutions.append(resolution)
        return {"resolution": resolution}

    class _Intg:
        stages = (0,)

        def stop(self, cycle, time):
            return False

    class _Mesh:
        def __init__(self, options):
            assert options["resolution"] == "low"
            self.finalized = False

        def to(self, device):
            self.device = device

        def initialize(self, vars_):
            return vars_, 0.0

        def module(self, name):
            assert name == "block0.intg"
            return _Intg()

        def set_cycle(self, cycle):
            self.cycle = cycle

        def max_time_step(self, vars_):
            return 0.5

        def forward(self, vars_, dt, stage):
            vars_[0]["hydro_w"] = vars_[0]["hydro_w"] + 1.0

        def check_redo(self, vars_):
            return 0

        def finalize(self, vars_, time):
            self.finalized = True

    monkeypatch.setattr(topora_online, "make_snapy_options", _make_options)
    monkeypatch.setattr(topora_online, "initialize_w92", lambda mesh, cfg, device: [{"hydro_w": torch.zeros(1)}])
    fake_snapy = ModuleType("snapy")
    fake_snapy.Mesh = _Mesh
    monkeypatch.setitem(sys.modules, "snapy", fake_snapy)

    runner = _LowResolutionSnapyRunner(config, device="cpu")
    steps = list(runner.run(max_steps=2))

    assert requested_resolutions == ["low"]
    assert [step.cycle for step in steps] == [1, 2]
    assert [step.time for step in steps] == [0.5, 1.0]


def test_build_runner_defaults_to_process_isolated_two_resolution():
    config = {
        "snapy": {
            "config": "configs/snapy_w92_tiny.yaml",
            "device": "cpu",
        }
    }

    runner = topora_online._build_runner(config)

    assert isinstance(runner, _ProcessIsolatedTwoResolutionSnapyRunner)
    assert runner.low_device == "cpu"
    assert runner.high_device == "cpu"


def test_build_runner_uses_separate_snapy_worker_devices():
    config = {
        "snapy": {
            "config": "configs/snapy_w92_tiny.yaml",
            "low_device": "cuda:0",
            "high_device": "cuda:1",
        }
    }

    runner = topora_online._build_runner(config)

    assert isinstance(runner, _ProcessIsolatedTwoResolutionSnapyRunner)
    assert runner.low_device == "cuda:0"
    assert runner.high_device == "cuda:1"


def test_process_isolated_runner_rejects_diverged_times():
    low_step = {"type": "advanced", "cycle": 3, "err": 0, "time": 1.0}
    high_step = {"type": "advanced", "cycle": 3, "err": 0, "time": 1.1}

    try:
        _ProcessIsolatedTwoResolutionSnapyRunner._validate_step(low_step, high_step, 3)
    except RuntimeError as exc:
        assert "accepted times diverged" in str(exc)
    else:
        raise AssertionError("expected diverged worker times to fail validation")


def test_dataparallel_predict_at_heights_uses_parallel_dispatch(monkeypatch):
    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("output_heights", torch.tensor([5.0, 10.0, 20.0]))

        def forward(self):
            raise AssertionError("forward should be reached through the adapter")

        def predict_at_heights(self, *args, **kwargs):
            raise AssertionError("wrapper must not call the base module directly")

    calls = []

    def _fake_data_parallel(module, inputs, *, module_kwargs, device_ids, output_device, dim=0):
        calls.append(
            {
                "module": module,
                "inputs": inputs,
                "module_kwargs": module_kwargs,
                "device_ids": device_ids,
                "output_device": output_device,
                "dim": dim,
            }
        )
        static_30m = inputs[0]
        z_out_values = inputs[4]
        return torch.zeros(static_30m.shape[0], 3, len(z_out_values), 2, 2)

    monkeypatch.setattr(topora_online, "data_parallel", _fake_data_parallel)
    model = _TopoRADataParallel(_Module())
    model.device_ids = [0, 1]
    model.output_device = 0

    pred, valid_mask = model.predict_at_heights(
        torch.zeros(2, 5, 4, 4),
        torch.zeros(2, 6, 2, 2),
        torch.zeros(2, 2, 2, 2),
        torch.ones(2),
        torch.tensor([5.0, 15.0]),
        coarse_profile_heights=torch.tensor([0.0, 100.0]),
    )

    assert len(calls) == 1
    assert isinstance(calls[0]["module"], topora_online._PredictAtHeightsAdapter)
    assert calls[0]["inputs"][4] == (5.0, 15.0)
    assert calls[0]["module_kwargs"]["coarse_profile_heights_values"] == (0.0, 100.0)
    assert calls[0]["device_ids"] == [0, 1]
    assert pred.shape == (2, 3, 2, 2, 2)
    torch.testing.assert_close(valid_mask, torch.tensor([True, True]))
