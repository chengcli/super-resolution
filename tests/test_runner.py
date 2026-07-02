from __future__ import annotations

from pathlib import Path

import torch

from super_resolution.runner import TwoResolutionRunner, state_rmse


CONFIG = """
geometry:
  type: gnomonic-equiangle
  cells: {nx1: 1, nx2: 4, nx3: 4, nghost: 1}
distribute:
  layout: cubed-sphere
  backend: gloo
dynamics:
  equation-of-state:
    type: shallow-water
"""


class _Intg:
    stages = (0, 1)

    def stop(self, cycle, time):
        return False


class _Coord:
    def __init__(self, nx2, nx3):
        self.nx2 = nx2
        self.nx3 = nx3


class _BlockOptions:
    def __init__(self):
        self.coord_value = _Coord(4, 4)
        self.out = None

    def coord(self):
        return self.coord_value

    def output_dir(self, value):
        self.out = value


class _Options:
    def __init__(self):
        self.block_options = _BlockOptions()
        self.local_cells = (4, 4)

    def block(self):
        return self.block_options

    def set_local_horizontal_cells(self, nx2, nx3):
        self.local_cells = (nx2, nx3)
        self.block_options.coord_value = _Coord(nx2, nx3)


class _MeshOptions:
    @staticmethod
    def from_yaml(path, verbose=False):
        return _Options()


class _Mesh:
    instances = []

    def __init__(self, options):
        self.options = options
        self.vars = None
        self.blocks = [object()]
        self.finalized = False
        self.cycles = []
        _Mesh.instances.append(self)

    def to(self, device):
        self.device = device

    def initialize(self, vars):
        self.vars = vars
        return vars, 0.0

    def module(self, name):
        assert name == "block0.intg"
        return _Intg()

    def set_cycle(self, cycle):
        self.cycles.append(cycle)

    def max_time_step(self, vars):
        return 0.5 if self.options.local_cells == (4, 4) else 0.25

    def forward(self, vars, dt, stage):
        vars[0]["hydro_w"] = vars[0]["hydro_w"] + dt + stage

    def check_redo(self, vars):
        return 0

    def finalize(self, vars, time):
        self.finalized = True
        self.final_time = time


class _Snapy:
    MeshOptions = _MeshOptions
    Mesh = _Mesh


class _Predictor:
    def __init__(self):
        self.metadata = None

    def predict(self, low_state, *, metadata):
        self.metadata = metadata
        return [{"hydro_w": torch.full((1, 8, 8, 1), 2.0)}]


def _initializer(mesh, config, device):
    nx2, nx3 = mesh.options.local_cells
    return [{"hydro_w": torch.zeros((1, nx3 + 2, nx2 + 2, 1), device=device)}]


def test_runner_advances_two_meshes_and_yields_observe_only_result(tmp_path: Path) -> None:
    config = tmp_path / "case.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    _Mesh.instances = []
    predictor = _Predictor()

    runner = TwoResolutionRunner(
        config,
        predictor=predictor,
        initializer=_initializer,
        use_paddle_dist=False,
        snapy_module=_Snapy,
    )

    result = next(runner.run(max_steps=1))

    assert result.cycle == 1
    assert result.time == 0.25
    assert predictor.metadata["dt"] == 0.25
    assert result.low[0]["hydro_w"].shape == (1, 6, 6, 1)
    assert result.truth[0]["hydro_w"].shape == (1, 10, 10, 1)
    torch.testing.assert_close(result.prediction[0]["hydro_w"], torch.full((1, 8, 8, 1), 2.0))
    assert all(mesh.finalized for mesh in _Mesh.instances)


def test_state_rmse_skips_shape_mismatches() -> None:
    metrics = state_rmse(
        [{"hydro_w": torch.ones((1, 2, 2, 1)), "other": torch.ones((1, 1))}],
        [{"hydro_w": torch.zeros((1, 2, 2, 1)), "other": torch.zeros((1, 2))}],
    )

    assert metrics == {"hydro_w_rmse": 1.0}
