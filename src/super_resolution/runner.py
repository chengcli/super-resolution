from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .config import (
    config_nghost,
    load_config,
    make_snapy_options,
    validate_shallow_water_config,
)
from .predictors import BilinearUpscaler
from .types import HighResPredictor, MeshState
from .w92 import initialize_w92

Initializer = Callable[[Any, dict[str, Any], torch.device], MeshState]


@dataclass(frozen=True)
class StepResult:
    cycle: int
    time: float
    low: MeshState
    prediction: MeshState
    truth: MeshState
    metrics: dict[str, float] = field(default_factory=dict)


class TwoResolutionRunner:
    """Observe-only two-resolution Snapy runner."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        predictor: HighResPredictor | None = None,
        initializer: Initializer = initialize_w92,
        low_output_dir: str | Path | None = None,
        high_output_dir: str | Path | None = None,
        make_outputs: bool = False,
        use_paddle_dist: bool = True,
        device: torch.device | str | None = None,
        snapy_module=None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        validate_shallow_water_config(self.config)
        self.nghost = config_nghost(self.config)
        self.predictor = predictor or BilinearUpscaler(self.nghost)
        self.initializer = initializer
        self.low_output_dir = Path(low_output_dir) if low_output_dir else None
        self.high_output_dir = Path(high_output_dir) if high_output_dir else None
        self.make_outputs = make_outputs
        self.use_paddle_dist = use_paddle_dist
        self.device = torch.device(device) if device is not None else None
        self.snapy_module = snapy_module

    def run(self, *, max_steps: int | None = None) -> Iterator[StepResult]:
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        snapy = self._snapy()
        device = self._start_device()
        low_vars: MeshState | None = None
        high_vars: MeshState | None = None
        low_mesh = None
        high_mesh = None
        current_time = 0.0
        try:
            low_options = make_snapy_options(
                self.config_path,
                resolution="low",
                output_dir=self.low_output_dir,
                snapy_module=snapy,
            )
            high_options = make_snapy_options(
                self.config_path,
                resolution="high",
                output_dir=self.high_output_dir,
                snapy_module=snapy,
            )
            low_mesh = snapy.Mesh(low_options)
            high_mesh = snapy.Mesh(high_options)
            low_mesh.to(device)
            high_mesh.to(device)

            low_vars, low_time = low_mesh.initialize(
                self.initializer(low_mesh, self.config, device)
            )
            high_vars, high_time = high_mesh.initialize(
                self.initializer(high_mesh, self.config, device)
            )
            current_time = _same_initial_time(low_time, high_time)

            if self.make_outputs:
                low_mesh.make_outputs(low_vars, current_time)
                high_mesh.make_outputs(high_vars, current_time)

            low_intg = low_mesh.module("block0.intg")
            high_intg = high_mesh.module("block0.intg")
            _validate_matching_stages(low_intg, high_intg)

            cycle = 0
            accepted = 0
            while max_steps is None or accepted < max_steps:
                if low_intg.stop(cycle, current_time) or high_intg.stop(
                    cycle, current_time
                ):
                    break

                cycle += 1
                low_mesh.set_cycle(cycle)
                high_mesh.set_cycle(cycle)
                dt = min(
                    float(low_mesh.max_time_step(low_vars)),
                    float(high_mesh.max_time_step(high_vars)),
                )

                for stage in range(len(low_intg.stages)):
                    low_mesh.forward(low_vars, dt, stage)
                    high_mesh.forward(high_vars, dt, stage)

                low_err = int(low_mesh.check_redo(low_vars))
                high_err = int(high_mesh.check_redo(high_vars))
                if low_err < 0 or high_err < 0:
                    break
                if low_err > 0 or high_err > 0:
                    continue

                current_time += dt
                prediction = self.predictor.predict(
                    low_vars,
                    metadata={
                        "cycle": cycle,
                        "time": current_time,
                        "dt": dt,
                        "nghost": self.nghost,
                        "config_path": str(self.config_path),
                    },
                )
                if self.make_outputs:
                    low_mesh.make_outputs(low_vars, current_time)
                    high_mesh.make_outputs(high_vars, current_time)

                accepted += 1
                yield StepResult(
                    cycle=cycle,
                    time=current_time,
                    low=low_vars,
                    prediction=prediction,
                    truth=high_vars,
                    metrics=state_rmse(prediction, high_vars),
                )
        finally:
            if low_mesh is not None and low_vars is not None:
                low_mesh.finalize(low_vars, current_time)
            if high_mesh is not None and high_vars is not None:
                high_mesh.finalize(high_vars, current_time)
            self._close_dist()

    def _snapy(self):
        if self.snapy_module is not None:
            return self.snapy_module
        import snapy

        return snapy

    def _start_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        if not self.use_paddle_dist:
            return torch.device("cpu")
        from paddle import start_dist

        backend = self.config.get("distribute", {}).get("backend", "gloo")
        return start_dist(backend)

    def _close_dist(self) -> None:
        if self.device is not None or not self.use_paddle_dist:
            return
        from paddle import close_dist

        close_dist()


def state_rmse(prediction: MeshState, truth: MeshState) -> dict[str, float]:
    totals: dict[str, tuple[float, int]] = {}
    for pred_block, truth_block in zip(prediction, truth, strict=False):
        for name, pred_tensor in pred_block.items():
            truth_tensor = truth_block.get(name)
            if truth_tensor is None or pred_tensor.shape != truth_tensor.shape:
                continue
            if not pred_tensor.is_floating_point() or not truth_tensor.is_floating_point():
                continue
            diff = (pred_tensor.detach() - truth_tensor.detach()).to(torch.float64)
            total, count = totals.get(name, (0.0, 0))
            totals[name] = (total + float(torch.sum(diff * diff).cpu()), count + diff.numel())
    return {
        f"{name}_rmse": (total / count) ** 0.5
        for name, (total, count) in totals.items()
        if count > 0
    }


def _same_initial_time(low_time: float, high_time: float) -> float:
    if abs(float(low_time) - float(high_time)) > 1.0e-12:
        raise ValueError(
            f"low and high models initialized at different times: {low_time} != {high_time}"
        )
    return float(low_time)


def _validate_matching_stages(low_intg, high_intg) -> None:
    if len(low_intg.stages) != len(high_intg.stages):
        raise ValueError(
            "low and high Snapy integrators must use the same number of stages"
        )
