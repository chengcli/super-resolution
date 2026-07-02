"""Online-train TopoRA as a sidecar beside a live snapy two-resolution run.

Each accepted snapy step yields the low-resolution coarse state; the sidecar
converts it into a TopoRA live sample (snapy wind forcing over a local 30 m
terrain tile) and applies one guarded online update, gated on live-only
coarse-consistency/seam/NaN/speed checks (see ``_candidate_accepted``).

Uses the vendored ``topo_ra`` package that ships with this repo.
"""

from __future__ import annotations

import argparse
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import data_parallel
from torch.nn import functional as F

from topo_ra.data.vertical_interp import heights_in_range
from topo_ra.data.snapy_reader import HYDRO_W_ORDER
from topo_ra.data.synthetic_dataset import make_synthetic_sample
from topo_ra.train.online_train import (
    LIVE_METRICS,
    _print_before_after,
    _sample_to_batch,
    _set_trainable_parameters,
    _z_out_from_config,
    guarded_online_update,
)
from topo_ra.utils.checkpoint import load_checkpoint, load_model_state, model_from_config
from topo_ra.utils.config import load_config
from topo_ra.utils.device import resolve_device
from topo_ra.utils.io import append_csv_row, ensure_dir
from topo_ra.utils.seeding import seed_everything

from .config import (
    config_nghost,
    load_config as load_snapy_config,
    make_snapy_options,
    validate_shallow_water_config,
)
from .types import MeshState
from .w92 import initialize_w92


class _PredictAtHeightsAdapter(torch.nn.Module):
    """Expose ``predict_at_heights`` as ``forward`` for DataParallel scatter."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        static_30m: torch.Tensor,
        dynamic_native: torch.Tensor,
        canonical_uv_100m: torch.Tensor,
        coarse_dx: torch.Tensor,
        z_out_values: tuple[float, ...],
        *,
        coarse_profile: torch.Tensor | None = None,
        coarse_profile_heights_values: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        z_out = torch.tensor(z_out_values, device=static_30m.device, dtype=static_30m.dtype)
        coarse_profile_heights = (
            None
            if coarse_profile_heights_values is None
            else torch.tensor(coarse_profile_heights_values, device=static_30m.device, dtype=static_30m.dtype)
        )
        pred, _valid_mask = self.module.predict_at_heights(
            static_30m,
            dynamic_native,
            canonical_uv_100m,
            coarse_dx,
            z_out,
            coarse_profile=coarse_profile,
            coarse_profile_heights=coarse_profile_heights,
        )
        return pred


class _TopoRADataParallel(torch.nn.DataParallel):
    """DataParallel wrapper that preserves TopoRA's height-query API."""

    def predict_at_heights(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) < 5:
            raise TypeError("predict_at_heights requires static, dynamic, canonical, coarse_dx, and z_out")
        static_30m, dynamic_native, canonical_uv_100m, coarse_dx, z_out, *rest = args
        if rest:
            raise TypeError("unexpected positional arguments after z_out")
        if kwargs.get("return_diagnostics", False):
            raise ValueError("DataParallel predict_at_heights does not support return_diagnostics=True")
        z_out_tensor = torch.as_tensor(z_out, dtype=static_30m.dtype)
        if z_out_tensor.ndim != 1:
            raise ValueError("z_out must be a 1D tensor of AGL heights")
        coarse_profile_heights = kwargs.pop("coarse_profile_heights", None)
        coarse_profile_heights_values = None
        if coarse_profile_heights is not None:
            heights_tensor = torch.as_tensor(coarse_profile_heights, dtype=static_30m.dtype)
            if heights_tensor.ndim != 1:
                raise ValueError("coarse_profile_heights must be a 1D tensor")
            coarse_profile_heights_values = tuple(float(value) for value in heights_tensor.detach().cpu())
        adapter = _PredictAtHeightsAdapter(self.module)
        pred = data_parallel(
            adapter,
            (
                static_30m,
                dynamic_native,
                canonical_uv_100m,
                coarse_dx,
                tuple(float(value) for value in z_out_tensor.detach().cpu()),
            ),
            module_kwargs={
                "coarse_profile": kwargs.pop("coarse_profile", None),
                "coarse_profile_heights_values": coarse_profile_heights_values,
            },
            device_ids=self.device_ids,
            output_device=getattr(self, "output_device", self.device_ids[0] if self.device_ids else None),
        )
        if kwargs:
            raise TypeError(f"unexpected predict_at_heights keyword arguments: {sorted(kwargs)}")
        source_heights = self.module.output_heights.to(device=pred.device, dtype=pred.dtype)
        valid_mask = heights_in_range(z_out_tensor.to(device=pred.device, dtype=pred.dtype), source_heights)
        return pred, valid_mask


def _configured_cuda_ids(config: dict[str, Any], device: torch.device) -> list[int]:
    parallel_cfg = config.get("parallel", {})
    requested = parallel_cfg.get("device_ids", config.get("device_ids"))
    if requested is None:
        return []
    if device.type != "cuda":
        raise ValueError("parallel.device_ids requires a CUDA training device")
    ids = [int(item) for item in requested]
    if len(ids) < 2:
        return []
    available = torch.cuda.device_count()
    missing = [idx for idx in ids if idx < 0 or idx >= available]
    if missing:
        raise ValueError(f"Requested CUDA device ids {missing} but only {available} CUDA devices are visible")
    return ids


def _repeat_batch(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    if count <= 1:
        return batch
    repeated: dict[str, torch.Tensor] = {}
    shared_vertical_keys = {"z_out", "valid_mask", "coarse_profile_heights"}
    for key, value in batch.items():
        if key in shared_vertical_keys:
            repeated[key] = value
        elif value.shape[:1] == (1,):
            repeated[key] = value.repeat((count, *([1] * (value.ndim - 1))))
        else:
            repeated[key] = value
    return repeated


def _model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    wrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
    return wrapped.state_dict()


@dataclass(frozen=True)
class _LiveSnapyStep:
    cycle: int
    time: float
    low: MeshState
    truth: MeshState | None = None


def _state_to_wire(state: MeshState) -> list[dict[str, Any]]:
    return [{name: tensor.detach().cpu().numpy().copy() for name, tensor in block.items()} for block in state]


def _state_from_wire(state: list[dict[str, Any]]) -> MeshState:
    return [{name: torch.from_numpy(value.copy()) for name, value in block.items()} for block in state]


class _LowResolutionSnapyRunner:
    """Run only the live low-resolution Snapy mesh used by online training."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        device: torch.device | str | None = None,
        use_paddle_dist: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_snapy_config(self.config_path)
        validate_shallow_water_config(self.config)
        self.nghost = config_nghost(self.config)
        self.device = torch.device(device) if device is not None else None
        self.use_paddle_dist = bool(use_paddle_dist)

    def run(self, *, max_steps: int | None = None):
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        import snapy

        device = self._start_device()
        mesh = None
        vars_: MeshState | None = None
        current_time = 0.0
        try:
            options = make_snapy_options(self.config_path, resolution="low", snapy_module=snapy)
            mesh = snapy.Mesh(options)
            mesh.to(device)
            vars_, current_time = mesh.initialize(initialize_w92(mesh, self.config, device))
            intg = mesh.module("block0.intg")

            cycle = 0
            accepted = 0
            while max_steps is None or accepted < max_steps:
                if intg.stop(cycle, current_time):
                    break
                cycle += 1
                mesh.set_cycle(cycle)
                dt = float(mesh.max_time_step(vars_))
                for stage in range(len(intg.stages)):
                    mesh.forward(vars_, dt, stage)

                err = int(mesh.check_redo(vars_))
                if err < 0:
                    break
                if err > 0:
                    continue

                current_time += dt
                accepted += 1
                yield _LiveSnapyStep(cycle=cycle, time=current_time, low=vars_)
        finally:
            if mesh is not None and vars_ is not None:
                mesh.finalize(vars_, current_time)
            self._close_dist()

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


def _snapy_resolution_worker(
    *,
    resolution: str,
    config_path: str,
    device_name: str | None,
    use_paddle_dist: bool,
    command_queue: Any,
    result_queue: Any,
) -> None:
    mesh = None
    vars_: MeshState | None = None
    current_time = 0.0
    dist_started = False
    try:
        import snapy

        config = load_snapy_config(config_path)
        validate_shallow_water_config(config)
        device = torch.device(device_name) if device_name is not None else None
        if device is None:
            if use_paddle_dist:
                from paddle import start_dist

                backend = config.get("distribute", {}).get("backend", "gloo")
                device = start_dist(backend)
                dist_started = True
            else:
                device = torch.device("cpu")

        options = make_snapy_options(config_path, resolution=resolution, snapy_module=snapy)
        mesh = snapy.Mesh(options)
        mesh.to(device)
        vars_, current_time = mesh.initialize(initialize_w92(mesh, config, device))
        intg = mesh.module("block0.intg")
        result_queue.put(
            {
                "type": "ready",
                "resolution": resolution,
                "time": float(current_time),
                "stages": len(intg.stages),
                "nghost": config_nghost(config),
            }
        )

        while True:
            command = command_queue.get()
            command_type = command.get("type")
            if command_type == "shutdown":
                break
            if command_type == "max_dt":
                cycle = int(command["cycle"])
                stopped = bool(intg.stop(cycle, current_time))
                dt = None if stopped else float(mesh.max_time_step(vars_))
                result_queue.put({"type": "max_dt", "resolution": resolution, "stopped": stopped, "dt": dt})
                continue
            if command_type == "advance":
                cycle = int(command["cycle"])
                dt = float(command["dt"])
                mesh.set_cycle(cycle)
                for stage in range(len(intg.stages)):
                    mesh.forward(vars_, dt, stage)
                err = int(mesh.check_redo(vars_))
                payload: dict[str, Any] = {
                    "type": "advanced",
                    "resolution": resolution,
                    "cycle": cycle,
                    "err": err,
                    "time": float(current_time),
                    "state": None,
                }
                if err == 0:
                    current_time += dt
                    payload["time"] = float(current_time)
                    payload["state"] = _state_to_wire(vars_)
                result_queue.put(payload)
                continue
            raise ValueError(f"Unknown worker command: {command_type!r}")
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "resolution": resolution,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if mesh is not None and vars_ is not None:
            try:
                mesh.finalize(vars_, current_time)
            except BaseException:
                pass
        if dist_started:
            try:
                from paddle import close_dist

                close_dist()
            except BaseException:
                pass


class _ProcessIsolatedTwoResolutionSnapyRunner:
    """Run low and high Snapy meshes in separate worker processes."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        low_device: str | None = None,
        high_device: str | None = None,
        use_paddle_dist: bool = False,
        start_method: str = "spawn",
        queue_timeout_s: float = 120.0,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_snapy_config(self.config_path)
        validate_shallow_water_config(self.config)
        self.nghost = config_nghost(self.config)
        self.low_device = low_device
        self.high_device = high_device
        self.use_paddle_dist = bool(use_paddle_dist)
        self.start_method = start_method
        self.queue_timeout_s = float(queue_timeout_s)

    def run(self, *, max_steps: int | None = None):
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        ctx = mp.get_context(self.start_method)
        low_commands = ctx.Queue()
        high_commands = ctx.Queue()
        low_results = ctx.Queue()
        high_results = ctx.Queue()
        low_process = ctx.Process(
            target=_snapy_resolution_worker,
            kwargs={
                "resolution": "low",
                "config_path": str(self.config_path),
                "device_name": self.low_device,
                "use_paddle_dist": self.use_paddle_dist,
                "command_queue": low_commands,
                "result_queue": low_results,
            },
        )
        high_process = ctx.Process(
            target=_snapy_resolution_worker,
            kwargs={
                "resolution": "high",
                "config_path": str(self.config_path),
                "device_name": self.high_device,
                "use_paddle_dist": self.use_paddle_dist,
                "command_queue": high_commands,
                "result_queue": high_results,
            },
        )
        processes = (low_process, high_process)
        command_queues = (low_commands, high_commands)
        try:
            low_process.start()
            high_process.start()
            low_ready = self._get_result(low_results, "low")
            high_ready = self._get_result(high_results, "high")
            self._raise_if_error(low_ready)
            self._raise_if_error(high_ready)
            if low_ready["type"] != "ready" or high_ready["type"] != "ready":
                raise RuntimeError(f"Unexpected worker startup payloads: {low_ready}, {high_ready}")
            self._validate_ready(low_ready, high_ready)

            cycle = 0
            accepted = 0
            while max_steps is None or accepted < max_steps:
                low_commands.put({"type": "max_dt", "cycle": cycle})
                high_commands.put({"type": "max_dt", "cycle": cycle})
                low_dt = self._get_result(low_results, "low")
                high_dt = self._get_result(high_results, "high")
                self._raise_if_error(low_dt)
                self._raise_if_error(high_dt)
                if bool(low_dt["stopped"]) or bool(high_dt["stopped"]):
                    break

                cycle += 1
                dt = min(float(low_dt["dt"]), float(high_dt["dt"]))
                low_commands.put({"type": "advance", "cycle": cycle, "dt": dt})
                high_commands.put({"type": "advance", "cycle": cycle, "dt": dt})
                low_step = self._get_result(low_results, "low")
                high_step = self._get_result(high_results, "high")
                self._raise_if_error(low_step)
                self._raise_if_error(high_step)
                self._validate_step(low_step, high_step, cycle)

                low_err = int(low_step["err"])
                high_err = int(high_step["err"])
                if low_err < 0 or high_err < 0:
                    break
                if low_err > 0 or high_err > 0:
                    continue

                accepted += 1
                yield _LiveSnapyStep(
                    cycle=cycle,
                    time=float(low_step["time"]),
                    low=_state_from_wire(low_step["state"]),
                    truth=_state_from_wire(high_step["state"]),
                )
        finally:
            for command_queue in command_queues:
                try:
                    command_queue.put({"type": "shutdown"})
                except BaseException:
                    pass
            for process in processes:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)

    def _get_result(self, result_queue: Any, resolution: str) -> dict[str, Any]:
        try:
            result = result_queue.get(timeout=self.queue_timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out waiting for {resolution} Snapy worker") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected {resolution} worker payload: {result!r}")
        return result

    @staticmethod
    def _raise_if_error(result: dict[str, Any]) -> None:
        if result.get("type") == "error":
            raise RuntimeError(f"{result.get('resolution')} Snapy worker failed: {result.get('error')}\n{result.get('traceback')}")

    @staticmethod
    def _validate_ready(low_ready: dict[str, Any], high_ready: dict[str, Any]) -> None:
        if abs(float(low_ready["time"]) - float(high_ready["time"])) > 1.0e-12:
            raise RuntimeError(f"Low/high workers initialized at different times: {low_ready['time']} != {high_ready['time']}")
        if int(low_ready["stages"]) != int(high_ready["stages"]):
            raise RuntimeError(f"Low/high workers have different integrator stage counts: {low_ready['stages']} != {high_ready['stages']}")
        if int(low_ready["nghost"]) != int(high_ready["nghost"]):
            raise RuntimeError(f"Low/high workers have different ghost-cell counts: {low_ready['nghost']} != {high_ready['nghost']}")

    @staticmethod
    def _validate_step(low_step: dict[str, Any], high_step: dict[str, Any], cycle: int) -> None:
        if low_step["type"] != "advanced" or high_step["type"] != "advanced":
            raise RuntimeError(f"Unexpected worker step payloads: {low_step}, {high_step}")
        if int(low_step["cycle"]) != cycle or int(high_step["cycle"]) != cycle:
            raise RuntimeError(f"Low/high worker cycle mismatch at coordinator cycle {cycle}: {low_step['cycle']} / {high_step['cycle']}")
        if int(low_step["err"]) == 0 and int(high_step["err"]) == 0:
            if abs(float(low_step["time"]) - float(high_step["time"])) > 1.0e-10:
                raise RuntimeError(f"Low/high accepted times diverged: {low_step['time']} != {high_step['time']}")


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
    coarse-consistency loss.
    """
    snapy_cfg = config.get("snapy", {})
    block_index = int(snapy_cfg.get("block_index", 0))
    velocity_scale = float(snapy_cfg.get("velocity_scale", 1.0))
    coarse_cells = int(snapy_cfg.get("coarse_cells", 9))
    candidate_indices = [block_index, *(idx for idx in range(len(low_state)) if idx != block_index)]
    uvw = None
    selected_block_index = block_index
    nonfinite_counts: dict[int, int] = {}
    candidates: dict[int, torch.Tensor] = {}
    for idx in candidate_indices:
        hydro_w = low_state[idx]["hydro_w"]
        if hydro_w.ndim == 5:
            hydro_w = hydro_w[-1]
        candidate = block_uvw(hydro_w.detach().cpu(), nghost) * velocity_scale
        nonfinite_count = int((~torch.isfinite(candidate)).sum().item())
        candidates[idx] = candidate
        if nonfinite_count == 0:
            uvw = candidate
            selected_block_index = idx
            break
        nonfinite_counts[idx] = nonfinite_count
    if uvw is None:
        if not bool(snapy_cfg.get("sanitize_nonfinite", False)):
            raise ValueError(f"snapy low-resolution state contains non-finite u/v/w values: {nonfinite_counts}")
        selected_block_index = min(nonfinite_counts, key=nonfinite_counts.get)
        uvw = torch.nan_to_num(candidates[selected_block_index], nan=0.0, posinf=0.0, neginf=0.0)
        print(
            "snapy blocks were non-finite; "
            f"sanitized block {selected_block_index} with {nonfinite_counts[selected_block_index]} bad values"
        )
    if selected_block_index != block_index:
        print(f"snapy block {block_index} was non-finite; using finite block {selected_block_index}")
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


def _build_runner(config: dict[str, Any]) -> _LowResolutionSnapyRunner | _ProcessIsolatedTwoResolutionSnapyRunner:
    # Import snapy eagerly (instead of inside runner.run) so the float32
    # default-dtype reset in run_snapy_online happens after snapy's float64 switch.
    import snapy  # noqa: F401

    snapy_cfg = config.get("snapy", {})
    case_path = snapy_cfg.get("config")
    if not case_path:
        raise ValueError("snapy.config must point to a snapy shallow-water YAML case file")
    mode = str(snapy_cfg.get("mode", "two_process"))
    if mode == "low_only":
        return _LowResolutionSnapyRunner(
            case_path,
            use_paddle_dist=bool(snapy_cfg.get("use_paddle_dist", False)),
            device=snapy_cfg.get("device", "cpu"),
        )
    if mode != "two_process":
        raise ValueError("snapy.mode must be 'two_process' or 'low_only'")
    default_device = snapy_cfg.get("device", "cpu")
    return _ProcessIsolatedTwoResolutionSnapyRunner(
        case_path,
        use_paddle_dist=bool(snapy_cfg.get("use_paddle_dist", False)),
        low_device=snapy_cfg.get("low_device", default_device),
        high_device=snapy_cfg.get("high_device", default_device),
        start_method=str(snapy_cfg.get("start_method", "spawn")),
        queue_timeout_s=float(snapy_cfg.get("queue_timeout_s", 120.0)),
    )


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

    model = model_from_config(config).to(device)
    init_from = config.get("init_from")
    if init_from:
        checkpoint = load_checkpoint(init_from, device)
        load_model_state(model, checkpoint["model"])
    trainable_parameters = _set_trainable_parameters(model, config)
    if not trainable_parameters:
        raise ValueError("No trainable parameters selected for online training")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(config.get("learning_rate", 2e-4)))
    cuda_ids = _configured_cuda_ids(config, device)
    if cuda_ids:
        model = _TopoRADataParallel(model, device_ids=cuda_ids, output_device=cuda_ids[0])
        print(f"using CUDA DataParallel on devices: {cuda_ids}")
    z_out = _z_out_from_config(config)

    # Procedural terrain tiles cycled across updates; no FuXi case data needed.
    model_cfg = config.get("model", {})
    static_tile_count = max(1, int(config.get("static_tile_count", 2)))
    static_tiles = [
        make_synthetic_sample(
            idx,
            static_channels=int(model_cfg.get("static_channels", 5)),
            dynamic_channels=int(model_cfg.get("dynamic_channels", 6)),
        )["static_30m"]
        for idx in range(static_tile_count)
    ]

    num_updates = int(config.get("num_updates", 4))
    parallel_cfg = config.get("parallel", {})
    live_repeat = int(parallel_cfg.get("live_repeat", max(1, len(cuda_ids))))
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
        live = _repeat_batch(live, live_repeat)
        result = guarded_online_update(model, optimizer, live, config)
        if result["accepted"]:
            accepted_count += 1
            checkpoint_path = checkpoint_dir / f"online_update_{update_idx:04d}.pt"
            torch.save({"model": _model_state_dict(model), "config": config, "update": update_idx}, checkpoint_path)
        else:
            rejected_count += 1
            checkpoint_path = checkpoint_dir / "rejected_no_checkpoint.pt"

        print(f"[snapy online update {update_idx:04d} | cycle={step.cycle} | t={step.time:.6g}]")
        print()
        _print_before_after("LIVE SNAPY METRICS", LIVE_METRICS, result["before_live"], result["after_live"])
        print()
        print(f"decision: {'ACCEPT' if result['accepted'] else 'REJECT'}")
        print(f"checkpoint: {checkpoint_path}")

        row: dict[str, Any] = {
            "update": update_idx,
            "snapy_cycle": step.cycle,
            "snapy_time": step.time,
            "high_truth_available": int(getattr(step, "truth", None) is not None),
            "accepted": int(result["accepted"]),
            "accepted_updates": accepted_count,
            "rejected_updates": rejected_count,
        }
        for key in LIVE_METRICS:
            row[f"live_before_{key}"] = result["before_live"][key]
            row[f"live_after_{key}"] = result["after_live"][key]
        append_csv_row(metrics_dir / "before_after.csv", row)

    if update_idx == 0:
        raise RuntimeError("snapy runner produced no accepted steps; nothing to train on")
    torch.save({"model": _model_state_dict(model), "config": config}, checkpoint_dir / "last.pt")
    print(f"accepted {accepted_count} / rejected {rejected_count} updates")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TopoRA online training beside a live snapy case.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_snapy_online(load_config(args.config))


if __name__ == "__main__":
    main()
