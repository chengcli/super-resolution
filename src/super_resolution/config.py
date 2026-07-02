from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml


Resolution = Literal["low", "high"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Snapy config must be a YAML mapping")
    return data


def validate_shallow_water_config(config: dict[str, Any]) -> None:
    geometry = _mapping(config, "geometry")
    if geometry.get("type") != "gnomonic-equiangle":
        raise ValueError("v1 supports Snapy gnomonic-equiangle cubed-sphere configs")

    cells = _mapping(geometry, "cells")
    if int(cells.get("nx1", 0)) != 1:
        raise ValueError("v1 shallow-water downscaling expects nx1 == 1")
    for key in ("nx2", "nx3", "nghost"):
        if int(cells.get(key, 0)) <= 0:
            raise ValueError(f"geometry.cells.{key} must be positive")

    distribute = _mapping(config, "distribute")
    if distribute.get("layout") != "cubed-sphere":
        raise ValueError("v1 supports distribute.layout: cubed-sphere")

    dynamics = _mapping(config, "dynamics")
    eos = _mapping(dynamics, "equation-of-state")
    if eos.get("type") != "shallow-water":
        raise ValueError("v1 supports dynamics.equation-of-state.type: shallow-water")


def doubled_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    cells = result["geometry"]["cells"]
    cells["nx2"] = 2 * int(cells["nx2"])
    cells["nx3"] = 2 * int(cells["nx3"])
    return result


def make_snapy_options(
    config_path: str | Path,
    *,
    resolution: Resolution,
    output_dir: str | Path | None = None,
    snapy_module=None,
):
    snapy = _snapy(snapy_module)
    options = _mesh_options_from_yaml(snapy, config_path)
    if resolution == "high":
        coord = options.block().coord()
        options.set_local_horizontal_cells(
            2 * int(_option_value(coord.nx2)),
            2 * int(_option_value(coord.nx3)),
        )
    if output_dir is not None:
        options.block().output_dir(str(output_dir))
    return options


def config_nghost(config: dict[str, Any]) -> int:
    return int(config["geometry"]["cells"]["nghost"])


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Snapy config section {key!r} must be a mapping")
    return value


def _snapy(snapy_module=None):
    if snapy_module is not None:
        return snapy_module
    import snapy

    return snapy


def _mesh_options_from_yaml(snapy, config_path: str | Path):
    try:
        return snapy.MeshOptions.from_yaml(str(config_path), verbose=False)
    except TypeError:
        return snapy.MeshOptions.from_yaml(str(config_path))


def _option_value(value):
    return value() if callable(value) else value
