from __future__ import annotations

from pathlib import Path

import pytest

from super_resolution.config import make_snapy_options, validate_shallow_water_config


def _valid_config() -> dict:
    return {
        "geometry": {
            "type": "gnomonic-equiangle",
            "cells": {"nx1": 1, "nx2": 12, "nx3": 10, "nghost": 3},
        },
        "distribute": {"layout": "cubed-sphere"},
        "dynamics": {"equation-of-state": {"type": "shallow-water"}},
    }


def test_validate_shallow_water_config_accepts_v1_shape() -> None:
    validate_shallow_water_config(_valid_config())


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda cfg: cfg["geometry"].update(type="cartesian"), "gnomonic"),
        (lambda cfg: cfg["geometry"]["cells"].update(nx1=2), "nx1"),
        (lambda cfg: cfg["distribute"].update(layout="slab"), "cubed-sphere"),
        (
            lambda cfg: cfg["dynamics"]["equation-of-state"].update(type="ideal-gas"),
            "shallow-water",
        ),
    ],
)
def test_validate_shallow_water_config_rejects_unsupported_cases(change, message) -> None:
    config = _valid_config()
    change(config)

    with pytest.raises(ValueError, match=message):
        validate_shallow_water_config(config)


class _Value:
    def __init__(self, value):
        self.value = value

    def __call__(self, value=None):
        if value is None:
            return self.value
        self.value = value
        return self


class _Coord:
    nx2 = _Value(8)
    nx3 = _Value(6)


class _BlockOptions:
    def __init__(self):
        self.out = None

    def coord(self):
        return _Coord()

    def output_dir(self, value):
        self.out = value


class _Options:
    def __init__(self):
        self.block_options = _BlockOptions()
        self.local_cells = None

    def block(self):
        return self.block_options

    def set_local_horizontal_cells(self, nx2, nx3):
        self.local_cells = (nx2, nx3)


class _MeshOptions:
    last = None

    @staticmethod
    def from_yaml(path, verbose=False):
        _MeshOptions.last = _Options()
        return _MeshOptions.last


class _Snapy:
    MeshOptions = _MeshOptions


def test_make_snapy_options_doubles_high_resolution_local_cells(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text("geometry: {}\n", encoding="utf-8")

    options = make_snapy_options(path, resolution="high", output_dir="high", snapy_module=_Snapy)

    assert options.local_cells == (16, 12)
    assert options.block_options.out == "high"
