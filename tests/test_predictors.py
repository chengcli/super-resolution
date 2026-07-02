from __future__ import annotations

import sys
from types import ModuleType

import torch

from super_resolution.predictors import BilinearUpscaler


def test_bilinear_upscaler_refines_selected_spatial_tensors(monkeypatch) -> None:
    restart_resize = ModuleType("paddle.restart_resize")

    def resize_spatial_tensor(tensor, *, mode, nghost):
        assert mode == "refine"
        assert nghost == 1
        resized = torch.nn.functional.interpolate(
            tensor.movedim(-1, -3).reshape(1, 1, 4, 4),
            size=(8, 8),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(1, 1, 8, 8).movedim(-3, -1)

    restart_resize.resize_spatial_tensor = resize_spatial_tensor
    paddle = ModuleType("paddle")
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    monkeypatch.setitem(sys.modules, "paddle.restart_resize", restart_resize)

    source = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4, 1)
    state = [{"hydro_w": source, "last_cycle": torch.tensor([3])}]

    result = BilinearUpscaler(nghost=1).predict(state, metadata={})

    assert result[0]["hydro_w"].shape == (1, 8, 8, 1)
    torch.testing.assert_close(result[0]["last_cycle"], torch.tensor([3]))
    assert result[0]["last_cycle"] is not state[0]["last_cycle"]


def test_bilinear_upscaler_ignores_unselected_spatial_tensor(monkeypatch) -> None:
    restart_resize = ModuleType("paddle.restart_resize")
    restart_resize.resize_spatial_tensor = lambda tensor, *, mode, nghost: tensor + 99
    monkeypatch.setitem(sys.modules, "paddle", ModuleType("paddle"))
    monkeypatch.setitem(sys.modules, "paddle.restart_resize", restart_resize)

    source = torch.ones((1, 4, 4, 1), dtype=torch.float64)

    result = BilinearUpscaler(nghost=1, variable_names=("hydro_w",)).predict(
        [{"tracer": source}], metadata={}
    )

    torch.testing.assert_close(result[0]["tracer"], source)
    assert result[0]["tracer"] is not source
