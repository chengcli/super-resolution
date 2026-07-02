from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from topo_ra.data.aloft_extension import extend_wind_from_top_level
from topo_ra.data.tile_builder import build_model_input
from topo_ra.data.vertical_interp import FU_XI_AGL_LEVELS, interpolate_to_agl


def _smooth_field(field: torch.Tensor, passes: int = 2) -> torch.Tensor:
    out = field.unsqueeze(0).unsqueeze(0)
    for _ in range(passes):
        out = F.avg_pool2d(F.pad(out, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
    return out[0, 0]


def make_synthetic_sample(
    idx: int = 0,
    dx: int = 300,
    static_channels: int = 5,
    dynamic_channels: int = 6,
    z_out: torch.Tensor | Sequence[float] | None = None,
    target_mode: str = "fuxi_interp",
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(10_000 + idx * 97 + dx)
    y = torch.linspace(-1.0, 1.0, 300)
    x = torch.linspace(-1.0, 1.0, 300)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = 0.07 * idx
    hill = torch.exp(-4.0 * ((xx - 0.25 * math.sin(phase)) ** 2 + (yy + 0.2 * math.cos(phase)) ** 2))
    ridge = 0.35 * torch.sin(2.0 * math.pi * (xx + 0.2 * yy + phase))
    noise = 0.04 * _smooth_field(torch.randn(300, 300, generator=generator), passes=3)
    dem = 800.0 * (hill + ridge + noise)
    roughness = 0.05 + 0.25 * torch.sigmoid(2.0 * hill + 0.5 * ridge)
    slope_y, slope_x = torch.gradient(dem, spacing=(30.0, 30.0))
    curvature = _smooth_field(slope_x, passes=1) + _smooth_field(slope_y, passes=1)
    static_fields = [dem / 1000.0, roughness, slope_x * 20.0, slope_y * 20.0, curvature * 10.0]
    while len(static_fields) < static_channels:
        static_fields.append(torch.zeros_like(dem))
    static_30m = torch.stack(static_fields[:static_channels])

    base_u = 8.0 + 1.5 * torch.sin(math.pi * yy + phase) + 0.8 * torch.cos(2.0 * math.pi * xx)
    base_v = 3.0 + 1.2 * torch.cos(math.pi * xx - phase) + 0.4 * torch.sin(2.0 * math.pi * yy)
    base_w = 0.05 * torch.sin(2.0 * math.pi * (xx + yy + phase))
    rho = 1.2 - 0.08 * static_30m[0]
    p = 90000.0 - 20.0 * dem
    q = 0.004 + 0.001 * torch.sigmoid(-yy)
    fine_dynamic = [base_u, base_v, base_w, rho, p / 100000.0, q * 1000.0]
    while len(fine_dynamic) < dynamic_channels:
        fine_dynamic.append(torch.zeros_like(base_u))
    fine_dynamic_tensor = torch.stack(fine_dynamic[:dynamic_channels])
    inputs = build_model_input(static_30m, fine_dynamic_tensor, dx=dx, tile_id=f"synthetic-{idx:04d}")

    heights = torch.tensor(FU_XI_AGL_LEVELS, dtype=torch.float32)
    profile = torch.clamp(torch.log1p(heights) / torch.log1p(torch.tensor(100.0)), min=0.65, max=1.35)
    terrain_wake = 0.25 * torch.tanh(static_30m[2]) - 0.15 * torch.tanh(static_30m[3])
    u_levels = (base_u.unsqueeze(0) * profile.view(-1, 1, 1)) + terrain_wake.unsqueeze(0)
    v_levels = (base_v.unsqueeze(0) * profile.view(-1, 1, 1)) - 0.5 * terrain_wake.unsqueeze(0)
    w_levels = base_w.unsqueeze(0).expand(27, 300, 300) + 0.03 * terrain_wake.unsqueeze(0)
    target = torch.stack((u_levels, v_levels, w_levels), dim=0)

    sample = {
        "static_30m": inputs["static_30m"].float(),
        "dynamic_native": inputs["dynamic_native"].float(),
        "canonical_uv_100m": inputs["canonical_uv_100m"].float(),
        "coarse_dx": inputs["coarse_dx"].float(),
        "target": target.float(),
    }
    if z_out is not None:
        z_tensor = torch.as_tensor(z_out, dtype=torch.float32)
        if z_tensor.ndim != 1:
            raise ValueError("z_out must be a 1D sequence of AGL heights")
        sample["z_out"] = z_tensor
        if target_mode == "fuxi_interp":
            sample["target"] = interpolate_to_agl(target, torch.tensor(FU_XI_AGL_LEVELS, dtype=torch.float32), z_tensor).float()
        elif target_mode == "synthetic_aloft":
            sample["target"] = extend_wind_from_top_level(target, z_tensor).float()
            sample["valid_mask"] = torch.ones_like(z_tensor, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported synthetic target_mode: {target_mode}")
    return sample


class SyntheticTopoDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        length: int = 8,
        dx_values: Sequence[int] = (300,),
        static_channels: int = 5,
        dynamic_channels: int = 6,
    ) -> None:
        self.length = int(length)
        self.dx_values = tuple(int(dx) for dx in dx_values)
        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dx = self.dx_values[idx % len(self.dx_values)]
        return make_synthetic_sample(idx, dx=dx, static_channels=self.static_channels, dynamic_channels=self.dynamic_channels)
