from __future__ import annotations

import math
from typing import Sequence

import torch

from topo_ra.data.vertical_interp import FU_XI_AGL_LEVELS, interpolate_to_agl


def extend_wind_from_top_level(
    target_27: torch.Tensor,
    z_out: torch.Tensor | Sequence[float],
    *,
    alpha: float = 0.12,
    veer_max_deg: float = 10.0,
    veer_reference_height_m: float = 3000.0,
    w_decay_m: float = 700.0,
) -> torch.Tensor:
    """Extend FuXi 27-level u/v/w truth upward from the top public level.

    For z above 214.29 m, the formula is pointwise and height-only:

    ``[u(z), v(z)] = (z / z_top)^alpha * R(theta(z)) * [u_top, v_top]``
    ``w(z) = w_top * exp(-(z - z_top) / w_decay_m)``

    where ``theta(z)`` grows logarithmically from zero at ``z_top`` to
    ``veer_max_deg`` at ``veer_reference_height_m``. No spatial smoothing,
    pooling, or filtering is applied, so high-resolution structure in the FuXi
    top slice is carried directly into the pseudo-aloft target.
    """
    z = torch.as_tensor(z_out, dtype=target_27.dtype, device=target_27.device)
    if z.ndim != 1:
        raise ValueError("z_out must be a 1D sequence of AGL heights")
    if target_27.shape[0] != 3 or target_27.shape[1] != len(FU_XI_AGL_LEVELS):
        raise ValueError("target_27 must have shape [3, 27, Y, X]")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if veer_reference_height_m <= FU_XI_AGL_LEVELS[-1]:
        raise ValueError("veer_reference_height_m must exceed the FuXi top level")
    if w_decay_m <= 0.0:
        raise ValueError("w_decay_m must be positive")

    source_heights = torch.tensor(FU_XI_AGL_LEVELS, dtype=target_27.dtype, device=target_27.device)
    top_z = source_heights[-1]
    below_or_in_range = interpolate_to_agl(target_27, source_heights, z)

    top = target_27[:, -1]
    z_clamped = z.clamp_min(top_z)
    dz = z_clamped - top_z
    profile = torch.clamp((z_clamped / top_z).pow(alpha), min=1.0)
    veer_fraction = torch.log(z_clamped / top_z).clamp_min(0.0) / math.log(veer_reference_height_m / float(top_z))
    veer = torch.deg2rad(torch.clamp(veer_fraction * veer_max_deg, min=0.0, max=veer_max_deg))
    w_decay = torch.exp(-dz / w_decay_m)

    cos_v = torch.cos(veer).view(-1, 1, 1)
    sin_v = torch.sin(veer).view(-1, 1, 1)
    scale = profile.view(-1, 1, 1)
    u_aloft = (top[0].unsqueeze(0) * cos_v - top[1].unsqueeze(0) * sin_v) * scale
    v_aloft = (top[0].unsqueeze(0) * sin_v + top[1].unsqueeze(0) * cos_v) * scale
    w_aloft = top[2].unsqueeze(0) * w_decay.view(-1, 1, 1)
    aloft = torch.stack((u_aloft, v_aloft, w_aloft), dim=0)

    mask = (z > top_z).view(1, -1, 1, 1)
    return torch.where(mask, aloft, below_or_in_range)
