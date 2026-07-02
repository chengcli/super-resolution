from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _as_batch_dx(coarse_dx: torch.Tensor | float | int, batch: int, device: torch.device) -> torch.Tensor:
    dx = torch.as_tensor(coarse_dx, device=device, dtype=torch.float32)
    if dx.ndim == 0:
        dx = dx.repeat(batch)
    if dx.shape != (batch,):
        raise ValueError(f"coarse_dx must be scalar or [B], got {tuple(dx.shape)} for B={batch}")
    return dx


def _coord_planes(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((xx, yy), dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)
    return coords


class NativeResolutionDynamicEncoder(nn.Module):
    """Encode native 9/18/30 coarse dynamic inputs with coordinate and dx channels."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 128, latent_size: int = 30) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 3, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, dynamic_native: torch.Tensor, coarse_dx: torch.Tensor | float | int) -> torch.Tensor:
        if dynamic_native.ndim != 4:
            raise ValueError("dynamic_native must have shape [B, C_dynamic, Hc, Wc]")
        if dynamic_native.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} dynamic channels, got {dynamic_native.shape[1]}")
        if dynamic_native.shape[-2:] not in {(9, 9), (18, 18), (30, 30)}:
            raise ValueError(f"Unsupported native grid shape: {tuple(dynamic_native.shape[-2:])}")

        batch, _, height, width = dynamic_native.shape
        dx = _as_batch_dx(coarse_dx, batch, dynamic_native.device)
        coords = _coord_planes(batch, height, width, dynamic_native.device, dynamic_native.dtype)
        dx_plane = (dx / 1000.0).to(dynamic_native.dtype).view(batch, 1, 1, 1).expand(batch, 1, height, width)
        x = torch.cat((dynamic_native, coords, dx_plane), dim=1)
        encoded = self.net(x)
        return F.interpolate(encoded, size=(self.latent_size, self.latent_size), mode="bilinear", align_corners=False)


class CanonicalBranch(nn.Module):
    """Teacher-compatible canonical branch for 9 x 9 u/v at 100 m AGL."""

    def __init__(self, embed_dim: int = 128, latent_size: int = 30) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Conv2d(4, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, canonical_uv_100m: torch.Tensor) -> torch.Tensor:
        if canonical_uv_100m.ndim != 4:
            raise ValueError("canonical_uv_100m must have shape [B, 2, 9, 9]")
        if canonical_uv_100m.shape[1:] != (2, 9, 9):
            raise ValueError(f"canonical_uv_100m must be [B, 2, 9, 9], got {tuple(canonical_uv_100m.shape)}")
        batch = canonical_uv_100m.shape[0]
        coords = _coord_planes(batch, 9, 9, canonical_uv_100m.device, canonical_uv_100m.dtype)
        encoded = self.net(torch.cat((canonical_uv_100m, coords), dim=1))
        return F.interpolate(encoded, size=(self.latent_size, self.latent_size), mode="bilinear", align_corners=False)
