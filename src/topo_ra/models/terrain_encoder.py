from __future__ import annotations

import torch
from torch import nn

from topo_ra.models.projection import adaptive_avg_pool2d_compat


class StaticTerrainEncoder(nn.Module):
    """Encode 30 m static terrain fields onto a compact latent grid."""

    def __init__(self, in_channels: int = 5, embed_dim: int = 128, latent_size: int = 30) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, static_30m: torch.Tensor) -> torch.Tensor:
        if static_30m.ndim != 4:
            raise ValueError("static_30m must have shape [B, C_static, 300, 300]")
        if static_30m.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} static channels, got {static_30m.shape[1]}")
        if static_30m.shape[-2:] != (300, 300):
            raise ValueError(f"static_30m must be 300 x 300, got {tuple(static_30m.shape[-2:])}")

        pooled = adaptive_avg_pool2d_compat(static_30m, (self.latent_size, self.latent_size))
        return self.stem(pooled)
