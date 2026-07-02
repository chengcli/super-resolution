from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        groups = 1
        for candidate in (8, 4, 2, 1):
            if embed_dim % candidate == 0:
                groups = candidate
                break
        self.net = nn.Sequential(
            nn.GroupNorm(groups, embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FusionProcessor(nn.Module):
    """Compact processor used in the encoder-processor-decoder core."""

    def __init__(self, embed_dim: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualConvBlock(embed_dim) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
