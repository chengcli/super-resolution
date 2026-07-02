from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DecoderHeads(nn.Module):
    """Decode latent features into u/v/w residuals at 27 vertical levels."""

    def __init__(self, embed_dim: int = 128, out_variables: int = 3, levels: int = 27) -> None:
        super().__init__()
        self.out_variables = out_variables
        self.levels = levels
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_variables * levels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: torch.Tensor, fine_size: tuple[int, int] = (300, 300)) -> torch.Tensor:
        residual = self.net(latent)
        residual = F.interpolate(residual, size=fine_size, mode="bilinear", align_corners=False)
        batch = residual.shape[0]
        return residual.view(batch, self.out_variables, self.levels, fine_size[0], fine_size[1])
