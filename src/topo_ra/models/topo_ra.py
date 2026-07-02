from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from topo_ra.data.vertical_interp import FU_XI_AGL_LEVELS, heights_in_range, interpolate_to_agl
from topo_ra.models.decoder import DecoderHeads
from topo_ra.models.dynamic_encoder import CanonicalBranch, NativeResolutionDynamicEncoder, _as_batch_dx
from topo_ra.models.fusion_processor import FusionProcessor
from topo_ra.models.terrain_encoder import StaticTerrainEncoder


class BaselineBuilder(nn.Module):
    """Build a deterministic coarse-to-fine wind baseline for residual learning."""

    def __init__(self, levels: int = 27) -> None:
        super().__init__()
        self.levels = levels
        heights = torch.tensor(FU_XI_AGL_LEVELS, dtype=torch.float32)
        # Mild profile keeps the synthetic baseline height-aware without claiming
        # a physical boundary-layer model.
        profile = torch.clamp(torch.log1p(heights) / torch.log1p(torch.tensor(100.0)), min=0.65, max=1.35)
        self.register_buffer("profile", profile.view(1, 1, levels, 1, 1), persistent=False)

    def forward(self, dynamic_native: torch.Tensor, canonical_uv_100m: torch.Tensor) -> torch.Tensor:
        if dynamic_native.ndim != 4:
            raise ValueError("dynamic_native must have shape [B, C_dynamic, Hc, Wc]")
        batch = dynamic_native.shape[0]
        source_uv = dynamic_native[:, :2] if dynamic_native.shape[1] >= 2 else canonical_uv_100m
        uv_fine = F.interpolate(source_uv, size=(300, 300), mode="bilinear", align_corners=False)
        if dynamic_native.shape[1] >= 3:
            w_fine = F.interpolate(dynamic_native[:, 2:3], size=(300, 300), mode="bilinear", align_corners=False)
        else:
            w_fine = torch.zeros(batch, 1, 300, 300, device=dynamic_native.device, dtype=dynamic_native.dtype)

        uv = uv_fine.unsqueeze(2).expand(batch, 2, self.levels, 300, 300) * self.profile.to(uv_fine.dtype)
        w = w_fine.unsqueeze(2).expand(batch, 1, self.levels, 300, 300)
        return torch.cat((uv, w), dim=1)


class DepthwiseRefineBlock(nn.Module):
    """Cheap full-resolution residual block for 30 m refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 1
        for candidate in (8, 4, 2, 1):
            if channels % candidate == 0:
                groups = candidate
                break
        self.norm = nn.GroupNorm(groups, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


FLOW_FEATURE_CHANNELS = 7


def compute_flow_features(static_30m: torch.Tensor, dynamic_native: torch.Tensor) -> torch.Tensor:
    """Explicit terrain-flow interaction channels at the 30 m grid.

    Returns [B, 7, H, W] stacking: upsampled coarse u, v, wind speed, unit-flow
    (u_hat, v_hat), and the along-flow (upwind) and cross-flow terrain-slope
    components. These give the refiner the wind direction it needs to place
    speed-ups on windward slopes and wakes on lee slopes with the correct sign.

    Slope is read from static channels 2 (slope_x) and 3 (slope_y) when present;
    if the static stack is thinner, the slope-derived channels are zeros.
    """
    fine_hw = static_30m.shape[-2:]
    uv_fine = F.interpolate(dynamic_native[:, :2], size=fine_hw, mode="bilinear", align_corners=False)
    u = uv_fine[:, 0:1]
    v = uv_fine[:, 1:2]
    speed = torch.sqrt((u * u + v * v).clamp_min(1e-8))
    u_hat = u / speed
    v_hat = v / speed
    if static_30m.shape[1] >= 4:
        slope_x = static_30m[:, 2:3]
        slope_y = static_30m[:, 3:4]
    else:
        slope_x = torch.zeros_like(u)
        slope_y = torch.zeros_like(u)
    upwind_slope = slope_x * u_hat + slope_y * v_hat
    crosswind_slope = -slope_x * v_hat + slope_y * u_hat
    return torch.cat((u, v, speed, u_hat, v_hat, upwind_slope, crosswind_slope), dim=1)


def _coarse_profile_to_fine(
    coarse_profile: torch.Tensor,
    coarse_profile_heights: torch.Tensor,
    z_out: torch.Tensor,
    fine_hw: tuple[int, int],
) -> torch.Tensor:
    """Interpolate a coarse Snapy u/v/w column to query heights and 30 m grid.

    Args:
      coarse_profile: [B, 3, Zc, Hc, Wc] coarse u/v/w profile.
      coarse_profile_heights: [Zc] AGL heights in meters.
      z_out: [Zq] query heights in meters AGL.
    Returns:
      [B, 3, Zq, H, W] bilinearly upsampled profile at ``z_out``.
    """
    if coarse_profile.ndim != 5 or coarse_profile.shape[1] != 3:
        raise ValueError("coarse_profile must have shape [B, 3, Zc, Hc, Wc]")
    heights = torch.as_tensor(
        coarse_profile_heights,
        device=coarse_profile.device,
        dtype=coarse_profile.dtype,
    )
    if heights.ndim != 1 or heights.numel() != coarse_profile.shape[2]:
        raise ValueError("coarse_profile_heights must be 1D with length matching coarse_profile Zc")
    z = z_out.to(device=coarse_profile.device, dtype=coarse_profile.dtype)
    at_z = interpolate_to_agl(coarse_profile, heights, z)
    batch, channels, levels, coarse_h, coarse_w = at_z.shape
    flat = at_z.reshape(batch * channels * levels, 1, coarse_h, coarse_w)
    fine = F.interpolate(flat, size=fine_hw, mode="bilinear", align_corners=False)
    return fine.reshape(batch, channels, levels, fine_hw[0], fine_hw[1])


class FlowAwareRefiner(nn.Module):
    """Full-resolution refiner conditioned on explicit terrain-flow features."""

    def __init__(
        self,
        static_channels: int,
        embed_dim: int,
        flow_channels: int = FLOW_FEATURE_CHANNELS,
        out_variables: int = 3,
        levels: int = 27,
        refine_channels: int = 16,
        refine_depth: int = 2,
    ) -> None:
        super().__init__()
        self.out_variables = out_variables
        self.levels = levels
        self.stem = nn.Sequential(
            nn.Conv2d(static_channels + flow_channels, refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_channels, refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.latent_proj = nn.Conv2d(embed_dim, refine_channels, kernel_size=1)
        self.blocks = nn.Sequential(*(DepthwiseRefineBlock(refine_channels) for _ in range(refine_depth)))
        self.out = nn.Conv2d(refine_channels, out_variables * levels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, static_30m: torch.Tensor, flow_features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        fine_hw = static_30m.shape[-2:]
        latent_fine = F.interpolate(latent, size=fine_hw, mode="bilinear", align_corners=False)
        stem_in = torch.cat((static_30m, flow_features), dim=1)
        features = self.stem(stem_in) + self.latent_proj(latent_fine)
        delta = self.out(self.blocks(features))
        batch = delta.shape[0]
        return delta.view(batch, self.out_variables, self.levels, fine_hw[0], fine_hw[1])


class AloftProfileBasisHead(nn.Module):
    """Fast aloft head using a physical profile plus low-rank vertical basis.

    The full-resolution CNN runs once per sample and predicts a small set of
    spatial parameter maps. Querying more heights then only expands cheap
    analytic height functions and low-rank basis terms, so inference cost grows
    weakly with ``Z_out`` instead of rerunning full-resolution convolutions for
    every height chunk.
    """

    def __init__(
        self,
        static_channels: int,
        embed_dim: int,
        flow_channels: int = FLOW_FEATURE_CHANNELS,
        hidden_channels: int = 16,
        depth: int = 2,
        basis_rank: int = 4,
        base_alpha: float = 0.12,
        base_veer_deg: float = 10.0,
        reference_height_m: float = 3000.0,
        base_w_decay_m: float = 700.0,
    ) -> None:
        super().__init__()
        self.top_height_m = float(FU_XI_AGL_LEVELS[-1])
        self.reference_height_m = float(reference_height_m)
        self.base_alpha = float(base_alpha)
        self.base_veer_deg = float(base_veer_deg)
        self.base_w_decay_m = float(base_w_decay_m)
        self.basis_rank = max(1, int(basis_rank))
        self.stem = nn.Sequential(
            nn.Conv2d(static_channels + flow_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.latent_proj = nn.Conv2d(embed_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*(DepthwiseRefineBlock(hidden_channels) for _ in range(depth)))
        # profile params: alpha delta, veer delta, w-decay delta, speed-bias.
        # basis params: component-specific coefficients for low-rank vertical
        # residuals, shaped [B, 3, K, H, W] after view().
        self.out = nn.Conv2d(hidden_channels, 4 + 3 * self.basis_rank, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _height_coordinate(self, z_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top = z_out.new_tensor(self.top_height_m)
        ref = z_out.new_tensor(self.reference_height_m)
        z_safe = z_out.clamp_min(1.0)
        ratio = (z_safe / top).clamp_min(1.0)
        log_ref = torch.log(ref / top).clamp_min(1e-6)
        x = (torch.log(ratio) / log_ref).clamp(0.0, 2.0)
        aloft_gate = (z_out > top).to(z_out.dtype)
        return ratio, x, aloft_gate

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        basis = [
            x,
            x * x,
            torch.sin(torch.pi * x),
            torch.sin(2.0 * torch.pi * x),
            x * torch.clamp(1.0 - x, min=0.0),
            x * x * x,
            torch.sin(3.0 * torch.pi * x),
            torch.sin(4.0 * torch.pi * x),
        ]
        if self.basis_rank <= len(basis):
            return torch.stack(basis[: self.basis_rank], dim=0)
        extra = []
        for power in range(len(basis) + 1, self.basis_rank + 1):
            extra.append(x.pow(power))
        return torch.stack([*basis, *extra], dim=0)

    def forward(
        self,
        static_30m: torch.Tensor,
        flow_features: torch.Tensor,
        latent: torch.Tensor,
        pred_fixed: torch.Tensor,
        fixed_interp: torch.Tensor,
        z_out: torch.Tensor,
        coarse_profile: torch.Tensor | None = None,
        coarse_profile_heights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_out.ndim != 1:
            raise ValueError("z_out must be a 1D tensor of AGL heights")
        fine_hw = static_30m.shape[-2:]
        latent_fine = F.interpolate(latent, size=fine_hw, mode="bilinear", align_corners=False)
        base = self.stem(torch.cat((static_30m, flow_features), dim=1)) + self.latent_proj(latent_fine)
        raw = self.out(self.blocks(base))
        batch, _, height, width = raw.shape
        profile_raw = raw[:, :4]
        coeff_raw = raw[:, 4:].view(batch, 3, self.basis_rank, height, width)

        z = z_out.to(device=base.device, dtype=base.dtype)
        ratio, x, gate = self._height_coordinate(z)
        ratio = ratio.view(1, 1, -1, 1, 1)
        x_view = x.view(1, 1, -1, 1, 1)
        gate_view = gate.view(1, 1, -1, 1, 1)

        alpha = self.base_alpha + 0.20 * torch.tanh(profile_raw[:, 0:1]).unsqueeze(2)
        veer_deg = self.base_veer_deg + 25.0 * torch.tanh(profile_raw[:, 1:2]).unsqueeze(2)
        w_decay = self.base_w_decay_m * torch.exp(1.0 * torch.tanh(profile_raw[:, 2:3]).unsqueeze(2))
        speed_bias = 0.20 * torch.tanh(profile_raw[:, 3:4]).unsqueeze(2) * x_view

        if coarse_profile is not None:
            if coarse_profile_heights is None:
                raise ValueError("coarse_profile_heights is required when coarse_profile is provided")
            coarse_profile = coarse_profile.to(device=base.device, dtype=base.dtype)
            coarse_profile_heights = coarse_profile_heights.to(device=base.device, dtype=base.dtype)
            profile_base = _coarse_profile_to_fine(coarse_profile, coarse_profile_heights, z, fine_hw)
            uv_scale = (1.0 + 0.20 * torch.tanh(profile_raw[:, 0:1]).unsqueeze(2) * x_view).clamp(0.5, 1.5)
            veer = torch.deg2rad(15.0 * torch.tanh(profile_raw[:, 1:2]).unsqueeze(2) * x_view)
            w_scale = (1.0 + 0.50 * torch.tanh(profile_raw[:, 2:3]).unsqueeze(2) * x_view).clamp(0.3, 2.0)
            w_bias = 0.50 * torch.tanh(profile_raw[:, 3:4]).unsqueeze(2) * x_view
            u_base = profile_base[:, 0:1]
            v_base = profile_base[:, 1:2]
            cos_v = torch.cos(veer)
            sin_v = torch.sin(veer)
            u_profile = (u_base * cos_v - v_base * sin_v) * uv_scale
            v_profile = (u_base * sin_v + v_base * cos_v) * uv_scale
            w_profile = profile_base[:, 2:3] * w_scale + w_bias
            profile = torch.cat((u_profile, v_profile, w_profile), dim=1)
        else:
            scale = ratio.pow(alpha).clamp(0.2, 5.0) * (1.0 + speed_bias).clamp(0.5, 1.5)
            veer_fraction = x_view.clamp_min(0.0)
            veer = torch.deg2rad(veer_deg * veer_fraction)
            cos_v = torch.cos(veer)
            sin_v = torch.sin(veer)

            top = pred_fixed[:, :, -1].unsqueeze(2)
            u0 = top[:, 0:1]
            v0 = top[:, 1:2]
            w0 = top[:, 2:3]
            u_profile = (u0 * cos_v - v0 * sin_v) * scale
            v_profile = (u0 * sin_v + v0 * cos_v) * scale
            dz = (z - z.new_tensor(self.top_height_m)).clamp_min(0.0).view(1, 1, -1, 1, 1)
            w_profile = w0 * torch.exp(-dz / w_decay.clamp_min(1.0))
            profile = torch.cat((u_profile, v_profile, w_profile), dim=1)

        basis = self._basis(x).to(device=base.device, dtype=base.dtype)
        basis = basis * gate.view(1, -1)
        basis_residual = torch.einsum("bckhw,kz->bczhw", coeff_raw, basis)
        correction = gate_view * (profile + basis_residual - fixed_interp)
        return correction, {
            "aloft_profile": profile,
            "basis_residual": basis_residual,
            "profile_params": profile_raw,
        }


class TopoRA(nn.Module):
    """Terrain-aware, resolution-adaptive residual downscaler."""

    def __init__(
        self,
        static_channels: int = 5,
        dynamic_channels: int = 6,
        embed_dim: int = 128,
        depth: int = 4,
        latent_size: int = 30,
        levels: int = 27,
        refine_channels: int = 16,
        refine_depth: int = 2,
        z_channels: int = 16,
        z_depth: int = 2,
        z_basis_rank: int = 4,
    ) -> None:
        super().__init__()
        if levels != len(FU_XI_AGL_LEVELS):
            raise ValueError(
                f"TopoRA is fixed to the {len(FU_XI_AGL_LEVELS)} public FuXi AGL levels; got levels={levels}. "
                "Use predict_at_heights(...) for future full-height/Snapy outputs."
            )
        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.levels = levels
        # Fixed AGL heights of the 27 decoder levels. This lets callers query
        # `forward()` output by height without changing the decoder itself.
        self.register_buffer("output_heights", torch.tensor(FU_XI_AGL_LEVELS, dtype=torch.float32), persistent=False)
        self.static_encoder = StaticTerrainEncoder(static_channels, embed_dim, latent_size)
        self.native_encoder = NativeResolutionDynamicEncoder(dynamic_channels, embed_dim, latent_size)
        self.canonical_branch = CanonicalBranch(embed_dim, latent_size)
        self.fuse = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)
        self.processor = FusionProcessor(embed_dim, depth)
        self.decoder = DecoderHeads(embed_dim, out_variables=3, levels=levels)
        self.baseline = BaselineBuilder(levels)
        self.refiner = FlowAwareRefiner(
            static_channels=static_channels,
            embed_dim=embed_dim,
            flow_channels=FLOW_FEATURE_CHANNELS,
            out_variables=3,
            levels=levels,
            refine_channels=refine_channels,
            refine_depth=refine_depth,
        )
        self.z_head = AloftProfileBasisHead(
            static_channels=static_channels,
            embed_dim=embed_dim,
            flow_channels=FLOW_FEATURE_CHANNELS,
            hidden_channels=z_channels,
            depth=z_depth,
            basis_rank=z_basis_rank,
        )

    def _forward_parts(
        self,
        static_30m: torch.Tensor,
        dynamic_native: torch.Tensor,
        canonical_uv_100m: torch.Tensor,
        coarse_dx: torch.Tensor | float | int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = static_30m.shape[0]
        if dynamic_native.shape[0] != batch or canonical_uv_100m.shape[0] != batch:
            raise ValueError("All inputs must share the same batch size")
        dx = _as_batch_dx(coarse_dx, batch, static_30m.device)
        supported = torch.tensor([300.0, 500.0, 1000.0], device=dx.device)
        if not torch.isin(dx, supported).all():
            raise ValueError(f"coarse_dx values must be in {{300, 500, 1000}}, got {dx.detach().cpu().tolist()}")

        static_latent = self.static_encoder(static_30m)
        native_latent = self.native_encoder(dynamic_native, dx)
        canonical_latent = self.canonical_branch(canonical_uv_100m)
        fused = self.fuse(torch.cat((static_latent, native_latent, canonical_latent), dim=1))
        processed = self.processor(fused)
        residual = self.decoder(processed)
        baseline = self.baseline(dynamic_native, canonical_uv_100m)
        flow_features = compute_flow_features(static_30m, dynamic_native)
        refinement = self.refiner(static_30m, flow_features, processed)
        pred = baseline + residual + refinement
        return pred, {
            "processed": processed,
            "baseline": baseline,
            "residual": residual,
            "flow_features": flow_features,
            "refinement": refinement,
        }

    def forward(
        self,
        static_30m: torch.Tensor,
        dynamic_native: torch.Tensor,
        canonical_uv_100m: torch.Tensor,
        coarse_dx: torch.Tensor | float | int,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run TopoRA.

        Inputs:
          static_30m: [B, C_static, 300, 300]
          dynamic_native: [B, C_dynamic, Hc, Wc], Hc/Wc in {9, 18, 30}
          canonical_uv_100m: [B, 2, 9, 9]
          coarse_dx: scalar or [B]
        Output:
          pred: [B, 3, 27, 300, 300]
        """
        pred, parts = self._forward_parts(static_30m, dynamic_native, canonical_uv_100m, coarse_dx)
        if return_diagnostics:
            return pred, {key: value for key, value in parts.items() if key != "processed" and key != "flow_features"}
        return pred

    def predict_at_heights(
        self,
        static_30m: torch.Tensor,
        dynamic_native: torch.Tensor,
        canonical_uv_100m: torch.Tensor,
        coarse_dx: torch.Tensor | float | int,
        z_out: torch.Tensor,
        return_diagnostics: bool = False,
        coarse_profile: torch.Tensor | None = None,
        coarse_profile_heights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Query the fixed-27-level prediction at arbitrary AGL heights.

        This keeps the fixed 27-level path as a near-ground warm-start. Above
        the public FuXi ceiling, ``z_head`` switches to a fast aloft profile:
        a pointwise power/veer/decay profile anchored at the predicted top FuXi
        level plus a low-rank vertical basis residual. The full-resolution CNN
        runs once per sample, so query cost is not dominated by ``Z_out``.

        The model is only ever supervised on the public 27 near-ground FuXi
        levels (up to 214.29 m AGL). Heights outside that span require
        extrapolation and must not be reported as FuXi-distilled accuracy;
        ``valid_mask`` marks which ``z_out`` entries fall inside the supervised
        range so callers can gate or report the aloft outputs separately.

        Args:
          z_out: 1D tensor of query heights in meters AGL, shape ``[Z_out]``.
          coarse_profile: optional Snapy coarse u/v/w profile, shape
            ``[B, 3, Zc, Hc, Wc]``. When provided, aloft predictions use this
            multi-height profile as the vertical backbone instead of the
            analytic top-level extrapolation.
          coarse_profile_heights: 1D AGL heights for ``coarse_profile``.
        Returns:
          pred_at_z: ``[B, 3, Z_out, 300, 300]``.
          valid_mask: ``[Z_out]`` bool, True where ``z_out`` is within the
            FuXi-supervised range and does not require extrapolation.
          diagnostics: only when ``return_diagnostics=True``; the unmodified
            ``forward`` diagnostics, still at the fixed 27 levels.
        """
        if z_out.ndim != 1:
            raise ValueError("z_out must be a 1D tensor of AGL heights")

        pred_fixed, parts = self._forward_parts(static_30m, dynamic_native, canonical_uv_100m, coarse_dx)

        source_heights = self.output_heights.to(device=pred_fixed.device, dtype=pred_fixed.dtype)
        z_out = z_out.to(device=pred_fixed.device, dtype=pred_fixed.dtype)
        fixed_interp = interpolate_to_agl(pred_fixed, source_heights, z_out)
        z_residual, z_head_diagnostics = self.z_head(
            static_30m,
            parts["flow_features"],
            parts["processed"],
            pred_fixed,
            fixed_interp,
            z_out,
            coarse_profile=coarse_profile,
            coarse_profile_heights=coarse_profile_heights,
        )
        pred_at_z = fixed_interp + z_residual
        valid_mask = heights_in_range(z_out, source_heights)

        if return_diagnostics:
            diagnostics = {
                "baseline": parts["baseline"],
                "residual": parts["residual"],
                "refinement": parts["refinement"],
                "z_residual": z_residual,
                "fixed_interp": fixed_interp,
                **z_head_diagnostics,
            }
            return pred_at_z, valid_mask, diagnostics
        return pred_at_z, valid_mask
