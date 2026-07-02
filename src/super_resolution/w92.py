from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .types import MeshState

FACE_NAMES = ("+X", "+Y", "-X", "+Z", "-Y", "-Z")

H0 = 8000.0
G = 9.80616
OMG = 7.848e-6
A = 6.37122e6
K = 7.848e-6
R = 4.0
OM_EARTH = 7.292e-5


def initialize_w92(mesh, config: dict[str, Any], device: torch.device) -> MeshState:
    block_vars: MeshState = []
    for face_id, block in enumerate(mesh.blocks):
        coord = block.module("coord")
        x2v = coord.buffer("x2v").detach().cpu().numpy()
        x3v = coord.buffer("x3v").detach().cpu().numpy()
        alpha, beta = np.meshgrid(x2v, x3v)
        lon, lat = ab_to_lonlat(FACE_NAMES[face_id], alpha, beta)
        gh, east, north = rossby_haurwitz_wave(lon, lat)
        vel2, vel3 = uv_to_contra(face_id, alpha, beta, lon, lat, east, north)

        w = torch.zeros((4, x3v.size, x2v.size, 1), dtype=torch.float64)
        w[0, :, :, 0] = torch.from_numpy(gh)
        w[1, :, :, 0] = 0.0
        w[2, :, :, 0] = torch.from_numpy(vel2)
        w[3, :, :, 0] = torch.from_numpy(vel3)
        block_vars.append({"hydro_w": w.to(device)})
    return block_vars


def ab_to_lonlat(
    face: str, alpha: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.tan(alpha)
    y = np.tan(beta)
    radius = np.sqrt(x * x + y * y + 1.0)
    if face == "+X":
        lon = alpha.copy()
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "+Y":
        lon = alpha + 0.5 * np.pi
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "-X":
        lon = alpha + np.pi
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "-Y":
        lon = alpha + 1.5 * np.pi
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "+Z":
        lon = np.arctan2(x, -y)
        lat = np.arcsin(1.0 / radius)
    elif face == "-Z":
        lon = np.arctan2(x, y)
        lat = -np.arcsin(1.0 / radius)
    else:
        raise ValueError(f"unknown cubed-sphere face {face!r}")
    return np.where(lon < 0.0, lon + 2 * np.pi, lon), lat


def rossby_haurwitz_wave(
    lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clat = np.cos(lat)
    slat = np.sin(lat)
    aa = 0.5 * OMG * (2 * OM_EARTH + OMG) * clat**2 + 0.25 * K * K * (
        (R + 1) * clat ** (2 * R + 2)
        + (2 * R * R - R - 2) * clat ** (2 * R)
        - 2 * R * R * clat ** (2 * R - 2)
    )
    bb = (
        2
        * (OM_EARTH + OMG)
        * K
        / ((R + 1) * (R + 2))
        * clat**R
        * ((R * R + 2 * R + 2) - (R + 1) ** 2 * clat**2)
    )
    cc = 0.25 * K * K * clat ** (2 * R) * ((R + 1) * clat**2 - (R + 2))
    gh = G * H0 + A * A * aa + A * A * bb * np.cos(R * lon) + A * A * cc * np.cos(2 * R * lon)
    east = A * OMG * clat + A * K * clat ** (R - 1) * np.cos(R * lon) * (
        R * slat * slat - clat * clat
    )
    north = -A * K * R * clat ** (R - 1) * np.sin(R * lon) * slat
    return gh, east, north


def uv_to_contra(
    face_id: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from paddle import cubed_sphere_remap as csr

    def east_north(v2: np.ndarray, v3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx, gy, gz = csr._local_contra_to_global_xyz(
            face_id, np.zeros_like(v2), v2, v3, alpha, beta
        )
        out_east = -np.sin(lon) * gx + np.cos(lon) * gy
        out_north = (
            -np.sin(lat) * np.cos(lon) * gx
            - np.sin(lat) * np.sin(lon) * gy
            + np.cos(lat) * gz
        )
        return out_east, out_north

    one = np.ones_like(alpha)
    zero = np.zeros_like(alpha)
    e1, n1 = east_north(one, zero)
    e2, n2 = east_north(zero, one)
    det = e1 * n2 - e2 * n1
    vel2 = (n2 * east - e2 * north) / det
    vel3 = (-n1 * east + e1 * north) / det
    return vel2, vel3
