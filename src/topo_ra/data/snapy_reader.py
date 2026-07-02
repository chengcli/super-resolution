from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

HYDRO_W_ORDER = ("rho", "w", "u", "v", "p", "q", "q2", "q3")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_snapy_state(path: str | Path) -> dict[str, Any]:
    """Read a sidecar snapy-like state export.

    The current reader supports synthetic `.npz` files. TorchScript `.part` and `.restart`
    files are recognized but intentionally left as skeleton integration points.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"snapy input not found: {p}")
    if p.suffix == ".npz":
        buffers = _load_npz(p)
    elif p.suffix in {".part", ".restart"}:
        raise NotImplementedError(
            "TorchScript .part/.restart snapy reading is scaffolded; "
            "export a synthetic .npz sidecar to exercise the pipeline."
        )
    else:
        raise ValueError(f"Unsupported snapy sidecar format: {p.suffix}")

    if "hydro_w" not in buffers:
        raise KeyError("snapy sidecar must contain a hydro_w buffer")
    hydro_w = buffers["hydro_w"]
    if hydro_w.ndim not in (4, 5):
        raise ValueError("hydro_w must have shape [var, x3, x2, x1] or [time, var, x3, x2, x1]")
    return {"path": str(p), "buffers": buffers, "hydro_w": hydro_w, "hydro_u": buffers.get("hydro_u")}


def inspect_snapy_input(path: str | Path) -> dict[str, Any]:
    state = read_snapy_state(path)
    buffers = state["buffers"]
    hydro_w = state["hydro_w"]
    var_axis = 1 if hydro_w.ndim == 5 else 0
    inferred = {name: i for i, name in enumerate(HYDRO_W_ORDER[: hydro_w.shape[var_axis]])}
    dx = None
    for key in ("dx", "grid_dx", "spacing"):
        if key in buffers:
            dx = float(np.asarray(buffers[key]).reshape(-1)[0])
            break
    report = {
        "available_buffers": sorted(buffers.keys()),
        "shapes": {key: tuple(value.shape) for key, value in buffers.items()},
        "variable_mapping": inferred,
        "grid_spacing": dx,
    }
    print("SNAPY INPUT INSPECTION")
    print(f"available buffers: {', '.join(report['available_buffers'])}")
    for key, shape in report["shapes"].items():
        print(f"{key}: {shape}")
    print(f"inferred variable mapping: {report['variable_mapping']}")
    print(f"inferred grid spacing: {report['grid_spacing']}")
    return report


def _vertical_heights_from_buffers(buffers: dict[str, np.ndarray], levels: int) -> np.ndarray:
    for key in ("z_agl", "heights_agl", "heights", "z", "x3"):
        if key in buffers:
            heights = np.asarray(buffers[key], dtype=np.float32).reshape(-1)
            if heights.size == levels:
                return heights
    return np.arange(levels, dtype=np.float32)


def extract_snapy_uvw_profile(path: str | Path, time_index: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Extract coarse Snapy u/v/w profile and AGL heights from a sidecar export.

    Returns:
      uvw: `[3, Z, Y, X]` ordered as `u, v, w`.
      heights: `[Z]` AGL heights when present, otherwise `0..Z-1`.
    """
    state = read_snapy_state(path)
    hydro_w = state["hydro_w"]
    if hydro_w.ndim == 5:
        hydro_w = hydro_w[time_index]
    mapping = {name: i for i, name in enumerate(HYDRO_W_ORDER[: hydro_w.shape[0]])}
    missing = [name for name in ("u", "v", "w") if name not in mapping]
    if missing:
        raise KeyError(f"snapy hydro_w is missing variables required for uvw profile: {missing}")
    uvw = np.stack((hydro_w[mapping["u"]], hydro_w[mapping["v"]], hydro_w[mapping["w"]]), axis=0).astype("float32")
    heights = _vertical_heights_from_buffers(state["buffers"], uvw.shape[1])
    return uvw, heights
