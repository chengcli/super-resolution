from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from topo_ra.data.synthetic_dataset import SyntheticTopoDataset


class FuXiCaseDataset(Dataset[dict[str, torch.Tensor]]):
    """Load FuXi-CFD case folders, with a synthetic fallback for quick starts."""

    def __init__(
        self,
        root: str | Path | None = None,
        synthetic: bool = True,
        length: int = 8,
        dx_values: Sequence[int] = (300,),
        static_channels: int = 5,
        dynamic_channels: int = 6,
        case_offset: int = 0,
        cache: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.synthetic = synthetic or self.root is None or not self.root.exists()
        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        # Optional in-process cache of decoded cases. With num_workers=0 (the
        # local default here), this makes epochs 2..N skip the ~20 MB/case npz
        # decode entirely, which is the dominant per-epoch cost. The target is
        # stored as float16 (its on-disk dtype) to roughly halve memory.
        self.cache = bool(cache)
        self._cache: dict[int, dict[str, torch.Tensor]] = {}
        if self.synthetic:
            self.synthetic_dataset = SyntheticTopoDataset(
                length=length,
                dx_values=dx_values,
                static_channels=static_channels,
                dynamic_channels=dynamic_channels,
            )
            self.case_dirs: list[Path] = []
        else:
            candidates = [self.root] if (self.root / "inputs.npz").exists() else sorted(self.root.glob("case_*"))
            all_cases = [path for path in candidates if (path / "inputs.npz").exists() and (path / "outputs.npz").exists()]
            if not all_cases:
                raise FileNotFoundError(f"No FuXi-CFD case folders with inputs.npz and outputs.npz found under {self.root}")
            offset = max(0, int(case_offset))
            self.case_dirs = all_cases[offset : offset + max(1, int(length))]
            if not self.case_dirs:
                raise ValueError(f"No FuXi-CFD cases available after case_offset={offset} under {self.root}")
            self.synthetic_dataset = None

    def __len__(self) -> int:
        if self.synthetic:
            assert self.synthetic_dataset is not None
            return len(self.synthetic_dataset)
        return len(self.case_dirs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.synthetic:
            assert self.synthetic_dataset is not None
            return self.synthetic_dataset[idx]
        if self.cache and idx in self._cache:
            return self._decompress(self._cache[idx])
        sample = self._load_real_case(self.case_dirs[idx])
        if self.cache:
            self._cache[idx] = self._compress(sample)
        return sample

    @staticmethod
    def _compress(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stored = dict(sample)
        stored["target"] = sample["target"].to(torch.float16)
        return stored

    @staticmethod
    def _decompress(stored: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Return fresh float32 tensors so downstream .to(device)/augmentation never
        # mutates the cached copy.
        out = {key: value.clone() for key, value in stored.items()}
        out["target"] = stored["target"].to(torch.float32)
        return out

    def _load_real_case(self, case_dir: Path) -> dict[str, torch.Tensor]:
        with np.load(case_dir / "inputs.npz") as inputs:
            dem = torch.from_numpy(inputs["dem"].astype("float32"))
            roughness = torch.from_numpy(inputs["roughness"].astype("float32"))
            u_100m = torch.from_numpy(inputs["u_100m"].astype("float32"))
            v_100m = torch.from_numpy(inputs["v_100m"].astype("float32"))

        slope_y, slope_x = torch.gradient(dem, spacing=(30.0, 30.0))
        curvature = torch.gradient(slope_x, spacing=(30.0, 30.0))[1] + torch.gradient(slope_y, spacing=(30.0, 30.0))[0]
        static_fields = [dem / 1000.0, roughness, slope_x * 20.0, slope_y * 20.0, curvature * 10.0]
        while len(static_fields) < self.static_channels:
            static_fields.append(torch.zeros_like(dem))
        static_30m = torch.stack(static_fields[: self.static_channels]).float()

        dynamic_fields = [u_100m, v_100m]
        while len(dynamic_fields) < self.dynamic_channels:
            dynamic_fields.append(torch.zeros_like(u_100m))
        dynamic_native = torch.stack(dynamic_fields[: self.dynamic_channels]).float()
        canonical_uv_100m = torch.stack((u_100m, v_100m)).float()

        with np.load(case_dir / "outputs.npz") as outputs:
            target = torch.stack(
                (
                    torch.from_numpy(outputs["u"].astype("float32")),
                    torch.from_numpy(outputs["v"].astype("float32")),
                    torch.from_numpy(outputs["w"].astype("float32")),
                )
            )

        return {
            "static_30m": static_30m,
            "dynamic_native": dynamic_native,
            "canonical_uv_100m": canonical_uv_100m,
            "coarse_dx": torch.tensor(1000.0, dtype=torch.float32),
            "target": target.float(),
        }
