from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .runner import StepResult, TwoResolutionRunner
from .types import MeshState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an observe-only two-resolution Snapy downscaling case."
    )
    parser.add_argument("config", help="Snapy shallow-water YAML config")
    parser.add_argument("--steps", type=int, default=1, help="Accepted steps to run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for saved low/prediction/truth tensor pairs",
    )
    parser.add_argument("--low-output-dir", type=Path, help="Snapy low-res output dir")
    parser.add_argument("--high-output-dir", type=Path, help="Snapy high-res output dir")
    parser.add_argument(
        "--make-snapy-outputs",
        action="store_true",
        help="Also call Snapy's configured output writers",
    )
    parser.add_argument(
        "--no-paddle-dist",
        action="store_true",
        help="Do not initialize Paddle/Torch distributed; use CPU directly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    runner = TwoResolutionRunner(
        args.config,
        low_output_dir=args.low_output_dir,
        high_output_dir=args.high_output_dir,
        make_outputs=args.make_snapy_outputs,
        use_paddle_dist=not args.no_paddle_dist,
    )
    for result in runner.run(max_steps=args.steps):
        print(_format_result(result), flush=True)
        if args.output_dir is not None:
            torch.save(_result_to_cpu(result), args.output_dir / f"step_{result.cycle:06d}.pt")
    return 0


def _format_result(result: StepResult) -> str:
    metrics = " ".join(f"{name}={value:.6g}" for name, value in result.metrics.items())
    suffix = f" {metrics}" if metrics else ""
    return f"cycle={result.cycle} time={result.time:.12g}{suffix}"


def _result_to_cpu(result: StepResult) -> dict[str, object]:
    return {
        "cycle": result.cycle,
        "time": result.time,
        "metrics": result.metrics,
        "low": _state_to_cpu(result.low),
        "prediction": _state_to_cpu(result.prediction),
        "truth": _state_to_cpu(result.truth),
    }


def _state_to_cpu(state: MeshState) -> MeshState:
    return [
        {name: tensor.detach().cpu() for name, tensor in block.items()}
        for block in state
    ]


if __name__ == "__main__":
    raise SystemExit(main())
