#!/usr/bin/env bash
# Run process-isolated TopoRA online training until convergence or a max budget.
#
# Usage:
#   scripts/run_online_training_until_converged.sh [base_config.yaml]
#
# Environment overrides:
#   OUTPUT_ROOT=/data00/topora_online_w92_tiny_2gpu_100k
#   MAX_UPDATES=100000
#   CHUNK_UPDATES=100
#   MIN_UPDATES_BEFORE_CONVERGENCE=500
#   CONVERGENCE_WINDOW=5
#   CONVERGENCE_MIN_IMPROVEMENT=0.0001
#   CONVERGENCE_METRIC=live_after_coarse_consistency_speed
#   PYTHON=python
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_CONFIG="${1:-configs/topora_online_data00_2gpu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data00/topora_online_w92_tiny_2gpu_100k}"
MAX_UPDATES="${MAX_UPDATES:-100000}"
CHUNK_UPDATES="${CHUNK_UPDATES:-100}"
MIN_UPDATES_BEFORE_CONVERGENCE="${MIN_UPDATES_BEFORE_CONVERGENCE:-500}"
CONVERGENCE_WINDOW="${CONVERGENCE_WINDOW:-5}"
CONVERGENCE_MIN_IMPROVEMENT="${CONVERGENCE_MIN_IMPROVEMENT:-0.0001}"
CONVERGENCE_METRIC="${CONVERGENCE_METRIC:-live_after_coarse_consistency_speed}"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p "$OUTPUT_ROOT/configs" "$OUTPUT_ROOT/logs"
SUMMARY="$OUTPUT_ROOT/summary.csv"
LATEST="$OUTPUT_ROOT/latest_checkpoint.txt"

if [ ! -f "$SUMMARY" ]; then
  printf "chunk,start_update,end_update,updates,accepted,rejected,metric_first,metric_last,metric_min,metric_mean,checkpoint,metrics_csv\n" > "$SUMMARY"
fi

"$PYTHON_BIN" - <<'EOF'
import importlib.util
import sys

missing = [name for name in ("snapy", "topo_ra", "super_resolution", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    sys.exit(
        f"missing packages: {', '.join(missing)}\n"
        "Run from this repo with PYTHONPATH=src or install with: pip install -e ."
    )
EOF

if [ ! -f "$BASE_CONFIG" ]; then
  echo "base config not found: $BASE_CONFIG" >&2
  exit 1
fi

completed_updates=$("$PYTHON_BIN" - "$SUMMARY" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print(int(rows[-1]["end_update"]) if rows else 0)
PY
)

next_chunk=$("$PYTHON_BIN" - "$SUMMARY" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(1)
    raise SystemExit
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print(len(rows) + 1)
PY
)

init_from=""
if [ -f "$LATEST" ]; then
  init_from="$(cat "$LATEST")"
fi

while [ "$completed_updates" -lt "$MAX_UPDATES" ]; do
  remaining=$((MAX_UPDATES - completed_updates))
  updates_this_chunk="$CHUNK_UPDATES"
  if [ "$remaining" -lt "$updates_this_chunk" ]; then
    updates_this_chunk="$remaining"
  fi

  chunk_name=$(printf "chunk_%06d" "$next_chunk")
  chunk_dir="$OUTPUT_ROOT/$chunk_name"
  chunk_config="$OUTPUT_ROOT/configs/${chunk_name}.yaml"
  chunk_log="$OUTPUT_ROOT/logs/${chunk_name}.log"
  mkdir -p "$chunk_dir"

  "$PYTHON_BIN" - "$BASE_CONFIG" "$chunk_config" "$chunk_dir" "$updates_this_chunk" "$init_from" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, out_config, out_dir, num_updates, init_from = sys.argv[1:6]
with open(base_config, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
if not isinstance(config, dict):
    raise SystemExit(f"Config must be a mapping: {base_config}")
config["output_dir"] = out_dir
config["num_updates"] = int(num_updates)
if init_from:
    config["init_from"] = init_from
Path(out_config).parent.mkdir(parents=True, exist_ok=True)
with open(out_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

  echo "[$(date -Is)] starting $chunk_name: updates=$updates_this_chunk start=$((completed_updates + 1)) config=$chunk_config"
  "$PYTHON_BIN" -m super_resolution.topora_online --config "$chunk_config" 2>&1 | tee "$chunk_log"

  metrics_csv="$chunk_dir/metrics/before_after.csv"
  checkpoint="$chunk_dir/checkpoints/last.pt"
  if [ ! -f "$metrics_csv" ]; then
    echo "missing metrics CSV after chunk: $metrics_csv" >&2
    exit 1
  fi
  if [ ! -f "$checkpoint" ]; then
    echo "missing checkpoint after chunk: $checkpoint" >&2
    exit 1
  fi

  "$PYTHON_BIN" - "$SUMMARY" "$metrics_csv" "$checkpoint" "$completed_updates" "$next_chunk" "$CONVERGENCE_METRIC" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

summary_path, metrics_path, checkpoint, completed, chunk, metric = sys.argv[1:7]
completed = int(completed)
chunk = int(chunk)
with open(metrics_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"No metric rows in {metrics_path}")
if metric not in rows[0]:
    raise SystemExit(f"Metric {metric!r} is not present in {metrics_path}")
values = [float(row[metric]) for row in rows]
accepted = sum(int(float(row.get("accepted", 0))) for row in rows)
rejected = len(rows) - accepted
start_update = completed + 1
end_update = completed + len(rows)
line = {
    "chunk": chunk,
    "start_update": start_update,
    "end_update": end_update,
    "updates": len(rows),
    "accepted": accepted,
    "rejected": rejected,
    "metric_first": values[0],
    "metric_last": values[-1],
    "metric_min": min(values),
    "metric_mean": statistics.fmean(values),
    "checkpoint": checkpoint,
    "metrics_csv": metrics_path,
}
with open(summary_path, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(line))
    writer.writerow(line)
print(
    f"chunk {chunk}: updates={len(rows)} accepted={accepted} rejected={rejected} "
    f"{metric}: first={values[0]:.8g} last={values[-1]:.8g} min={min(values):.8g}"
)
PY

  printf "%s\n" "$checkpoint" > "$LATEST"
  init_from="$checkpoint"
  completed_updates=$((completed_updates + updates_this_chunk))

  converged=$("$PYTHON_BIN" - "$SUMMARY" "$CONVERGENCE_WINDOW" "$CONVERGENCE_MIN_IMPROVEMENT" "$MIN_UPDATES_BEFORE_CONVERGENCE" <<'PY'
import csv
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
window = int(sys.argv[2])
min_improvement = float(sys.argv[3])
min_updates = int(sys.argv[4])
with summary_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) < window:
    print("no")
    raise SystemExit
if int(rows[-1]["end_update"]) < min_updates:
    print("no")
    raise SystemExit
recent = rows[-window:]
start = float(recent[0]["metric_last"])
end = float(recent[-1]["metric_last"])
improvement = start - end
print("yes" if improvement >= 0.0 and improvement < min_improvement else "no")
PY
)

  if [ "$converged" = "yes" ]; then
    echo "[$(date -Is)] convergence reached after $completed_updates updates"
    break
  fi

  next_chunk=$((next_chunk + 1))
done

echo "[$(date -Is)] finished. Summary: $SUMMARY"
echo "Latest checkpoint: $(cat "$LATEST")"
