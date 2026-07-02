#!/usr/bin/env bash
# Run TopoRA online training beside the tiny live snapy W92 case.
# Usage: scripts/run_online_training.sh [config.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/topora_online_w92_tiny.yaml}"

python - <<'EOF'
import importlib.util, sys
missing = [m for m in ("snapy", "topo_ra", "super_resolution") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(
        f"missing packages: {', '.join(missing)}\n"
        "install with: pip install -e . (in an environment with snapy/paddle)"
    )
EOF

INIT_FROM=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('init_from') or '')")
DATA_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('data_root') or '')")
if [ -n "$INIT_FROM" ] && [ ! -e "$INIT_FROM" ]; then
  echo "missing distilled weights: $INIT_FROM (see README)" >&2
  exit 1
fi
if [ -n "$DATA_ROOT" ] && [ ! -e "$DATA_ROOT" ]; then
  echo "note: $DATA_ROOT not found; teacher replay falls back to synthetic samples (see README)" >&2
fi

exec topora-online-run --config "$CONFIG"
