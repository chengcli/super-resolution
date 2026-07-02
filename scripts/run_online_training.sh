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
        "install with: pip install -e . && pip install -e ../ --no-deps"
    )
EOF

for path in $(python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('init_from') or '')
print(cfg.get('data_root') or '')
"); do
  [ -e "$path" ] || { echo "missing: $path (see README: distilled checkpoint / teacher cases)" >&2; exit 1; }
done

exec topora-online-run --config "$CONFIG"
