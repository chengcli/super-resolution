# super-resolution

Python tooling for observe-only downscaling experiments with Snapy dynamic
models.

The initial workflow runs a low-resolution Snapy shallow-water model beside a
high-resolution truth model whose horizontal resolution is doubled. At each
accepted step, a predictor receives the low-resolution state and returns a
high-resolution estimate. The default predictor is a deterministic bilinear
upscaler; future ML predictors can implement the same `HighResPredictor`
interface.

```bash
super-resolution-run /path/to/w92.yaml --steps 10 --output-dir ./pairs
```

The v1 runner targets Snapy's cubed-sphere shallow-water W92-style Python
driver path. Predictions are never fed back into either model.

## TopoRA online-training sidecar

`src/super_resolution/topora_online.py` trains the TopoRA wind downscaler
beside a live snapy run. TopoRA starts from FuXi-distilled weights
(`init_from`); from there, all online adaptation is driven purely by the live
snapy signal, with no FuXi teacher data involved. Each accepted snapy step
becomes a live coarse-wind sample (interior u/v/w pooled to a 9x9 column over
a procedurally generated 30 m terrain tile) and triggers one guarded online
update: a weak coarse-consistency loss on the live sample. Updates that blow
up speeds, produce non-finite values, or worsen tile seams/coarse consistency
are rejected and rolled back (model and optimizer state).

Everything needed ships with this repo:

- `src/topo_ra/` — vendored TopoRA package (model, online training, data
  utilities);
- `weights/topora_distill_best.pt` (~1 MB) — distilled weights that online
  training starts from (`init_from` in the config); do not re-distill first.

### Run online training

```bash
pip install -e .   # in an environment with snapy and paddle
```

Then, from this directory:

```bash
topora-online-run --config configs/topora_online_w92_tiny.yaml
# or
scripts/run_online_training.sh
```

This starts the tiny single-process W92 shallow-water case
(`configs/snapy_w92_tiny.yaml`) and applies `num_updates` guarded updates, one
per accepted snapy step. Device selection is automatic: CUDA if available,
then Apple MPS, then CPU; set `device: cpu` (or `cuda:1`, ...) in the config
to override. The snapy meshes themselves stay on CPU (`snapy.device`).

Outputs land in `runs/topora_online_w92_tiny/`:

- `checkpoints/online_update_NNNN.pt` — model after each accepted update
- `checkpoints/last.pt` — final weights
- `metrics/before_after.csv` — per-update live metrics plus the accept/reject
  decision. A healthy run shows `live_*_coarse_consistency_speed` decreasing.

### Resumable `/data00` training

For longer CUDA runs, use the chunked driver:

```bash
OUTPUT_ROOT=/data00/topora_online_w92_tiny_2gpu_1000 \
MAX_UPDATES=1000 \
CHUNK_UPDATES=100 \
scripts/run_online_training_until_converged.sh configs/topora_online_data00_2gpu.yaml
```

The driver writes each chunk under `$OUTPUT_ROOT/chunk_NNNNNN/`, records the
latest resumable checkpoint in `$OUTPUT_ROOT/latest_checkpoint.txt`, and
appends chunk-level metrics to `$OUTPUT_ROOT/summary.csv`. Re-running the same
command resumes from the latest checkpoint and continues until `MAX_UPDATES` or
the convergence gate stops the run.

The 1000-update validation run on July 2, 2026 wrote to
`/data00/topora_online_w92_tiny_2gpu_1000`. It logged 1000 updates with 805
accepted and 195 rejected guarded updates. The tracked loss proxy
`live_after_coarse_consistency_speed` improved from 1.4164 on the first update
to a best value of 0.01462 in chunk 9, then finished at 0.03260 after the final
chunk. `nan_count`, `inf_count`, and `nonfinite_count` stayed at 0.

### Adapting to a real case

Point `snapy.config` at your snapy YAML and adjust in
`configs/topora_online_w92_tiny.yaml`:

- `init_from` — FuXi-distilled starting checkpoint (relative to the working
  directory).
- `snapy.velocity_scale` — bring the snapy wind magnitudes into the ~10 m/s
  regime TopoRA was trained on (0.1 suits the W92 case's contravariant
  velocities).
- `num_updates`, `learning_rate`, `training.gradient_steps_per_update` —
  online-training budget.
- `acceptance.*` — rejection-gate tolerances.
- For multi-level snapy states (`nx1 > 1`), set `snapy.profile_heights` (m
  AGL, one per x1 level) and a `snapy.z_out` grid; the coarse column then
  drives TopoRA's height-query head.
