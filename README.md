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
beside a live snapy run. Each accepted snapy step becomes a live coarse-wind
sample (interior u/v/w pooled to a 9x9 column over a 30 m terrain tile) and
triggers one guarded online update: a weak coarse-consistency loss on the live
sample plus a supervised loss on FuXi teacher replay samples. Updates that
degrade replay skill, blow up speeds, produce non-finite values, or worsen
tile seams are rejected and rolled back (model and optimizer state).

### Requirements

1. This repo, plus `snapy` and `paddle` installed (see `pyproject.toml`).
2. The `topo-ra` **wheel** (~54 KB, distributed privately — the TopoRA source
   is not on GitHub):

   ```bash
   pip install -e .                          # this repo
   pip install topo_ra-*.whl --no-deps      # deps already covered by snapy env
   ```

3. The **distilled weights** ship with this repo:
   `weights/topora_distill_best.pt` (~1 MB). Online training starts from them
   (`init_from` in the config); do not re-distill first.
4. Optional but recommended: a few FuXi teacher case folders (~20 MB each,
   distributed privately) extracted to `../data/fuxi/dataset/case_XXXXXX/`
   (`inputs.npz` + `outputs.npz`). They anchor the guarded updates with real
   teacher truth. Without them, the replay anchor automatically falls back to
   synthetic samples — the run works, but the guard on distilled skill is
   weaker. The sidecar prints which replay source it is using.

### Run online training

From this directory:

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
- `metrics/before_after.csv` — per-update live and replay metrics plus the
  accept/reject decision. A healthy run shows
  `live_*_coarse_consistency_speed` decreasing while
  `replay_*_speed_RMSE` stays flat.

### Adapting to a real case

Point `snapy.config` at your snapy YAML and adjust in
`configs/topora_online_w92_tiny.yaml`:

- `init_from` / `data_root` — checkpoint and teacher-case paths (relative to
  the working directory).
- `snapy.velocity_scale` — bring the snapy wind magnitudes into the ~10 m/s
  regime TopoRA was trained on (0.1 suits the W92 case's contravariant
  velocities).
- `num_updates`, `learning_rate`, `training.gradient_steps_per_update` —
  online-training budget.
- `acceptance.*` — rejection-gate tolerances.
- For multi-level snapy states (`nx1 > 1`), set `snapy.profile_heights` (m
  AGL, one per x1 level) and a `snapy.z_out` grid; the coarse column then
  drives TopoRA's height-query head.
