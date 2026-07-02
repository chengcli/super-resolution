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
