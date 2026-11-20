# SR base experiment

Recreates the oblivious and sign-flip base experiments from `../paper/main.tex`
using the supplied `qrk.py` and `noisy_methods_optimized.py` unchanged.
The requested run has 5 trials per model, 6,000 updates per method, and seed 42.
This is a new reproducible run, not an exact replay of the original paper's seeds.

## Run

From `fall26-demo/`, with Python 3.9 or newer:

```sh
python3 -m venv code/.venv
code/.venv/bin/python -m pip install -r code/requirements.txt
code/.venv/bin/python -m code.base_experiment
```

The default output directory is `code/base_experiment/`. A numerical run requires
an empty output directory to prevent mixing results. To run another configuration:

```sh
code/.venv/bin/python -m code.base_experiment --trials 5 --iterations 6000 --seed 42 --output code/another_run
```

Replot saved results without rerunning any methods:

```sh
code/.venv/bin/python -m code.base_experiment --plot-only
```

Use `--no-plots` for a numerical-only run. Run validation with:

```sh
code/.venv/bin/python -m unittest code.test_base_experiment -v
```

The runner defaults BLAS thread limits to 1 before importing NumPy. Explicit
existing environment settings take precedence. This reduces overhead for these
small matrix-vector products. Numerical roundoff may differ between BLAS builds.

## Decisions and TeX sources

Line numbers refer to the manuscript at implementation time, not PDF pages.

| Decision | Setting | `main.tex` lines or override |
| --- | --- | --- |
| Scope | Oblivious and sign-flip only; no singular-vector model | User override of 1296-1313 |
| Trial count | 5 per model | User override of 1262 |
| Duration | 6,000 updates, adaptive stopping disabled | User override; fixed durations allowed at 283-285 |
| Matrix | 2,000 by 100 independent standard Gaussian entries, normalized by row | 1276-1277 |
| Target | Independent standard Gaussian solution; clean RHS computed after normalization | 1278 |
| Corruption | 600 rows, fraction 0.3 | 1278-1279 |
| Oblivious | Uniform support without replacement; independent uniform corruption in [2, 3] | 1285-1286, 1309-1310 |
| Sign-flip | Uniform support without replacement; rho = -2 times clean RHS on support | 1287-1288 |
| Measurement noise | Independent uniform [-0.02, 0.02] on each query | 1280 |
| Persistent component | Corruption fixed across queries; no fixed dense noise added | 195-205 |
| Initialization | Zero solution vector | 1263-1264 |
| Fairness | Same matrix, target, and corruption for all methods within a trial | 1262-1263 |
| Filter | Full screening, q = 0.6, 1,200 eligible rows in the absence of ties | 1261, 1279, 229-241 |
| RK | Uniform single-row projection with step 1 | 211-215, 302-305 |
| ARK | Uniform replacement batch of 100 rows, step 1 | Sampling: 1258-1260; batch and step: user-selected defaults |
| QRK | Reused AQRK, batch 1, step 1; no rejected idle iterations | 1260-1261 |
| AQRK fresh | Batch 100, step 1, fresh independent noise for every sampled occurrence | 1303, 1261, 251-254, 325-327 |
| AQRK reused | Batch 100, step 1, cached selection residuals | 1303, 1261, 251-254 |
| AQRK-all | Mean of all eligible cached updates once each; no re-querying | 287-300, 1303 |
| AQRK sampling | Independent uniform replacement draws from eligible rows | 243-249, 269 |
| Error | Euclidean distance to clean solution divided by its norm | 1263 |
| Aggregation | Pointwise median, 25th and 75th percentiles over trials | 1264-1265 |
| Reference | Median relative error of corrupted least squares, without query noise | 1311-1313 |
| Separate plots | One figure for each model, six methods in each | User override of three-panel figure |
| Presentation | Linear iteration axis, logarithmic relative-error axis, fixed colors | Chosen default; unspecified in TeX |
| Seed | Root seed 42, NumPy PCG64 with SeedSequence keys | User-selected default; unspecified in TeX |
| Cross-model pairing | Same clean system for trial t in both models; independent corruption supports | Chosen default; unspecified in TeX |
| Method streams | Independent sampling, selection noise, and update noise per method/model/trial | Chosen default; query independence described at 442-458 |
| Indexing | Store pre-update errors at k = 0 through 5999; x_6000 is not in the history | Existing solver convention; unspecified in TeX |
| Solver changes | None; explicitly select uniform noise | User decision |

The supplied filter uses NumPy's interpolated quantile threshold rather than an
exact-size order-statistic selection with randomized tie breaking (229-241).
For distinct residual magnitudes at m = 2,000 and q = 0.6, both rules select the
same 1,200 rows. Tie behavior differs; the methods remain unchanged as requested.
The supplied RK/ARK add symmetric noise with the opposite residual sign from the
paper's oracle; symmetry preserves the specified noise distribution.

## Saved artifacts

- `metadata.json`: configuration, completion status, software versions, solver
  source hashes, thread settings, and random-stream keys. All keys are zero-based.
- `system_tt.npz`: `A`, `x_true`, and `b_clean`, shared across models for trial tt.
- `{model}_trial_tt.npz`: corruption `support`, `rho`, `b_corrupted`, least-squares
  solution/error, method names, and relative `errors` of shape `(6, iterations)`.
- `{model}_results.npz`: relative `errors` of shape `(trials, 6, iterations)`,
  `ls_errors`, iteration indices, method names, and `median`, `q25`, `q75` arrays
  of shape `(6, iterations)`, plus scalar `ls_median`.
- `base_oblivious.pdf` / `.png` and `base_sign_flip.pdf` / `.png`: the two figures.

Method order is RK, ARK, QRK, AQRK fresh, AQRK reused, AQRK-all. For any fixed
method, the default history slice has shape `(5, 6000)`.

Stream keys under root seed 42:

- Clean system: `(trial, 0)`.
- Corruption: `(trial, 1, model_index)`.
- Method randomness: `(trial, 2, model_index, method_index, stream_index)`.

Models are ordered oblivious, sign-flip. Streams are ordered row sampling,
selection noise, update noise. Unused streams do not affect other streams.
Raw query-noise arrays are regenerated from these keys, not stored. The shared
filtered solver also draws final diagnostic noise after its updates; it does not
alter the stored error histories.

## Validation

The runner checks unit row norms, clean RHS consistency, support size, corruption
formulas, finite histories, and initial relative error 1. Tests check deterministic
reruns, independent trial generation, saved history dimensions, least-squares
normal-equation residuals, aggregation along the trial axis, and overwrite refusal.
PDF contents are not inspected without an explicit request.

## Completed default run

Executed with Python 3.9.6, NumPy 1.23.5, and Matplotlib 3.9.4 in an isolated
virtual environment. `base_experiment/environment.txt` records the full installed
package versions; with Python 3.9, it can be used as a pinned requirements file.
Three pipeline unit tests passed with warnings treated as errors. The full saved
results passed all numerical checks recorded in `base_experiment/validation.log`.

Median relative errors at the last stored iterate, k = 5999:

| Method | Oblivious | Sign-flip |
| --- | --- | --- |
| RK | 1.41880 | 1.12075 |
| ARK | 0.370558 | 0.621657 |
| QRK | 0.0138899 | 0.0129362 |
| AQRK fresh | 0.000955115 | 0.000921214 |
| AQRK reused | 0.00244522 | 0.00258286 |
| AQRK-all | 0.00174435 | 0.00215282 |
| Corrupted least-squares reference | 0.340703 | 0.630450 |

`run.log` records all 60 completed method runs and an initial plotting failure:
a Matplotlib dependency deprecation became an error under strict warning checks.
After installing the compatible cached pyparsing 3.2.5 package, `--plot-only`
succeeded with warnings treated as errors, as recorded in `plot.log`. Numerical
experiments were not rerun. Both PNG exports were checked for readability; PDF
contents were not inspected.
