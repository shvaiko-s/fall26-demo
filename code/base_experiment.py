"""Reproduce the two selected SR base experiments using the supplied solvers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

# Small matrix-vector products run best without a multithreaded BLAS pool.
# Set before importing NumPy; no solver implementation is changed.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from .noisy_methods_optimized import filtered_kaczmarz
from .qrk import ARK, RK, quantileRK

MODELS = ("oblivious", "sign_flip")
METHODS = ("RK", "ARK", "QRK", "AQRK_fresh", "AQRK_reused", "AQRK_all")
LABELS = ("RK", "ARK (s=100)", "QRK", "AQRK (s=100, fresh)",
          "AQRK (s=100, reused)", "AQRK-all")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "base_experiment"


def seed_sequence(seed, *key):
    """Stable keys avoid changing another run's streams when loop order changes."""
    return np.random.SeedSequence(seed, spawn_key=tuple(key))


def generator(seed, *key):
    return np.random.Generator(np.random.PCG64(seed_sequence(seed, *key)))


def generate_system(seed, trial):
    # main.tex:1276-1278. Noise and corruption are added after normalization.
    rng = generator(seed, trial, 0)
    A = rng.standard_normal((2000, 100))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    x_true = rng.standard_normal(100)
    return A, x_true, A @ x_true


def generate_corruption(b_clean, seed, trial, model):
    # main.tex:1285-1288, 1309-1310. Independent supports for the two models.
    model_index = MODELS.index(model)
    rng = generator(seed, trial, 1, model_index)
    support = np.sort(rng.choice(b_clean.size, size=600, replace=False))
    rho = np.zeros_like(b_clean)
    if model == "oblivious":
        rho[support] = rng.uniform(2.0, 3.0, size=support.size)
    else:
        rho[support] = -2.0 * b_clean[support]
    return support, rho


def run_method(method, A, b, x_true, iterations, streams):
    rng, selection_rng, update_rng = streams
    common = dict(noise_distribution="uniform", selection_rng=selection_rng,
                  update_rng=update_rng)
    if method == "RK":
        errors = RK(A, b, iterations, x_true, 0.02, rng, **common)
    elif method == "ARK":
        errors = ARK(A, b, 0.02, iterations, x_true, 1.0, 100, rng, **common)
    elif method == "QRK":
        errors = quantileRK(A, b, 0.6, iterations, x_true, 0.02, rng, **common)
    else:
        settings = dict(q=0.6, alpha=1.0, inner_iterations=iterations,
                        adaptive_stopping=False, window=1, residual_tolerance=0.0)
        batch = "all" if method == "AQRK_all" else 100
        _, _, history = filtered_kaczmarz(
            A, b, np.zeros(100), batch, settings, rng, query_width=0.02,
            requery=(method == "AQRK_fresh"), soln=x_true, **common)
        errors = history["errors"]
    return errors / np.linalg.norm(x_true)


def aggregate(errors, ls_errors):
    return dict(median=np.median(errors, axis=0),
                q25=np.percentile(errors, 25, axis=0),
                q75=np.percentile(errors, 75, axis=0),
                ls_median=np.asarray(np.median(ls_errors)))


def validate_system(A, x_true, b_clean, support, rho, model):
    np.testing.assert_allclose(np.linalg.norm(A, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(A @ x_true, b_clean)
    assert support.shape == (600,) and np.unique(support).size == 600
    outside = np.ones(2000, dtype=bool)
    outside[support] = False
    assert np.all(rho[outside] == 0)
    if model == "oblivious":
        assert np.all((rho[support] >= 2) & (rho[support] <= 3))
    else:
        np.testing.assert_array_equal(rho[support], -2 * b_clean[support])


def run_experiment(output, trials=5, iterations=6000, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental mixing/overwriting of datasets from different runs.
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    methods_dir = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((methods_dir / name).read_bytes()).hexdigest()
                     for name in ("base_experiment.py", "qrk.py", "noisy_methods_optimized.py")}
    metadata = dict(
        status="running", trials=trials, iterations=iterations, root_seed=seed,
        models=list(MODELS), methods=list(METHODS), m=2000, n=100, beta=0.3,
        q=0.6, h=1200, alpha=1.0, batch_size=100, query_half_width=0.02,
        noise_distribution="uniform", bit_generator="PCG64",
        seed_keys={"system": "(trial, 0)", "corruption": "(trial, 1, model_index)",
                   "method": "(trial, 2, model_index, method_index, stream_index)",
                   "stream_indices": ["row_sampling", "selection_noise", "update_noise"]},
        history="pre-update relative error, k=0,...,iterations-1",
        cross_model_systems="shared", source_sha256=source_hashes,
        python=platform.python_version(), numpy=np.__version__,
        thread_environment={key: os.environ.get(key) for key in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    histories = {model: np.empty((trials, len(METHODS), iterations)) for model in MODELS}
    ls_errors = {model: np.empty(trials) for model in MODELS}
    for trial in range(trials):
        A, x_true, b_clean = generate_system(seed, trial)
        np.savez_compressed(output / f"system_{trial:02d}.npz",
                            A=A, x_true=x_true, b_clean=b_clean)
        for model_index, model in enumerate(MODELS):
            support, rho = generate_corruption(b_clean, seed, trial, model)
            validate_system(A, x_true, b_clean, support, rho, model)
            b = b_clean + rho  # Fixed corruption only; solvers sample query noise.
            x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
            ls_errors[model][trial] = np.linalg.norm(x_ls - x_true) / np.linalg.norm(x_true)
            for method_index, method in enumerate(METHODS):
                streams = [generator(seed, trial, 2, model_index, method_index, j)
                           for j in range(3)]
                errors = run_method(method, A, b, x_true, iterations, streams)
                assert errors.shape == (iterations,) and np.isfinite(errors).all()
                np.testing.assert_allclose(errors[0], 1.0)
                histories[model][trial, method_index] = errors
                print(f"{model}: trial {trial + 1}/{trials}, {method}, "
                      f"last recorded relative error {errors[-1]:.6g}", flush=True)
            np.savez_compressed(
                output / f"{model}_trial_{trial:02d}.npz", support=support, rho=rho,
                b_corrupted=b, x_ls=x_ls, ls_error=ls_errors[model][trial],
                errors=histories[model][trial], methods=np.asarray(METHODS))
    for model in MODELS:
        errors = histories[model]
        stats = aggregate(errors, ls_errors[model])
        np.savez_compressed(output / f"{model}_results.npz", errors=errors,
                            ls_errors=ls_errors[model], iterations=np.arange(iterations),
                            methods=np.asarray(METHODS), **stats)
    metadata["status"] = "complete"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def plot_results(output):
    # Lazy import permits numerical runs without a plotting installation.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("The numerical run has not completed")
    colors = ("#777777", "#d89019", "#b83c37", "#2367b2", "#298849", "#8355a6")
    for model in MODELS:
        with np.load(output / f"{model}_results.npz", allow_pickle=False) as data:
            fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
            for i, (label, color) in enumerate(zip(LABELS, colors)):
                ax.fill_between(data["iterations"], data["q25"][i], data["q75"][i],
                                color=color, alpha=0.15, linewidth=0)
                ax.plot(data["iterations"], data["median"][i], label=label,
                        color=color, linewidth=1.5)
            ax.axhline(float(data["ls_median"]), color="black", linestyle="--",
                       linewidth=1.2, label="Corrupted least squares (no query noise)")
            name = "Oblivious corruption, U[2, 3]" if model == "oblivious" else "Sign-flip corruption"
            ax.set(title=f"{name} ({metadata['trials']} trials)",
                   xlabel="Iteration k", ylabel="Relative solution error", yscale="log")
            ax.set_xlim(0, metadata["iterations"] - 1)
            ax.grid(True, which="both", alpha=0.18)
            ax.legend(fontsize=8, loc="best")
            for suffix in ("pdf", "png"):
                fig.savefig(output / f"base_{model}.{suffix}", dpi=200)
            plt.close(fig)
    print(f"Saved both plots to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.iterations < 1 or args.seed < 0:
        parser.error("trials and iterations must be positive; seed must be nonnegative")
    if args.plot_only and args.no_plots:
        parser.error("--plot-only and --no-plots are mutually exclusive")
    if not args.plot_only:
        run_experiment(args.output, args.trials, args.iterations, args.seed)
    if not args.no_plots:
        plot_results(args.output)


if __name__ == "__main__":
    main()
