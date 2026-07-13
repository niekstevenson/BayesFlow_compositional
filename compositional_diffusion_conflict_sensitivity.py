"""Sensitivity analysis for the trained 120-trial compositional conflict model.

This script never retrains a network. It loads the existing global approximator,
simulates one paired test set, caches the SetTransformer summaries, and compares
compositional sampler settings. The default is a tractable one-factor-at-a-time
design; pass ``--full-factorial`` only when the much larger run is intentional.

Usage:
    conda run --no-capture-output -n tf-gpu python -u \
        compositional_diffusion_conflict_sensitivity.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np

import compositional_diffusion_conflict as conflict


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / conflict.SAVE_DIR / "global_approximator.keras"
DEFAULT_OUTPUT = ROOT / "results" / "compositional_diffusion_conflict_120trials_sensitivity"
DEFAULT_COMPOSITION_SIZES = (10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)
BRIDGE_SETTINGS = ((1.0, 1.0), (1.0, 0.1), (0.1, 0.1), (0.03, 0.03))
MINI_BATCH_SETTINGS = (10, 50, 200)
CLIP_SETTINGS = (3.0, 5.0)
SWEEP_NAMES = ("composition", "bridge", "mini_batch", "clip")


@dataclass(frozen=True)
class Experiment:
    n_subjects: int
    bridge_d0: float
    bridge_d1: float
    mini_batch_size: int
    clip_limit: float

    @property
    def n_compositions(self) -> int:
        return self.n_subjects // conflict.N_LOCAL_SUBJECTS

    @property
    def slug(self) -> str:
        return (
            f"n{self.n_subjects:05d}"
            f"_d0{format_number(self.bridge_d0)}"
            f"_d1{format_number(self.bridge_d1)}"
            f"_m{self.mini_batch_size:03d}"
            f"_c{format_number(self.clip_limit)}"
        )


def format_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.") from error
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_sweeps(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(values) - set(SWEEP_NAMES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown sweeps: {', '.join(invalid)}")
    if not values:
        raise argparse.ArgumentTypeError("At least one sweep is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--composition-sizes",
        type=parse_int_list,
        default=DEFAULT_COMPOSITION_SIZES,
        help="Comma-separated subject counts; each must be divisible by 5.",
    )
    parser.add_argument("--sweeps", type=parse_sweeps, default=SWEEP_NAMES)
    parser.add_argument("--summary-batch-size", type=int, default=64)
    parser.add_argument("--dataset-batch-size", type=int, default=4)
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=25,
        help="Posterior draws per GPU call; lower this if a large mini-batch runs out of VRAM.",
    )
    parser.add_argument("--workers", type=int, default=conflict.PARALLEL_WORKERS)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--full-factorial",
        action="store_true",
        help="Run every size x bridge x mini-batch x clip combination (240 with default sizes).",
    )
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-sampling", action="store_true")
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for tiny smoke tests. Full sweeps should use the GPU environment.",
    )
    args = parser.parse_args()

    if args.n_test < 1 or args.n_samples < 1:
        parser.error("--n-test and --n-samples must be positive.")
    if args.summary_batch_size < 1 or args.dataset_batch_size < 1 or args.sample_chunk_size < 1:
        parser.error("All batch sizes must be positive.")
    if args.workers < 1:
        parser.error("--workers must be positive.")

    sizes = tuple(sorted(set(args.composition_sizes)))
    invalid_sizes = [
        size
        for size in sizes
        if size < 2 * conflict.N_LOCAL_SUBJECTS or size % conflict.N_LOCAL_SUBJECTS != 0
    ]
    if invalid_sizes:
        parser.error(
            "Composition sizes must be at least 10 and divisible by "
            f"{conflict.N_LOCAL_SUBJECTS}: {invalid_sizes}"
        )
    args.composition_sizes = sizes
    args.model = args.model.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.force_cache:
        args.force_sampling = True
    return args


def build_sweeps(args: argparse.Namespace) -> OrderedDict[str, list[Experiment]]:
    max_subjects = max(args.composition_sizes)
    baseline = Experiment(max_subjects, 1.0, 1.0, 10, 3.0)
    sweeps: OrderedDict[str, list[Experiment]] = OrderedDict()

    if "composition" in args.sweeps:
        sweeps["composition"] = [
            Experiment(size, baseline.bridge_d0, baseline.bridge_d1, baseline.mini_batch_size, baseline.clip_limit)
            for size in args.composition_sizes
        ]
    if "bridge" in args.sweeps:
        sweeps["bridge"] = [
            Experiment(max_subjects, d0, d1, baseline.mini_batch_size, baseline.clip_limit)
            for d0, d1 in BRIDGE_SETTINGS
        ]
    if "mini_batch" in args.sweeps:
        sweeps["mini_batch"] = [
            Experiment(max_subjects, baseline.bridge_d0, baseline.bridge_d1, size, baseline.clip_limit)
            for size in MINI_BATCH_SETTINGS
        ]
    if "clip" in args.sweeps:
        sweeps["clip"] = [
            Experiment(max_subjects, baseline.bridge_d0, baseline.bridge_d1, baseline.mini_batch_size, limit)
            for limit in CLIP_SETTINGS
        ]

    if args.full_factorial:
        sweeps["factorial"] = [
            Experiment(size, d0, d1, mini_batch_size, clip_limit)
            for size in args.composition_sizes
            for d0, d1 in BRIDGE_SETTINGS
            for mini_batch_size in MINI_BATCH_SETTINGS
            for clip_limit in CLIP_SETTINGS
        ]
    return sweeps


def unique_experiments(sweeps: OrderedDict[str, list[Experiment]]) -> list[Experiment]:
    seen = set()
    experiments = []
    for sweep in sweeps.values():
        for experiment in sweep:
            if experiment not in seen:
                seen.add(experiment)
                experiments.append(experiment)
    return experiments


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runtime(allow_cpu: bool):
    if allow_cpu:
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.disable(logging.INFO)
        import bayesflow as bf
        import keras
        logging.disable(logging.NOTSET)
        conflict.bf = bf
        conflict.keras = keras
        print(f"Keras backend: {keras.backend.backend()} (CPU allowed)", flush=True)
    else:
        conflict.load_runtime()

    return conflict.bf, conflict.keras


def deterministic_parallel_sample(
    n_samples: int,
    n_subjects: int,
    workers: int,
    seed: int,
) -> dict[str, np.ndarray]:
    from joblib import Parallel, delayed

    counts = conflict.split_counts(n_samples, workers * 4)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = [int(state.generate_state(1)[0]) for state in seed_sequence.spawn(len(counts))]
    batches = Parallel(n_jobs=workers)(
        delayed(conflict.sample_serial)(
            count,
            n_subjects=n_subjects,
            n_trials=conflict.N_TRIALS,
            seed=child_seed,
        )
        for count, child_seed in zip(counts, child_seeds)
        if count > 0
    )
    return conflict.merge_batches(batches)


def cache_metadata(args: argparse.Namespace, model_sha256: str) -> dict:
    return {
        "model": str(args.model),
        "model_sha256": model_sha256,
        "n_test": args.n_test,
        "max_subjects": max(args.composition_sizes),
        "n_local_subjects": conflict.N_LOCAL_SUBJECTS,
        "n_trials": conflict.N_TRIALS,
        "summary_batch_size": args.summary_batch_size,
        "seed": args.seed,
    }


def save_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_or_create_summary_cache(
    approximator,
    args: argparse.Namespace,
    run_dir: Path,
    model_sha256: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cache_path = run_dir / "summary_cache.npz"
    expected_metadata = cache_metadata(args, model_sha256)

    if cache_path.exists() and not args.force_cache:
        with np.load(cache_path) as cached:
            observed_metadata = json.loads(str(cached["metadata"].item()))
            if observed_metadata != expected_metadata:
                raise RuntimeError(
                    f"Cached summaries at {cache_path} do not match this run. "
                    "Use a different output directory or pass --force-cache."
                )
            summaries = cached["summaries"]
            targets = {name: cached[f"target_{name}"] for name in conflict.PARAM_NAMES_GLOBAL}
        print(f"[cache] loaded summaries {summaries.shape} from {cache_path}", flush=True)
        return summaries, targets

    max_subjects = max(args.composition_sizes)
    n_compositions = max_subjects // conflict.N_LOCAL_SUBJECTS
    print(
        f"[cache] simulating {args.n_test} paired datasets with {max_subjects} subjects each",
        flush=True,
    )
    started = time.monotonic()
    test_data = deterministic_parallel_sample(
        n_samples=args.n_test,
        n_subjects=max_subjects,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"[cache] simulation completed in {time.monotonic() - started:.1f}s", flush=True)

    summary_rows = []
    for dataset_index in range(args.n_test):
        print(f"[cache] summarizing dataset {dataset_index + 1}/{args.n_test}", flush=True)
        chunked = test_data["sim_data"][dataset_index].reshape(
            n_compositions,
            conflict.N_LOCAL_SUBJECTS * conflict.N_TRIALS,
            3,
        )
        summary_rows.append(
            approximator.summarize(
                {"sim_data": chunked},
                batch_size=args.summary_batch_size,
            )
        )

    summaries = np.stack(summary_rows).astype(np.float32, copy=False)
    targets = {
        name: np.asarray(test_data[name], dtype=np.float32)
        for name in conflict.PARAM_NAMES_GLOBAL
    }
    arrays = {
        "metadata": np.array(json.dumps(expected_metadata, sort_keys=True)),
        "summaries": summaries,
    }
    arrays.update({f"target_{name}": value for name, value in targets.items()})
    save_npz_atomic(cache_path, **arrays)
    print(f"[cache] saved summaries {summaries.shape} to {cache_path}", flush=True)
    del test_data, summary_rows
    gc.collect()
    return summaries, targets


def posterior_arrays(posterior: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"estimates_{name}": np.asarray(posterior[name]) for name in conflict.PARAM_NAMES_GLOBAL}


def load_partial(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {
            name: saved[f"estimates_{name}"]
            for name in conflict.PARAM_NAMES_GLOBAL
        }


def save_partial(path: Path, posterior: dict[str, np.ndarray]) -> None:
    save_npz_atomic(path, **posterior_arrays(posterior))


def sample_experiment(
    approximator,
    summaries: np.ndarray,
    experiment: Experiment,
    args: argparse.Namespace,
    partial_path: Path,
) -> dict[str, np.ndarray]:
    n_items = experiment.n_compositions
    existing = None if args.force_sampling else load_partial(partial_path)
    posterior_batches = [] if existing is None else [existing]
    completed = 0 if existing is None else len(next(iter(existing.values())))
    if completed:
        print(f"[{experiment.slug}] resuming after {completed}/{args.n_test} datasets", flush=True)

    for dataset_start in range(completed, args.n_test, args.dataset_batch_size):
        dataset_stop = min(dataset_start + args.dataset_batch_size, args.n_test)
        batch_summaries = summaries[dataset_start:dataset_stop, :n_items]
        sample_chunks = []

        for sample_start in range(0, args.n_samples, args.sample_chunk_size):
            sample_stop = min(sample_start + args.sample_chunk_size, args.n_samples)
            seed = (args.seed + dataset_start * 100_003 + sample_start) % (2**31 - 1)
            print(
                f"[{experiment.slug}] datasets {dataset_start + 1}-{dataset_stop}/{args.n_test}, "
                f"draws {sample_start + 1}-{sample_stop}/{args.n_samples}",
                flush=True,
            )
            sample_chunks.append(
                approximator.compositional_sample(
                    num_samples=sample_stop - sample_start,
                    summary_outputs=batch_summaries,
                    compute_prior_score=conflict.prior_global_score,
                    method="two_step_adaptive",
                    steps="adaptive",
                    mini_batch_size=experiment.mini_batch_size,
                    compositional_bridge_d0=experiment.bridge_d0,
                    compositional_bridge_d1=experiment.bridge_d1,
                    clip=(-experiment.clip_limit, experiment.clip_limit),
                    batch_size=args.dataset_batch_size,
                    seed=seed,
                )
            )

        batch_posterior = {
            name: np.concatenate([chunk[name] for chunk in sample_chunks], axis=1)
            for name in conflict.PARAM_NAMES_GLOBAL
        }
        posterior_batches.append(batch_posterior)
        combined = {
            name: np.concatenate([batch[name] for batch in posterior_batches], axis=0)
            for name in conflict.PARAM_NAMES_GLOBAL
        }
        save_partial(partial_path, combined)
        del sample_chunks, batch_posterior, combined
        gc.collect()

    return {
        name: np.concatenate([batch[name] for batch in posterior_batches], axis=0)
        for name in conflict.PARAM_NAMES_GLOBAL
    }


def save_result(
    path: Path,
    posterior: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    experiment: Experiment,
) -> None:
    arrays = posterior_arrays(posterior)
    arrays.update({f"targets_{name}": targets[name] for name in conflict.PARAM_NAMES_GLOBAL})
    arrays["experiment"] = np.array(json.dumps(asdict(experiment), sort_keys=True))
    save_npz_atomic(path, **arrays)


def load_result(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path) as saved:
        posterior = {
            name: saved[f"estimates_{name}"]
            for name in conflict.PARAM_NAMES_GLOBAL
        }
        targets = {
            name: saved[f"targets_{name}"]
            for name in conflict.PARAM_NAMES_GLOBAL
        }
    return posterior, targets


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(
    posterior: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    experiment: Experiment,
) -> dict:
    per_parameter = {}
    for name in conflict.PARAM_NAMES_GLOBAL:
        draws = np.asarray(posterior[name]).squeeze(-1)
        truth = np.asarray(targets[name]).reshape(-1)
        point = np.median(draws, axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975], axis=1)
        prior_mean, prior_sd = conflict.GLOBAL_PRIOR[name]
        standardized_point = (point - prior_mean) / prior_sd
        posterior_variance = np.var(draws, axis=1, ddof=1) if draws.shape[1] > 1 else np.zeros(draws.shape[0])

        per_parameter[name] = {
            "correlation": safe_correlation(truth, point),
            "normalized_rmse": float(np.sqrt(np.mean((point - truth) ** 2)) / prior_sd),
            "normalized_abs_bias": float(abs(np.mean(point - truth)) / prior_sd),
            "coverage_95": float(np.mean((truth >= lower) & (truth <= upper))),
            "median_contraction": float(np.median(1.0 - posterior_variance / prior_sd**2)),
            "active_clip_rate": float(
                np.mean(np.abs(standardized_point) >= experiment.clip_limit - 0.05)
            ),
            "outside_3sd_rate": float(np.mean(np.abs(standardized_point) >= 2.95)),
            "median_posterior_sd_standardized": float(np.median(np.std(draws, axis=1) / prior_sd)),
        }

    aggregate_keys = (
        "normalized_rmse",
        "normalized_abs_bias",
        "coverage_95",
        "median_contraction",
        "active_clip_rate",
        "outside_3sd_rate",
        "median_posterior_sd_standardized",
    )
    aggregate = {
        key: float(np.nanmean([per_parameter[name][key] for name in conflict.PARAM_NAMES_GLOBAL]))
        for key in aggregate_keys
    }
    return {
        "experiment": asdict(experiment),
        "aggregate": aggregate,
        "per_parameter": per_parameter,
    }


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_recovery_figure(
    bf,
    posterior: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig = bf.diagnostics.recovery(
        estimates=posterior,
        targets=targets,
        variable_names=conflict.PRETTY_PARAM_NAMES_GLOBAL,
    )
    conflict.save_figure(path, fig)


def experiment_label(sweep_name: str, experiment: Experiment) -> str:
    if sweep_name == "composition":
        return f"{experiment.n_subjects:g}"
    if sweep_name == "bridge":
        return f"({experiment.bridge_d0:g}, {experiment.bridge_d1:g})"
    if sweep_name == "mini_batch":
        return f"{experiment.mini_batch_size:g}"
    if sweep_name == "clip":
        return f"[-{experiment.clip_limit:g}, {experiment.clip_limit:g}]"
    return experiment.slug


def plot_aggregate_sweep(
    sweep_name: str,
    experiments: list[Experiment],
    metrics: dict[Experiment, dict],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    metric_specs = (
        ("normalized_rmse", "Mean normalized RMSE", None),
        ("coverage_95", "Mean 95% coverage", 0.95),
        ("active_clip_rate", "Mean active-boundary hit rate", 0.0),
        ("outside_3sd_rate", "Mean outside +/-3 SD rate", 0.0),
        ("median_contraction", "Mean median contraction", None),
    )
    values = {
        key: [metrics[experiment]["aggregate"][key] for experiment in experiments]
        for key, _, _ in metric_specs
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    if sweep_name == "composition":
        x = np.array([experiment.n_subjects for experiment in experiments])
        for ax, (key, ylabel, reference) in zip(axes.flat, metric_specs):
            ax.plot(x, values[key], marker="o", color="#1f4e79")
            ax.set_xscale("log")
            ax.set_xlabel("Subjects")
            ax.set_ylabel(ylabel)
            if reference is not None:
                ax.axhline(reference, color="#b22222", linestyle="--", linewidth=1)
            ax.grid(alpha=0.25)
    else:
        x = np.arange(len(experiments))
        labels = [experiment_label(sweep_name, experiment) for experiment in experiments]
        for ax, (key, ylabel, reference) in zip(axes.flat, metric_specs):
            ax.plot(x, values[key], marker="o", color="#1f4e79")
            ax.set_xticks(x, labels, rotation=20, ha="right")
            ax.set_ylabel(ylabel)
            if reference is not None:
                ax.axhline(reference, color="#b22222", linestyle="--", linewidth=1)
            ax.grid(alpha=0.25)
    for ax in axes.flat[len(metric_specs) :]:
        ax.set_visible(False)
    fig.suptitle(f"Compositional sensitivity: {sweep_name.replace('_', ' ')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def plot_parameter_sweep(
    sweep_name: str,
    experiments: list[Experiment],
    metrics: dict[Experiment, dict],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = [experiment_label(sweep_name, experiment) for experiment in experiments]
    metric_specs = (
        ("normalized_rmse", "Normalized RMSE", "magma", None),
        ("coverage_95", "95% coverage", "viridis", (0.0, 1.0)),
        ("active_clip_rate", "Active-boundary hit rate", "magma", (0.0, 1.0)),
        ("outside_3sd_rate", "Outside +/-3 SD rate", "magma", (0.0, 1.0)),
    )
    width = max(10.0, 0.85 * len(experiments) + 5.0)
    fig, axes = plt.subplots(4, 1, figsize=(width, 15), constrained_layout=True)

    for ax, (metric_name, title, cmap, limits) in zip(axes, metric_specs):
        matrix = np.array(
            [
                [metrics[experiment]["per_parameter"][name][metric_name] for experiment in experiments]
                for name in conflict.PARAM_NAMES_GLOBAL
            ]
        )
        kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, **kwargs)
        ax.set_yticks(np.arange(len(conflict.PARAM_NAMES_GLOBAL)), conflict.PARAM_NAMES_GLOBAL)
        ax.set_xticks(np.arange(len(experiments)), labels, rotation=30, ha="right")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle(f"Parameter diagnostics: {sweep_name.replace('_', ' ')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def run() -> None:
    args = parse_args()
    sweeps = build_sweeps(args)
    experiments = unique_experiments(sweeps)
    run_dir = args.output_dir / f"ntest{args.n_test}_nsamples{args.n_samples}_seed{args.seed}"
    result_dir = run_dir / "posteriors"
    metric_dir = run_dir / "metrics"
    recovery_dir = run_dir / "figures" / "recovery"
    comparison_dir = run_dir / "figures" / "comparison"

    print("Sensitivity plan:", flush=True)
    for name, settings in sweeps.items():
        print(f"  {name}: {len(settings)} settings", flush=True)
    print(f"  unique configurations: {len(experiments)}", flush=True)
    print(f"  test datasets: {args.n_test}; posterior draws: {args.n_samples}", flush=True)
    if args.full_factorial:
        print("  WARNING: full factorial requested; this can take days.", flush=True)
    if args.plan_only:
        return

    if not args.model.exists():
        raise FileNotFoundError(f"Trained approximator not found: {args.model}")

    bf, keras = load_runtime(args.allow_cpu)
    model_sha256 = file_sha256(args.model)
    print(f"[model] loading {args.model}", flush=True)
    approximator = keras.saving.load_model(args.model)
    summaries, targets = load_or_create_summary_cache(approximator, args, run_dir, model_sha256)

    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(args.model),
        "model_sha256": model_sha256,
        "n_test": args.n_test,
        "n_samples": args.n_samples,
        "n_trials": conflict.N_TRIALS,
        "n_local_subjects": conflict.N_LOCAL_SUBJECTS,
        "composition_sizes": list(args.composition_sizes),
        "bridge_settings": [list(setting) for setting in BRIDGE_SETTINGS],
        "mini_batch_settings": list(MINI_BATCH_SETTINGS),
        "clip_settings": list(CLIP_SETTINGS),
        "sweeps": list(args.sweeps),
        "full_factorial": args.full_factorial,
        "summary_batch_size": args.summary_batch_size,
        "dataset_batch_size": args.dataset_batch_size,
        "sample_chunk_size": args.sample_chunk_size,
        "seed": args.seed,
        "experiments": [asdict(experiment) for experiment in experiments],
    }
    save_json(run_dir / "run_config.json", run_config)

    all_metrics: dict[Experiment, dict] = {}
    for index, experiment in enumerate(experiments, start=1):
        print(f"\n=== configuration {index}/{len(experiments)}: {experiment.slug} ===", flush=True)
        result_path = result_dir / f"{experiment.slug}.npz"
        partial_path = result_dir / f"{experiment.slug}.partial.npz"

        if result_path.exists() and not args.force_sampling:
            print(f"[{experiment.slug}] loading completed posterior", flush=True)
            posterior, result_targets = load_result(result_path)
        else:
            if args.force_sampling and partial_path.exists():
                partial_path.unlink()
            started = time.monotonic()
            posterior = sample_experiment(
                approximator,
                summaries,
                experiment,
                args,
                partial_path,
            )
            result_targets = targets
            save_result(result_path, posterior, result_targets, experiment)
            partial_path.unlink(missing_ok=True)
            print(
                f"[{experiment.slug}] completed in {time.monotonic() - started:.1f}s; saved {result_path}",
                flush=True,
            )

        metrics = compute_metrics(posterior, result_targets, experiment)
        all_metrics[experiment] = metrics
        save_json(metric_dir / f"{experiment.slug}.json", metrics)
        if not args.no_recovery:
            save_recovery_figure(
                bf,
                posterior,
                result_targets,
                recovery_dir / f"{experiment.slug}.png",
            )
        del posterior
        gc.collect()

    save_json(
        run_dir / "all_metrics.json",
        {experiment.slug: metrics for experiment, metrics in all_metrics.items()},
    )
    for sweep_name, sweep_experiments in sweeps.items():
        if sweep_name == "factorial":
            continue
        plot_aggregate_sweep(
            sweep_name,
            sweep_experiments,
            all_metrics,
            comparison_dir / f"{sweep_name}_aggregate.png",
        )
        plot_parameter_sweep(
            sweep_name,
            sweep_experiments,
            all_metrics,
            comparison_dir / f"{sweep_name}_parameters.png",
        )

    print(f"\nComplete. Sensitivity results saved in {run_dir}", flush=True)


if __name__ == "__main__":
    run()
