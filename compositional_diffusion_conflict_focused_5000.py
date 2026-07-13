"""Paper-informed compositional-sampler comparison for the 120-trial model.

The script evaluates eight sampler variants at 5,000 subjects without
retraining the network. By default it reuses the summaries from the completed
10,000-subject sensitivity run and slices each dataset to its first 5,000
subjects. Results are resumable and are written to a separate directory. The
variants focus on the paper's endpoint-damping path, with one paper-optimized
schedule, one constant-bridge comparator, and one mini-batch diagnostic.

Usage:
    conda run --no-capture-output -n tf-gpu python -u \
        compositional_diffusion_conflict_focused_5000.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import compositional_diffusion_conflict_sensitivity as sensitivity

import numpy as np


ROOT = Path(__file__).resolve().parent
N_SUBJECTS = 5_000
CLIP_LIMIT = 3.0
DEFAULT_OUTPUT = ROOT / "results" / "compositional_diffusion_conflict_120trials_focused_5000_paper_v5"
DEFAULT_SOURCE_CACHE = (
    ROOT
    / "results"
    / "compositional_diffusion_conflict_120trials_sensitivity"
    / "ntest20_nsamples100_seed20260710"
    / "summary_cache.npz"
)


def build_variants() -> OrderedDict[str, tuple[str, sensitivity.Experiment]]:
    """Return the eight planned configurations in plotting order."""
    experiment = sensitivity.Experiment
    return OrderedDict(
        (
            (
                "endpoint_d1_1_m010",
                ("d=(1,1), M=10", experiment(N_SUBJECTS, 1.0, 1.0, 10, CLIP_LIMIT)),
            ),
            (
                "endpoint_d1_01_m010",
                ("d=(1,.1), M=10", experiment(N_SUBJECTS, 1.0, 0.1, 10, CLIP_LIMIT)),
            ),
            (
                "endpoint_d1_003_m010",
                ("d=(1,.03), M=10", experiment(N_SUBJECTS, 1.0, 0.03, 10, CLIP_LIMIT)),
            ),
            (
                "endpoint_d1_001_m010",
                ("d=(1,.01), M=10", experiment(N_SUBJECTS, 1.0, 0.01, 10, CLIP_LIMIT)),
            ),
            (
                "endpoint_d1_0003_m010",
                ("d=(1,.003), M=10", experiment(N_SUBJECTS, 1.0, 0.003, 10, CLIP_LIMIT)),
            ),
            (
                "paper_094_0005_m010",
                ("d=(.94,.005), M=10", experiment(N_SUBJECTS, 0.94, 0.005, 10, CLIP_LIMIT)),
            ),
            (
                "constant_003_m010",
                ("d=(.03,.03), M=10", experiment(N_SUBJECTS, 0.03, 0.03, 10, CLIP_LIMIT)),
            ),
            (
                "endpoint_d1_003_m100",
                ("d=(1,.03), M=100", experiment(N_SUBJECTS, 1.0, 0.03, 100, CLIP_LIMIT)),
            ),
        )
    )


VARIANT_GROUPS = OrderedDict(
    (
        (
            "d1_path",
            (
                "endpoint_d1_1_m010",
                "endpoint_d1_01_m010",
                "endpoint_d1_003_m010",
                "endpoint_d1_001_m010",
                "endpoint_d1_0003_m010",
            ),
        ),
        (
            "endpoint_damping",
            (
                "endpoint_d1_003_m010",
                "paper_094_0005_m010",
                "constant_003_m010",
            ),
        ),
        (
            "mini_batch",
            (
                "endpoint_d1_003_m010",
                "endpoint_d1_003_m100",
            ),
        ),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=sensitivity.DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument(
        "--no-source-cache",
        action="store_true",
        help="Do not reuse the completed 10,000-subject summary cache.",
    )
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--summary-batch-size", type=int, default=64)
    parser.add_argument("--dataset-batch-size", type=int, default=4)
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=25,
        help="Posterior draws per GPU call; reduce this if M=100 exhausts VRAM.",
    )
    parser.add_argument("--workers", type=int, default=sensitivity.conflict.PARALLEL_WORKERS)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-sampling", action="store_true")
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU execution for smoke tests; use the GPU for the full comparison.",
    )
    args = parser.parse_args()

    positive_names = (
        "n_test",
        "n_samples",
        "summary_batch_size",
        "dataset_batch_size",
        "sample_chunk_size",
        "workers",
    )
    invalid = [name for name in positive_names if getattr(args, name) < 1]
    if invalid:
        parser.error(f"These arguments must be positive: {', '.join(invalid)}")

    args.model = args.model.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.source_cache = args.source_cache.expanduser().resolve()
    args.composition_sizes = (N_SUBJECTS,)
    if args.force_cache:
        args.force_sampling = True
    return args


def source_cache_mismatches(
    metadata: dict,
    summaries: np.ndarray,
    args: argparse.Namespace,
    model_sha256: str,
) -> list[str]:
    mismatches = []
    expected = {
        "model_sha256": model_sha256,
        "n_local_subjects": sensitivity.conflict.N_LOCAL_SUBJECTS,
        "n_trials": sensitivity.conflict.N_TRIALS,
        "seed": args.seed,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append(f"{key}={metadata.get(key)!r}, expected {value!r}")

    if int(metadata.get("n_test", 0)) < args.n_test:
        mismatches.append(f"n_test={metadata.get('n_test')!r}, need at least {args.n_test}")
    if int(metadata.get("max_subjects", 0)) < N_SUBJECTS:
        mismatches.append(
            f"max_subjects={metadata.get('max_subjects')!r}, need at least {N_SUBJECTS}"
        )

    required_compositions = N_SUBJECTS // sensitivity.conflict.N_LOCAL_SUBJECTS
    if summaries.ndim < 3:
        mismatches.append(f"summaries have invalid shape {summaries.shape}")
    else:
        if summaries.shape[0] < args.n_test:
            mismatches.append(
                f"summary datasets={summaries.shape[0]}, need at least {args.n_test}"
            )
        if summaries.shape[1] < required_compositions:
            mismatches.append(
                f"summary compositions={summaries.shape[1]}, need at least {required_compositions}"
            )
    return mismatches


def load_source_cache(
    args: argparse.Namespace,
    model_sha256: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict] | None:
    if args.no_source_cache or args.force_cache or not args.source_cache.exists():
        return None

    with np.load(args.source_cache) as cached:
        metadata = json.loads(str(cached["metadata"].item()))
        summaries = cached["summaries"]
        mismatches = source_cache_mismatches(metadata, summaries, args, model_sha256)
        if mismatches:
            print(
                f"[cache] cannot reuse {args.source_cache}: " + "; ".join(mismatches),
                flush=True,
            )
            return None

        n_compositions = N_SUBJECTS // sensitivity.conflict.N_LOCAL_SUBJECTS
        selected_summaries = np.asarray(
            summaries[: args.n_test, :n_compositions], dtype=np.float32
        )
        targets = {
            name: np.asarray(cached[f"target_{name}"][: args.n_test], dtype=np.float32)
            for name in sensitivity.conflict.PARAM_NAMES_GLOBAL
        }

    print(
        f"[cache] reused {selected_summaries.shape} summaries from {args.source_cache}",
        flush=True,
    )
    return selected_summaries, targets, metadata


def save_scorecard(
    path: Path,
    variants: OrderedDict[str, tuple[str, sensitivity.Experiment]],
    metrics: dict[str, dict],
) -> None:
    aggregate_names = (
        "normalized_rmse",
        "normalized_abs_bias",
        "coverage_95",
        "median_contraction",
        "active_clip_rate",
        "outside_3sd_rate",
        "median_posterior_sd_standardized",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "variant",
                "label",
                "n_subjects",
                "bridge_d0",
                "bridge_d1",
                "mini_batch_size",
                "clip_limit",
                *aggregate_names,
            )
        )
        for key, (label, experiment) in variants.items():
            aggregate = metrics[key]["aggregate"]
            writer.writerow(
                (
                    key,
                    label,
                    experiment.n_subjects,
                    experiment.bridge_d0,
                    experiment.bridge_d1,
                    experiment.mini_batch_size,
                    experiment.clip_limit,
                    *(aggregate[name] for name in aggregate_names),
                )
            )
    print(f"saved {path}", flush=True)


def plot_aggregate_comparison(
    title: str,
    keys: tuple[str, ...] | list[str],
    variants: OrderedDict[str, tuple[str, sensitivity.Experiment]],
    metrics: dict[str, dict],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    metric_specs = (
        ("normalized_rmse", "Mean normalized RMSE", None),
        ("normalized_abs_bias", "Mean normalized absolute bias", None),
        ("coverage_95", "Mean 95% coverage", 0.95),
        ("active_clip_rate", "Mean active-boundary rate", 0.0),
        ("median_posterior_sd_standardized", "Median posterior SD / prior SD", None),
        ("median_contraction", "Median contraction", None),
    )
    labels = [variants[key][0] for key in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    for ax, (metric_name, ylabel, reference) in zip(axes.flat, metric_specs):
        values = [metrics[key]["aggregate"][metric_name] for key in keys]
        ax.plot(x, values, marker="o", color="#1f4e79", linewidth=1.8)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        if reference is not None:
            ax.axhline(reference, color="#b22222", linestyle="--", linewidth=1)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def plot_parameter_comparison(
    variants: OrderedDict[str, tuple[str, sensitivity.Experiment]],
    metrics: dict[str, dict],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    keys = tuple(variants)
    labels = [variants[key][0] for key in keys]
    metric_specs = (
        ("correlation", "Correlation", "coolwarm", (-1.0, 1.0)),
        ("normalized_rmse", "Normalized RMSE", "magma", None),
        ("normalized_abs_bias", "Normalized absolute bias", "magma", None),
        ("coverage_95", "95% coverage", "viridis", (0.0, 1.0)),
        ("active_clip_rate", "Active-boundary rate", "magma", (0.0, 1.0)),
        (
            "median_posterior_sd_standardized",
            "Median posterior SD / prior SD",
            "viridis",
            None,
        ),
    )
    fig, axes = plt.subplots(6, 1, figsize=(15, 22), constrained_layout=True)
    for ax, (metric_name, title, cmap, limits) in zip(axes, metric_specs):
        matrix = np.array(
            [
                [
                    metrics[key]["per_parameter"][name][metric_name]
                    for key in keys
                ]
                for name in sensitivity.conflict.PARAM_NAMES_GLOBAL
            ]
        )
        kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, **kwargs)
        ax.set_yticks(
            np.arange(len(sensitivity.conflict.PARAM_NAMES_GLOBAL)),
            sensitivity.conflict.PARAM_NAMES_GLOBAL,
        )
        ax.set_xticks(np.arange(len(keys)), labels, rotation=25, ha="right")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle(f"Focused parameter diagnostics at {N_SUBJECTS:,} subjects")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def run() -> None:
    args = parse_args()
    variants = build_variants()
    run_dir = args.output_dir / f"ntest{args.n_test}_nsamples{args.n_samples}_seed{args.seed}"
    posterior_dir = run_dir / "posteriors"
    metric_dir = run_dir / "metrics"
    recovery_dir = run_dir / "figures" / "recovery"
    comparison_dir = run_dir / "figures" / "comparison"

    print(f"Focused 5,000-subject plan: {len(variants)} configurations", flush=True)
    for index, (key, (label, _)) in enumerate(variants.items(), start=1):
        print(f"  {index}. {key}: {label}", flush=True)
    print(f"  test datasets: {args.n_test}; posterior draws: {args.n_samples}", flush=True)
    print(f"  source cache: {args.source_cache}", flush=True)
    if args.plan_only:
        return

    if not args.model.exists():
        raise FileNotFoundError(f"Trained approximator not found: {args.model}")

    bf, keras = sensitivity.load_runtime(args.allow_cpu)
    model_sha256 = sensitivity.file_sha256(args.model)
    print(f"[model] loading {args.model}", flush=True)
    approximator = keras.saving.load_model(args.model)

    source = load_source_cache(args, model_sha256)
    if source is None:
        summaries, targets = sensitivity.load_or_create_summary_cache(
            approximator,
            args,
            run_dir,
            model_sha256,
        )
        source_cache_used = None
        source_metadata = None
    else:
        summaries, targets, source_metadata = source
        source_cache_used = str(args.source_cache)

    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(args.model),
        "model_sha256": model_sha256,
        "n_subjects": N_SUBJECTS,
        "n_test": args.n_test,
        "n_samples": args.n_samples,
        "n_trials": sensitivity.conflict.N_TRIALS,
        "n_local_subjects": sensitivity.conflict.N_LOCAL_SUBJECTS,
        "summary_batch_size": args.summary_batch_size,
        "dataset_batch_size": args.dataset_batch_size,
        "sample_chunk_size": args.sample_chunk_size,
        "seed": args.seed,
        "source_cache": source_cache_used,
        "source_cache_metadata": source_metadata,
        "variants": {
            key: {"label": label, "experiment": asdict(experiment)}
            for key, (label, experiment) in variants.items()
        },
    }
    sensitivity.save_json(run_dir / "run_config.json", run_config)

    all_metrics: dict[str, dict] = {}
    for index, (key, (label, experiment)) in enumerate(variants.items(), start=1):
        print(f"\n=== configuration {index}/{len(variants)}: {label} ===", flush=True)
        result_path = posterior_dir / f"{experiment.slug}.npz"
        partial_path = posterior_dir / f"{experiment.slug}.partial.npz"

        if result_path.exists() and not args.force_sampling:
            print(f"[{key}] loading completed posterior", flush=True)
            posterior, result_targets = sensitivity.load_result(result_path)
        else:
            if args.force_sampling and partial_path.exists():
                partial_path.unlink()
            started = time.monotonic()
            posterior = sensitivity.sample_experiment(
                approximator,
                summaries,
                experiment,
                args,
                partial_path,
            )
            result_targets = targets
            sensitivity.save_result(result_path, posterior, result_targets, experiment)
            partial_path.unlink(missing_ok=True)
            print(
                f"[{key}] completed in {time.monotonic() - started:.1f}s; "
                f"saved {result_path}",
                flush=True,
            )

        metrics = sensitivity.compute_metrics(posterior, result_targets, experiment)
        metrics["variant"] = key
        metrics["label"] = label
        all_metrics[key] = metrics
        sensitivity.save_json(metric_dir / f"{key}.json", metrics)
        if not args.no_recovery:
            sensitivity.save_recovery_figure(
                bf,
                posterior,
                result_targets,
                recovery_dir / f"{key}.png",
            )
        del posterior
        gc.collect()

    sensitivity.save_json(run_dir / "all_metrics.json", all_metrics)
    save_scorecard(run_dir / "scorecard.csv", variants, all_metrics)
    all_keys = tuple(variants)
    plot_aggregate_comparison(
        f"Eight sampler variants at {N_SUBJECTS:,} subjects",
        all_keys,
        variants,
        all_metrics,
        comparison_dir / "all_variants_aggregate.png",
    )
    plot_parameter_comparison(
        variants,
        all_metrics,
        comparison_dir / "all_variants_parameters.png",
    )
    for group_name, group_keys in VARIANT_GROUPS.items():
        plot_aggregate_comparison(
            f"{group_name.replace('_', ' ').title()} at {N_SUBJECTS:,} subjects",
            group_keys,
            variants,
            all_metrics,
            comparison_dir / f"{group_name}_aggregate.png",
        )

    print(f"\nComplete. Focused results saved in {run_dir}", flush=True)


if __name__ == "__main__":
    run()
