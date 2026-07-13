"""Corrected K=50 conflict-DDM training and compositional benchmark.

The model is trained in the exact standardized prior coordinates required by
``compute_prior_score="standard_normal"``. The simulator uses the requested
100-second deadline and represents a timeout as choice 0. A fixed validation
set selects the saved model.

The default benchmark first checks ordinary 50-subject recovery, then compares
all 100 likelihood factors with the published 10% mini-batch estimator for a
5,000-subject posterior.  Both use the untempered bridge d(t)=1 and the adaptive
two-step SDE solver from the local BayesFlow checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOCAL_BAYESFLOW = ROOT / "bayesflow"
while str(LOCAL_BAYESFLOW) in sys.path:
    sys.path.remove(str(LOCAL_BAYESFLOW))
sys.path.insert(0, str(LOCAL_BAYESFLOW))

import conflict_ddm_censored as censored  # noqa: E402
import compositional_diffusion_conflict as conflict  # noqa: E402
import compositional_diffusion_conflict_50subjects as k50  # noqa: E402

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402


def assert_exact_local_bayesflow() -> dict[str, Path]:
    """Fail before any work if a critical module is not from this checkout."""
    expected = {
        "bayesflow": LOCAL_BAYESFLOW / "bayesflow/__init__.py",
        "bayesflow.approximators.compositional_approximator": LOCAL_BAYESFLOW
        / "bayesflow/approximators/compositional_approximator.py",
        "bayesflow.approximators.helpers.compositional": LOCAL_BAYESFLOW
        / "bayesflow/approximators/helpers/compositional.py",
        "bayesflow.networks.inference.diffusion.diffusion_model": LOCAL_BAYESFLOW
        / "bayesflow/networks/inference/diffusion/diffusion_model.py",
        "bayesflow.networks.inference.diffusion.schedules.noise_schedule": LOCAL_BAYESFLOW
        / "bayesflow/networks/inference/diffusion/schedules/noise_schedule.py",
        "bayesflow.utils.integrate": LOCAL_BAYESFLOW / "bayesflow/utils/integrate.py",
        "bayesflow.workflows.compositional_workflow": LOCAL_BAYESFLOW
        / "bayesflow/workflows/compositional_workflow.py",
    }
    actual = {"bayesflow": Path(bf.__file__).resolve()}
    for module_name in tuple(expected)[1:]:
        module = importlib.import_module(module_name)
        actual[module_name] = Path(module.__file__).resolve()
    mismatches = {
        name: {"expected": str(expected[name].resolve()), "actual": str(actual[name])}
        for name in expected
        if actual[name] != expected[name].resolve()
    }
    if mismatches:
        raise ImportError(
            "Refusing to run with non-local BayesFlow modules: "
            + json.dumps(mismatches, sort_keys=True)
        )

    required_hashes = {
        "bayesflow.approximators.compositional_approximator": "5f17db615330c13e4b5c1108f491fd167a4a8c1c481e1a6e42cfe5179609ee70",
        "bayesflow.approximators.helpers.compositional": "414cfbde585117a264e8b8370b92eebfb0bbaf6ce4e99c93fb6c381e741454bb",
        "bayesflow.networks.inference.diffusion.diffusion_model": "e3da0e501fc077c71913b2eb10e86839e4c1e92e7a008e3854432c9606ace9a5",
        "bayesflow.networks.inference.diffusion.schedules.noise_schedule": "f014f563a0bac4c8690837507fb5e8e995260cea83801dcd0897cae46f577c58",
        "bayesflow.utils.integrate": "e3796dc2d595d8a1b2d4861358573a692b0fef45f427aed9c8c671e773fcea38",
        "bayesflow.workflows.compositional_workflow": "681ec469a124fcc73ac238a44e5ab6fa10b214b501903e0cd6a08982151f7c81",
    }
    hash_mismatches = {}
    for module_name, required in required_hashes.items():
        observed = hashlib.sha256(actual[module_name].read_bytes()).hexdigest()
        if observed != required:
            hash_mismatches[module_name] = {
                "required": required,
                "observed": observed,
            }
    if hash_mismatches:
        raise ImportError(
            "Local BayesFlow does not match the reviewed modified source: "
            + json.dumps(hash_mismatches, sort_keys=True)
        )
    return actual


LOCAL_BAYESFLOW_MODULES = assert_exact_local_bayesflow()


N_BLOCK_SUBJECTS = 50
N_TOTAL_SUBJECTS = 5_000
N_FACTORS = N_TOTAL_SUBJECTS // N_BLOCK_SUBJECTS
N_TRIALS = conflict.N_TRIALS
GLOBAL_BATCH_SIZE = 8
SUMMARY_BATCH_SIZE = 8
VALIDATION_SAMPLES = 256
VALIDATION_SEED = 20_260_713
DEFAULT_SEED = 20_260_713

SAVE_DIR = ROOT / "results/compositional_diffusion_conflict_50subjects_100s_validated"
MODEL_PATH = SAVE_DIR / "global_approximator.keras"
MODEL_CONFIG_PATH = SAVE_DIR / "model_config.json"
TRAINING_COMPLETE_PATH = SAVE_DIR / "training_complete.json"
HISTORY_PATH = SAVE_DIR / "global_history.json"

PRIOR_MEAN = np.asarray(
    [conflict.GLOBAL_PRIOR[name][0] for name in conflict.PARAM_NAMES_GLOBAL],
    dtype=np.float32,
)
PRIOR_STD = np.asarray(
    [conflict.GLOBAL_PRIOR[name][1] for name in conflict.PARAM_NAMES_GLOBAL],
    dtype=np.float32,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_source_hashes() -> dict[str, str]:
    paths = {
        "censored_simulator": ROOT / "conflict_ddm_censored.py",
        "conflict_model": ROOT / "compositional_diffusion_conflict.py",
        "hierarchical_summary": ROOT / "compositional_diffusion_conflict_50subjects.py",
        "compositional_approximator": LOCAL_BAYESFLOW
        / "bayesflow/approximators/compositional_approximator.py",
        "compositional_helper": LOCAL_BAYESFLOW
        / "bayesflow/approximators/helpers/compositional.py",
        "diffusion_model": LOCAL_BAYESFLOW
        / "bayesflow/networks/inference/diffusion/diffusion_model.py",
        "noise_schedule": LOCAL_BAYESFLOW
        / "bayesflow/networks/inference/diffusion/schedules/noise_schedule.py",
        "integrator": LOCAL_BAYESFLOW / "bayesflow/utils/integrate.py",
        "compositional_workflow": LOCAL_BAYESFLOW
        / "bayesflow/workflows/compositional_workflow.py",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hashes["workflow_builder"] = hashlib.sha256(
        inspect.getsource(make_workflow).encode()
    ).hexdigest()
    return hashes


def expected_model_config() -> dict:
    config = {
        "schema_version": 1,
        "bayesflow_source": "local_checkout",
        "bayesflow_version": getattr(bf, "__version__", "unknown"),
        "n_block_subjects": N_BLOCK_SUBJECTS,
        "n_trials": N_TRIALS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "online_batches_per_epoch": conflict.ONLINE_BATCHES_PER_EPOCH,
        "epochs": conflict.GLOBAL_EPOCHS,
        "validation_samples": VALIDATION_SAMPLES,
        "validation_seed": VALIDATION_SEED,
        "parameter_names": conflict.PARAM_NAMES_GLOBAL,
        "prior_mean": PRIOR_MEAN.tolist(),
        "prior_std": PRIOR_STD.tolist(),
        "standardization": "fixed_analytical_prior",
        "dynamic_standardization": None,
        "timeout_encoding": {
            "choice": censored.TIMEOUT_CHOICE,
            "deadline": censored.DEFAULT_MAX_TIME,
        },
        "trial_summary_network_kwargs": k50.TRIAL_SUMMARY_NETWORK_KWARGS,
        "subject_summary_network_kwargs": k50.SUBJECT_SUMMARY_NETWORK_KWARGS,
        "inference_network_kwargs": conflict.GLOBAL_INFERENCE_NETWORK_KWARGS,
        "training_source_sha256": training_source_hashes(),
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return config | {"fingerprint": hashlib.sha256(encoded).hexdigest()}


def configure_simulation(*, workers: int) -> None:
    """Point online training at the importable, correctly censored simulator."""

    conflict.N_LOCAL_SUBJECTS = N_BLOCK_SUBJECTS
    conflict.N_SUBJECTS = N_TOTAL_SUBJECTS
    conflict.N_TRIALS = N_TRIALS
    conflict.BATCH_SIZE = GLOBAL_BATCH_SIZE
    conflict.PARALLEL_WORKERS = workers
    conflict.SAVE_DIR = SAVE_DIR

    def make_training_data(n_samples: int):
        return censored.make_global_training_data(
            n_samples,
            n_subjects=N_BLOCK_SUBJECTS,
            n_trials=N_TRIALS,
            workers=conflict.PARALLEL_WORKERS,
        )

    conflict.make_global_training_data = make_training_data


def make_workflow(*, checkpoint: bool) -> bf.CompositionalWorkflow:
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(conflict.PARAM_NAMES_GLOBAL, into="inference_variables")
        .standardize(
            include="inference_variables",
            mean=PRIOR_MEAN,
            std=PRIOR_STD,
        )
        .rename("sim_data", "summary_variables")
    )
    kwargs = {}
    if checkpoint:
        kwargs = {
            "checkpoint_filepath": str(SAVE_DIR),
            "checkpoint_name": "global_approximator",
            "save_best_only": True,
        }
    return bf.CompositionalWorkflow(
        adapter=adapter,
        standardize=None,
        summary_network=k50.HierarchicalSetTransformer(
            n_subjects=N_BLOCK_SUBJECTS,
            n_trials=N_TRIALS,
        ),
        inference_network=bf.networks.DiffusionModel(
            **conflict.GLOBAL_INFERENCE_NETWORK_KWARGS
        ),
        **kwargs,
    )


def require_gpu() -> None:
    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit(
            "TensorFlow cannot see a GPU. Run this in the tf-gpu environment."
        )
    print(f"Verified local BayesFlow source: {Path(bf.__file__).resolve()}", flush=True)
    print(f"TensorFlow GPUs: {tf.config.list_logical_devices('GPU')}", flush=True)


def reshape_global_data(data: dict, n_subjects: int) -> dict:
    n_samples = data["sim_data"].shape[0]
    data["sim_data"] = data["sim_data"].reshape(
        n_samples,
        n_subjects * N_TRIALS,
        3,
    )
    return data


def validate_existing_model(expected: dict) -> bool:
    paths = (MODEL_PATH, MODEL_CONFIG_PATH, TRAINING_COMPLETE_PATH)
    if not any(path.exists() for path in paths):
        return False
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Refusing to reuse an incomplete training run; missing "
            + ", ".join(missing)
            + ". Use --force-train to replace it."
        )
    actual = json.loads(MODEL_CONFIG_PATH.read_text())
    complete = json.loads(TRAINING_COMPLETE_PATH.read_text())
    if actual.get("fingerprint") != expected["fingerprint"]:
        raise RuntimeError(
            "Saved model configuration does not match this script. "
            "Use --force-train to train a compatible model."
        )
    if complete.get("fingerprint") != expected["fingerprint"]:
        raise RuntimeError(
            "Training marker does not match the model configuration. "
            "Use --force-train to replace it."
        )
    return True


def fit_or_load_global(*, force_train: bool, workers: int):
    expected = expected_model_config()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_train and validate_existing_model(expected):
        workflow = make_workflow(checkpoint=False)
        workflow.approximator = keras.saving.load_model(MODEL_PATH)
        print(f"[global] loaded verified model {MODEL_PATH}", flush=True)
        return workflow, expected

    TRAINING_COMPLETE_PATH.unlink(missing_ok=True)
    atomic_json(MODEL_CONFIG_PATH, expected)
    workflow = make_workflow(checkpoint=True)
    workflow.simulator = conflict.BufferedOnlineSimulator(
        "global", conflict.ONLINE_BATCHES_PER_EPOCH
    )

    print(
        f"[validation] simulating {VALIDATION_SAMPLES} fixed K=50 datasets",
        flush=True,
    )
    validation = censored.sample_data(
        VALIDATION_SAMPLES,
        n_subjects=N_BLOCK_SUBJECTS,
        n_trials=N_TRIALS,
        workers=workers,
        seed=VALIDATION_SEED,
    )
    reshape_global_data(validation, N_BLOCK_SUBJECTS)

    print(
        "[global] training in exact N(0,I) prior coordinates with a 100-second deadline",
        flush=True,
    )
    started = time.monotonic()
    history = workflow.fit_online(
        epochs=conflict.GLOBAL_EPOCHS,
        num_batches_per_epoch=conflict.ONLINE_BATCHES_PER_EPOCH,
        batch_size=GLOBAL_BATCH_SIZE,
        validation_data=validation,
        callbacks=[conflict.EpochProgressCallback("global").callback],
        verbose=0,
    )
    elapsed = time.monotonic() - started
    metrics = conflict.summarize_history(history)
    validation_losses = metrics.get("val_loss", [])
    if not validation_losses:
        raise RuntimeError(
            "Training did not report val_loss; best-model selection is invalid."
        )
    best_epoch = int(np.argmin(validation_losses)) + 1
    atomic_json(
        HISTORY_PATH,
        {
            "metrics": metrics,
            "best_epoch": best_epoch,
            "best_val_loss": float(validation_losses[best_epoch - 1]),
            "training_elapsed_seconds": elapsed,
        },
    )
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Best-model checkpoint was not created at {MODEL_PATH}.")
    workflow.approximator = keras.saving.load_model(MODEL_PATH)
    atomic_json(
        TRAINING_COMPLETE_PATH,
        {
            "fingerprint": expected["fingerprint"],
            "best_epoch": best_epoch,
            "best_val_loss": float(validation_losses[best_epoch - 1]),
        },
    )
    print(
        f"[global] loaded best validation checkpoint from epoch {best_epoch}",
        flush=True,
    )
    return workflow, expected


def parameter_matrix(data: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(data[name]).reshape(len(data[name]), -1)[:, :1]
            for name in conflict.PARAM_NAMES_GLOBAL
        ],
        axis=-1,
    )


def posterior_matrix(samples: dict) -> np.ndarray:
    missing = [name for name in conflict.PARAM_NAMES_GLOBAL if name not in samples]
    if missing:
        raise KeyError(f"Posterior output is missing variables: {missing}")
    return np.concatenate(
        [np.asarray(samples[name]) for name in conflict.PARAM_NAMES_GLOBAL], axis=-1
    )


def dataset_scores(samples: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    medians = np.median(samples, axis=1)
    normalized_error = (medians - targets) / PRIOR_STD
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=1)
    posterior_variance = np.var(samples, axis=1, ddof=1)
    return {
        "normalized_squared_error": np.mean(normalized_error**2, axis=-1),
        "normalized_absolute_error": np.mean(np.abs(normalized_error), axis=-1),
        "coverage_95": np.mean((targets >= lower) & (targets <= upper), axis=-1),
        "normalized_posterior_sd": np.mean(
            np.sqrt(posterior_variance) / PRIOR_STD, axis=-1
        ),
        "contraction": np.mean(1.0 - posterior_variance / PRIOR_STD**2, axis=-1),
    }


def recovery_report(
    samples: np.ndarray,
    targets: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_draws: int = 2_000,
) -> dict:
    scores = dataset_scores(samples, targets)
    medians = np.median(samples, axis=1)
    per_parameter = {}
    for index, name in enumerate(conflict.PARAM_NAMES_GLOBAL):
        if targets.shape[0] < 2:
            correlation = None
        else:
            value = np.corrcoef(medians[:, index], targets[:, index])[0, 1]
            correlation = float(value) if np.isfinite(value) else None
        per_parameter[name] = {
            "correlation": correlation,
            "normalized_rmse": float(
                np.sqrt(
                    np.mean(
                        ((medians[:, index] - targets[:, index]) / PRIOR_STD[index])
                        ** 2
                    )
                )
            ),
            "coverage_95": float(
                np.mean(
                    (
                        targets[:, index]
                        >= np.quantile(samples[:, :, index], 0.025, axis=1)
                    )
                    & (
                        targets[:, index]
                        <= np.quantile(samples[:, :, index], 0.975, axis=1)
                    )
                )
            ),
        }

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, targets.shape[0], size=(bootstrap_draws, targets.shape[0])
    )
    report = {"n_datasets": int(targets.shape[0]), "per_parameter": per_parameter}
    for name, values in scores.items():
        bootstrap = np.mean(values[indices], axis=1)
        estimate = float(np.mean(values))
        if name == "normalized_squared_error":
            name = "normalized_rmse"
            bootstrap = np.sqrt(bootstrap)
            estimate = float(np.sqrt(estimate))
        report[name] = {
            "estimate": estimate,
            "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    return report


def load_or_make_factor_data(args, run_dir: Path):
    cache = run_dir / "factor_test_data.npz"
    if cache.exists() and not args.force_evaluation:
        stored = np.load(cache, allow_pickle=False)
        return stored["sim_data"], stored["targets"], int(stored["timeouts"])

    print(f"[factor check] simulating {args.factor_n_test} K=50 datasets", flush=True)
    data = censored.sample_data(
        args.factor_n_test,
        n_subjects=N_BLOCK_SUBJECTS,
        n_trials=N_TRIALS,
        workers=args.workers,
        seed=args.seed + 1,
    )
    timeouts = censored.count_timeouts(data["sim_data"])
    sim_data = data["sim_data"].reshape(
        args.factor_n_test, N_BLOCK_SUBJECTS * N_TRIALS, 3
    )
    targets = parameter_matrix(data)
    atomic_npz(cache, sim_data=sim_data, targets=targets, timeouts=np.asarray(timeouts))
    return sim_data, targets, timeouts


def factor_recovery(workflow, args, run_dir: Path) -> dict:
    result_path = run_dir / "factor_recovery.npz"
    report_path = run_dir / "factor_recovery.json"
    sim_data, targets, timeouts = load_or_make_factor_data(args, run_dir)
    if result_path.exists() and report_path.exists() and not args.force_evaluation:
        return json.loads(report_path.read_text())

    print("[factor check] drawing ordinary K=50 posterior samples", flush=True)
    posterior = workflow.sample(
        num_samples=args.factor_n_samples,
        conditions={"sim_data": sim_data},
        batch_size=args.dataset_batch_size,
        method="two_step_adaptive",
        steps="adaptive",
        seed=args.seed + 2,
    )
    samples = posterior_matrix(posterior)
    if not np.isfinite(samples).all():
        raise FloatingPointError("Non-finite samples in the K=50 factor check.")
    report = recovery_report(samples, targets, bootstrap_seed=args.seed + 3)
    report["timeout_trials"] = timeouts
    atomic_npz(result_path, samples=samples, targets=targets)
    atomic_json(report_path, report)
    return report


def load_or_make_compositional_data(workflow, args, run_dir: Path):
    cache = run_dir / "compositional_summaries.npz"
    if cache.exists() and not args.force_evaluation:
        stored = np.load(cache, allow_pickle=False)
        return stored["summaries"], stored["targets"], int(stored["timeouts"])

    print(
        f"[composition] simulating {args.n_test} datasets with {N_TOTAL_SUBJECTS} subjects",
        flush=True,
    )
    data = censored.sample_data(
        args.n_test,
        n_subjects=N_TOTAL_SUBJECTS,
        n_trials=N_TRIALS,
        workers=args.workers,
        seed=args.seed + 10,
    )
    timeouts = censored.count_timeouts(data["sim_data"])
    targets = parameter_matrix(data)
    blocks = data["sim_data"].reshape(
        args.n_test * N_FACTORS,
        N_BLOCK_SUBJECTS * N_TRIALS,
        3,
    )
    print(
        f"[composition] summarizing {args.n_test * N_FACTORS} K=50 factors", flush=True
    )
    summaries = workflow.approximator.summarize(
        {"sim_data": blocks},
        batch_size=args.summary_batch_size,
    )
    summaries = np.asarray(summaries).reshape(args.n_test, N_FACTORS, -1)
    atomic_npz(
        cache, summaries=summaries, targets=targets, timeouts=np.asarray(timeouts)
    )
    return summaries, targets, timeouts


def sample_variant(
    workflow,
    summaries: np.ndarray,
    targets: np.ndarray,
    args,
    run_dir: Path,
    *,
    label: str,
    mini_batch_size: int | None,
) -> tuple[np.ndarray, dict]:
    result_path = run_dir / f"{label}.npz"
    report_path = run_dir / f"{label}.json"
    partial_path = run_dir / f"{label}.partial.npz"
    metadata = {
        "label": label,
        "mini_batch_size": N_FACTORS if mini_batch_size is None else mini_batch_size,
        "n_factors": N_FACTORS,
        "num_samples": args.n_samples,
        "method": "two_step_adaptive",
        "steps": "adaptive",
        "compute_prior_score": "standard_normal",
        "compositional_bridge_d0": 1.0,
        "compositional_bridge_d1": 1.0,
        "clip": None,
    }
    metadata_string = json.dumps(metadata, sort_keys=True)

    if result_path.exists() and report_path.exists() and not args.force_evaluation:
        stored = np.load(result_path, allow_pickle=False)
        if str(stored["metadata"]) != metadata_string:
            raise RuntimeError(f"Cached {label} result has incompatible metadata.")
        return stored["samples"], json.loads(report_path.read_text())

    completed = 0
    posterior_batches = []
    if partial_path.exists() and not args.force_evaluation:
        stored = np.load(partial_path, allow_pickle=False)
        if str(stored["metadata"]) != metadata_string:
            raise RuntimeError(f"Partial {label} result has incompatible metadata.")
        completed = int(stored["completed"])
        partial_samples = stored["samples"]
        if completed > summaries.shape[0] or partial_samples.shape[0] != completed:
            raise RuntimeError(
                f"Partial {label} result has an invalid completed count."
            )
        posterior_batches = [partial_samples]
        print(f"[{label}] resuming after {completed} datasets", flush=True)

    for start in range(completed, summaries.shape[0], args.dataset_batch_size):
        stop = min(start + args.dataset_batch_size, summaries.shape[0])
        chunks = []
        for sample_start in range(0, args.n_samples, args.sample_chunk_size):
            count = min(args.sample_chunk_size, args.n_samples - sample_start)
            posterior = workflow.compositional_sample(
                num_samples=count,
                summaries=summaries[start:stop],
                compute_prior_score="standard_normal",
                method="two_step_adaptive",
                steps="adaptive",
                mini_batch_size=mini_batch_size,
                compositional_bridge_d0=1.0,
                compositional_bridge_d1=1.0,
                clip=None,
                batch_size=args.dataset_batch_size,
                seed=args.seed + 100 + start * args.n_samples + sample_start,
            )
            chunks.append(posterior_matrix(posterior))
        batch = np.concatenate(chunks, axis=1)
        if not np.isfinite(batch).all():
            raise FloatingPointError(f"Non-finite posterior samples in {label}.")
        posterior_batches.append(batch)
        combined = np.concatenate(posterior_batches, axis=0)
        atomic_npz(
            partial_path,
            samples=combined,
            completed=np.asarray(stop),
            metadata=np.asarray(metadata_string),
        )
        print(
            f"[{label}] completed datasets {start + 1}-{stop}/{summaries.shape[0]}",
            flush=True,
        )

    samples = np.concatenate(posterior_batches, axis=0)
    report = recovery_report(
        samples,
        targets,
        bootstrap_seed=args.seed + (201 if mini_batch_size is None else 202),
    )
    report["sampler"] = metadata
    atomic_npz(
        result_path,
        samples=samples,
        targets=targets,
        metadata=np.asarray(metadata_string),
    )
    atomic_json(report_path, report)
    partial_path.unlink(missing_ok=True)
    return samples, report


def paired_comparison(
    full: np.ndarray, minibatch: np.ndarray, targets: np.ndarray, seed: int
) -> dict:
    full_scores = dataset_scores(full, targets)
    minibatch_scores = dataset_scores(minibatch, targets)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, targets.shape[0], size=(2_000, targets.shape[0]))
    comparison = {}
    for name in full_scores:
        difference = minibatch_scores[name] - full_scores[name]
        bootstrap = np.mean(difference[indices], axis=1)
        comparison[f"minibatch_minus_full_{name}"] = {
            "estimate": float(np.mean(difference)),
            "paired_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    return comparison


def benchmark(workflow, model_config: dict, args) -> None:
    model_hash = sha256_file(MODEL_PATH)
    run_name = (
        f"model-{model_hash[:12]}_test-{args.n_test}_samples-{args.n_samples}"
        f"_factor-{args.factor_n_test}x{args.factor_n_samples}_seed-{args.seed}"
    )
    run_dir = SAVE_DIR / "benchmarks" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_dir / "run_config.json",
        {
            "model_sha256": model_hash,
            "model_config_fingerprint": model_config["fingerprint"],
            "n_total_subjects": N_TOTAL_SUBJECTS,
            "n_factors": N_FACTORS,
            "factor_subjects": N_BLOCK_SUBJECTS,
            "arguments": vars(args),
        },
    )

    factor_report = factor_recovery(workflow, args, run_dir)
    print(
        f"[factor check] normalized RMSE={factor_report['normalized_rmse']['estimate']:.4f}",
        flush=True,
    )

    summaries, targets, timeouts = load_or_make_compositional_data(
        workflow, args, run_dir
    )
    full, full_report = sample_variant(
        workflow,
        summaries,
        targets,
        args,
        run_dir,
        label="full_m100",
        mini_batch_size=None,
    )
    minibatch, minibatch_report = sample_variant(
        workflow,
        summaries,
        targets,
        args,
        run_dir,
        label="minibatch_m10",
        mini_batch_size=10,
    )
    comparison = paired_comparison(full, minibatch, targets, args.seed + 300)
    comparison["timeout_trials"] = timeouts
    comparison["full_normalized_rmse"] = full_report["normalized_rmse"]
    comparison["minibatch_normalized_rmse"] = minibatch_report["normalized_rmse"]
    atomic_json(run_dir / "paired_comparison.json", comparison)
    print(f"[benchmark] results saved in {run_dir}", flush=True)


def smoke_test() -> None:
    if censored.DEFAULT_MAX_TIME != 100.0 or censored.TIMEOUT_CHOICE != 0.0:
        raise AssertionError("The requested timeout convention is not active.")
    choice, rt = censored.simulate_ddm_trial(
        drift=0.0,
        boundary=1e12,
        non_decision_time=0.01,
        dt=0.001,
        max_time=0.02,
    )
    if choice != censored.TIMEOUT_CHOICE or rt != 0.02:
        raise AssertionError(f"Timeout encoding failed: choice={choice}, rt={rt}")

    workflow = make_workflow(checkpoint=False)
    data = censored.sample_serial(
        2,
        n_subjects=N_BLOCK_SUBJECTS,
        n_trials=N_TRIALS,
        seed=DEFAULT_SEED,
    )
    reshape_global_data(data, N_BLOCK_SUBJECTS)
    adapted = workflow.adapter(data, strict=True)
    standardized = np.asarray(adapted["inference_variables"])
    manual = (parameter_matrix(data) - PRIOR_MEAN) / PRIOR_STD
    np.testing.assert_allclose(standardized, manual, rtol=1e-6, atol=1e-6)

    summaries = workflow.approximator.summarize(
        {"sim_data": data["sim_data"]}, batch_size=1
    )
    if tuple(summaries.shape) != (2, 64):
        raise AssertionError(f"Unexpected summary shape {summaries.shape}")
    history = workflow.fit_offline(data, epochs=1, batch_size=1, verbose=0)
    loss = float(history.history["loss"][-1])
    if not np.isfinite(loss):
        raise FloatingPointError(f"Non-finite smoke-test loss {loss}")

    smoke_path = Path(
        "/tmp/compositional_diffusion_conflict_50subjects_validated.keras"
    )
    before = workflow.approximator.summarize(
        {"sim_data": data["sim_data"]}, batch_size=1
    )
    keras.saving.save_model(workflow.approximator, smoke_path, overwrite=True)
    restored = keras.saving.load_model(smoke_path)
    after = restored.summarize({"sim_data": data["sim_data"]}, batch_size=1)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
    print(f"Smoke test passed: summary={summaries.shape}, loss={loss:.5g}", flush=True)


def print_plan(args) -> None:
    print("Corrected K=50 compositional benchmark")
    print(f"  BayesFlow: {Path(bf.__file__).resolve()}")
    print(f"  training block: {N_BLOCK_SUBJECTS} subjects x {N_TRIALS} trials")
    print("  prior coordinates: fixed analytical N(0,I)")
    print("  timeout: choice=0 at exactly 100 seconds")
    print(
        f"  validation: {VALIDATION_SAMPLES} fixed datasets; best val_loss checkpoint"
    )
    print(f"  composition: {N_FACTORS} factors; compare M=100 with M=10")
    print("  bridge: d0=d1=1; adaptive two-step SDE; no clipping")
    print(f"  benchmark: {args.n_test} datasets x {args.n_samples} posterior draws")
    print(f"  output: {SAVE_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--workers", type=int, default=conflict.PARALLEL_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--factor-n-test", type=int, default=200)
    parser.add_argument("--factor-n-samples", type=int, default=500)
    parser.add_argument("--dataset-batch-size", type=int, default=4)
    parser.add_argument("--summary-batch-size", type=int, default=SUMMARY_BATCH_SIZE)
    parser.add_argument("--sample-chunk-size", type=int, default=25)
    args = parser.parse_args()
    for name in (
        "workers",
        "n_test",
        "n_samples",
        "factor_n_test",
        "factor_n_samples",
        "dataset_batch_size",
        "summary_batch_size",
        "sample_chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_samples < 2 or args.factor_n_samples < 2:
        parser.error("posterior sample counts must be at least 2")
    if args.evaluation_only and args.force_train:
        parser.error("--evaluation-only cannot be combined with --force-train")
    return args


def main() -> None:
    args = parse_args()
    configure_simulation(workers=args.workers)
    print_plan(args)
    if args.plan_only:
        return
    if args.smoke_test:
        smoke_test()
        return

    require_gpu()
    if args.evaluation_only:
        expected = expected_model_config()
        if not validate_existing_model(expected):
            raise RuntimeError(
                "No complete compatible model exists for --evaluation-only."
            )
        workflow = make_workflow(checkpoint=False)
        workflow.approximator = keras.saving.load_model(MODEL_PATH)
        model_config = expected
    else:
        workflow, model_config = fit_or_load_global(
            force_train=args.force_train,
            workers=args.workers,
        )
    if not args.train_only:
        benchmark(workflow, model_config, args)


if __name__ == "__main__":
    main()
