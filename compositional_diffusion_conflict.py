"""Run the compositional diffusion tutorial with the hierarchical conflict DDM.

This is intentionally a close variant of ``compositional_diffusion.py``:
the TensorFlow runtime, compositional sampling settings, and output structure
are kept the same. The Bayesian model is changed to the hierarchical conflict
model from ``initial.py``, with a fixed trial count instead of variable trial
counts.

Usage:
    conda run --no-capture-output -n tf-gpu python -u compositional_diffusion_conflict.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
from numba import njit


# Tutorial defaults. Kept in sync with compositional_diffusion.py.
N_TRAINING_BATCHES = 256
BATCH_SIZE = 64
GLOBAL_EPOCHS = 1_000
LOCAL_EPOCHS = 50
N_LOCAL_SUBJECTS = 5
N_TRIALS = 120
N_SUBJECTS = 10_000
N_TEST = 100
N_SAMPLES = 100
N_LOCAL_CONDITIONS = 10
MINI_BATCH_SIZE = 10
# The global SetTransformer attends over 5 * 120 = 600 observations. A batch
# of 640 therefore needs a multi-gigabyte attention tensor during evaluation.
COMPOSITIONAL_SUMMARY_BATCH_SIZE = BATCH_SIZE
# Diffusion sampling repeats every dataset N_SAMPLES times while retaining all
# compositional summaries. Keep this outer dataset batch small enough for VRAM.
COMPOSITIONAL_DATASET_BATCH_SIZE = 4
PARALLEL_WORKERS = 8
ONLINE_TRAINING = True
ONLINE_BATCHES_PER_EPOCH = N_TRAINING_BATCHES
REUSE_SAVED_MODELS = True
SAVE_DIR = Path("results/compositional_diffusion_conflict_120trials_online_large")

GLOBAL_SUMMARY_NETWORK_KWARGS = {
    "summary_dim": 64,
    "embed_dims": (96, 96, 96),
    "num_heads": (4, 4, 4),
    "num_seeds": 4,
    "dropout": 0.05,
}
GLOBAL_INFERENCE_NETWORK_KWARGS = {
    "subnet_kwargs": {
        "widths": (256, 256, 256, 256, 256),
        "dropout": 0.05,
    },
}

GLOBAL_PRIOR = {
    "mu_v": (2.0, 1.5),
    "mu_v_cond": (2.0, 1.5),
    "mu_log_a": (np.log(1.4), 0.40),
    "mu_log_t0": (np.log(0.35), 0.25),
    "mu_log_sv": (np.log(0.50), 0.40),
    "log_sigma_v": (np.log(0.70), 0.50),
    "log_sigma_v_cond": (np.log(0.70), 0.50),
    "log_sigma_log_a": (np.log(0.35), 0.50),
    "log_sigma_log_t0": (np.log(0.20), 0.50),
    "log_sigma_log_sv": (np.log(0.35), 0.50),
}

PARAM_NAMES_GLOBAL = list(GLOBAL_PRIOR)
PARAM_NAMES_LOCAL = ["v", "v_cond", "log_a", "log_t0", "log_sv"]
PRETTY_PARAM_NAMES_GLOBAL = [
    r"$\mu_v$",
    r"$\mu_{v_\mathrm{cond}}$",
    r"$\mu_{\log a}$",
    r"$\mu_{\log t_0}$",
    r"$\mu_{\log s_v}$",
    r"$\log \sigma_v$",
    r"$\log \sigma_{v_\mathrm{cond}}$",
    r"$\log \sigma_{\log a}$",
    r"$\log \sigma_{\log t_0}$",
    r"$\log \sigma_{\log s_v}$",
]
PRETTY_PARAM_NAMES_LOCAL = [
    r"$v_p$",
    r"$v_{\mathrm{cond},p}$",
    r"$\log a_p$",
    r"$\log t_{0,p}$",
    r"$\log s_{v,p}$",
]

bf = None
keras = None


@njit
def seed_numba(seed: int) -> None:
    np.random.seed(seed)


@njit
def simulate_ddm_trial(
    drift,
    boundary,
    non_decision_time,
    start_fraction=0.5,
    dt=1e-3,
    max_time=10.0,
):
    evidence = start_fraction * boundary
    rt = non_decision_time
    noise_scale = np.sqrt(dt)

    while (boundary >= evidence >= 0.0) and rt <= max_time:
        evidence += drift * dt + noise_scale * np.random.randn()
        rt += dt

    choice = 1.0 if evidence >= boundary else 0.0
    return choice, rt


@njit
def simulate_subjects(v, v_cond, log_a, log_t0, log_sv, n_trials, sim_data, dt, max_time):
    for i_subject in range(v.shape[0]):
        half = n_trials // 2

        condition = np.empty(n_trials)
        for i_trial in range(half):
            condition[i_trial] = 0.5
        for i_trial in range(half, n_trials):
            condition[i_trial] = -0.5
        np.random.shuffle(condition)

        boundary = np.exp(log_a[i_subject])
        non_decision_time = np.exp(log_t0[i_subject])
        drift_sd = np.exp(log_sv[i_subject])

        for i_trial in range(n_trials):
            trial_drift = v[i_subject] + v_cond[i_subject] * condition[i_trial]
            trial_drift += drift_sd * np.random.randn()

            choice, rt = simulate_ddm_trial(
                trial_drift,
                boundary,
                non_decision_time,
                0.5,
                dt=dt,
                max_time=max_time,
            )
            sim_data[i_subject, i_trial, 0] = choice
            sim_data[i_subject, i_trial, 1] = rt
            sim_data[i_subject, i_trial, 2] = condition[i_trial]


def score_normal(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return -(x - mean) / sd**2


def sample_hierarchical_priors(n_subjects=1):
    mu_v = np.random.normal(*GLOBAL_PRIOR["mu_v"])
    mu_v_cond = np.random.normal(*GLOBAL_PRIOR["mu_v_cond"])
    mu_log_a = np.random.normal(*GLOBAL_PRIOR["mu_log_a"])
    mu_log_t0 = np.random.normal(*GLOBAL_PRIOR["mu_log_t0"])
    mu_log_sv = np.random.normal(*GLOBAL_PRIOR["mu_log_sv"])

    log_sigma_v = np.random.normal(*GLOBAL_PRIOR["log_sigma_v"])
    log_sigma_v_cond = np.random.normal(*GLOBAL_PRIOR["log_sigma_v_cond"])
    log_sigma_log_a = np.random.normal(*GLOBAL_PRIOR["log_sigma_log_a"])
    log_sigma_log_t0 = np.random.normal(*GLOBAL_PRIOR["log_sigma_log_t0"])
    log_sigma_log_sv = np.random.normal(*GLOBAL_PRIOR["log_sigma_log_sv"])

    v = np.random.normal(mu_v, np.exp(log_sigma_v), size=n_subjects)
    v_cond = np.random.normal(mu_v_cond, np.exp(log_sigma_v_cond), size=n_subjects)
    log_a = np.random.normal(mu_log_a, np.exp(log_sigma_log_a), size=n_subjects)
    log_t0 = np.random.normal(mu_log_t0, np.exp(log_sigma_log_t0), size=n_subjects)
    log_sv = np.random.normal(mu_log_sv, np.exp(log_sigma_log_sv), size=n_subjects)

    return {
        "mu_v": mu_v,
        "mu_v_cond": mu_v_cond,
        "mu_log_a": mu_log_a,
        "mu_log_t0": mu_log_t0,
        "mu_log_sv": mu_log_sv,
        "log_sigma_v": log_sigma_v,
        "log_sigma_v_cond": log_sigma_v_cond,
        "log_sigma_log_a": log_sigma_log_a,
        "log_sigma_log_t0": log_sigma_log_t0,
        "log_sigma_log_sv": log_sigma_log_sv,
        "v": v,
        "v_cond": v_cond,
        "log_a": log_a,
        "log_t0": log_t0,
        "log_sv": log_sv,
        "a": np.exp(log_a),
        "t0": np.exp(log_t0),
        "sv": np.exp(log_sv),
    }


def prior_global_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: score_normal(x[name], mean=GLOBAL_PRIOR[name][0], sd=GLOBAL_PRIOR[name][1])
        for name in PARAM_NAMES_GLOBAL
    }


def simulate_hierarchical_ddm(
    v,
    v_cond,
    log_a,
    log_t0,
    log_sv,
    n_trials=N_TRIALS,
    dt=1e-3,
    max_time=10.0,
):
    if n_trials % 2 != 0:
        raise ValueError("n_trials must be even to keep the two conditions balanced.")

    v = np.asarray(v)
    n_subjects = v.shape[0]
    data = np.zeros((n_subjects, n_trials, 3), dtype=np.float64)
    simulate_subjects(
        v,
        np.asarray(v_cond),
        np.asarray(log_a),
        np.asarray(log_t0),
        np.asarray(log_sv),
        n_trials,
        data,
        dt,
        max_time,
    )

    if n_subjects == 1 and n_trials == 1:
        data = data[0, 0]
    elif n_subjects == 1:
        data = data[0]
    elif n_trials == 1:
        data = data[:, 0]

    return {"sim_data": data}


def simulate_one_dataset(n_subjects=1, n_trials=N_TRIALS):
    params = sample_hierarchical_priors(n_subjects=n_subjects)
    data = simulate_hierarchical_ddm(
        params["v"],
        params["v_cond"],
        params["log_a"],
        params["log_t0"],
        params["log_sv"],
        n_trials=n_trials,
    )
    return params | data


def stack_simulation_results(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    combined = {}
    keys = results[0].keys()
    for key in keys:
        value = np.stack([result[key] for result in results], axis=0)
        if value.ndim == 1:
            value = np.expand_dims(value, axis=-1)
        combined[key] = value
    return combined


def sample_serial(n_samples: int, n_subjects=1, n_trials=N_TRIALS, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
        seed_numba(seed + 1)
    return stack_simulation_results(
        [simulate_one_dataset(n_subjects=n_subjects, n_trials=n_trials) for _ in range(n_samples)]
    )


def split_counts(n_samples: int, n_chunks: int) -> list[int]:
    n_chunks = max(1, min(n_chunks, n_samples))
    base = n_samples // n_chunks
    extra = n_samples % n_chunks
    return [base + int(i < extra) for i in range(n_chunks)]


def merge_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in batches[0]}


def sample_parallel(n_samples: int, n_subjects=1, n_trials=N_TRIALS, workers=PARALLEL_WORKERS):
    from joblib import Parallel, delayed

    counts = split_counts(n_samples, workers * 4)
    seeds = np.random.SeedSequence().generate_state(len(counts)).astype(int)
    batches = Parallel(n_jobs=workers)(
        delayed(sample_serial)(
            count,
            n_subjects=n_subjects,
            n_trials=n_trials,
            seed=int(seed),
        )
        for count, seed in zip(counts, seeds)
        if count > 0
    )
    return merge_batches(batches)


def sample_data(n_samples: int, n_subjects=1, n_trials=N_TRIALS):
    return sample_parallel(
        n_samples,
        n_subjects=n_subjects,
        n_trials=n_trials,
        workers=PARALLEL_WORKERS,
    )


def make_global_training_data(n_samples: int):
    data = sample_data(n_samples, n_subjects=N_LOCAL_SUBJECTS, n_trials=N_TRIALS)
    data["sim_data"] = data["sim_data"].reshape(n_samples, N_LOCAL_SUBJECTS * N_TRIALS, 3)
    return data


def make_local_training_data(n_samples: int):
    return sample_data(n_samples, n_trials=N_TRIALS)


class BufferedOnlineSimulator:
    """Generate fresh epoch-sized buffers for BayesFlow online training."""

    def __init__(self, label: str, n_batches_per_epoch: int):
        self.label = label
        self.n_batches_per_epoch = n_batches_per_epoch
        self.buffer = None
        self.cursor = 0

    def sample(self, batch_size: int):
        if self.buffer is None or self.cursor + batch_size > self._buffer_size():
            n_samples = self.n_batches_per_epoch * batch_size
            if self.label == "global":
                self.buffer = make_global_training_data(n_samples)
            elif self.label == "local":
                self.buffer = make_local_training_data(n_samples)
            else:
                raise ValueError(f"Unknown online simulator label: {self.label}")
            self.cursor = 0

        start = self.cursor
        stop = start + batch_size
        self.cursor = stop
        return {key: value[start:stop] for key, value in self.buffer.items()}

    def _buffer_size(self):
        return len(next(iter(self.buffer.values())))


def load_runtime():
    global bf, keras
    if bf is not None and keras is not None:
        return

    import tensorflow as tf

    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not physical_gpus:
        raise SystemExit("TensorFlow cannot see a GPU. Run this in the tf-gpu conda environment.")

    logging.disable(logging.INFO)
    import bayesflow as bf_module
    import keras as keras_module
    logging.disable(logging.NOTSET)

    bf = bf_module
    keras = keras_module
    logging.getLogger("bayesflow").setLevel(logging.WARNING)

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow GPU devices: {logical_gpus or physical_gpus}")


def make_workflows():
    load_runtime()

    adapter_global = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(PARAM_NAMES_GLOBAL, into="inference_variables")
        .rename("sim_data", "summary_variables")
    )
    workflow_global = bf.CompositionalWorkflow(
        adapter=adapter_global,
        summary_network=bf.networks.SetTransformer(**GLOBAL_SUMMARY_NETWORK_KWARGS),
        inference_network=bf.networks.DiffusionModel(**GLOBAL_INFERENCE_NETWORK_KWARGS),
    )

    adapter_local = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(PARAM_NAMES_LOCAL, into="inference_variables")
        .concatenate(PARAM_NAMES_GLOBAL, into="inference_conditions")
        .rename("sim_data", "summary_variables")
    )
    workflow_local = bf.BasicWorkflow(
        adapter=adapter_local,
        summary_network=bf.networks.SetTransformer(summary_dim=16),
        inference_network=bf.networks.StableConsistencyModel(),
    )

    return workflow_global, workflow_local


class EpochProgressCallback:
    def __init__(self, label: str):
        load_runtime()
        self.callback = self._make_callback(label)

    @staticmethod
    def _make_callback(label: str):
        class Callback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.started_at = time.monotonic()
                self.first_loss = None
                print(f"[{label}] training started", flush=True)

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                elapsed = time.monotonic() - self.started_at
                loss = logs.get("loss")
                val_loss = logs.get("val_loss")
                if loss is not None and self.first_loss is None:
                    self.first_loss = float(loss)

                parts = [f"[{label}] epoch={epoch + 1}", f"elapsed={elapsed:.1f}s"]
                if loss is not None:
                    loss = float(loss)
                    parts.append(f"loss={loss:.5g}")
                    parts.append(f"loss_delta={loss - self.first_loss:+.5g}")
                if val_loss is not None:
                    parts.append(f"val_loss={float(val_loss):.5g}")
                print(" ".join(parts), flush=True)

        return Callback()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {path}", flush=True)


def save_figure(path: Path, fig) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def save_npz_dict(path: Path, **sections: dict[str, np.ndarray]) -> None:
    arrays = {}
    for section_name, values in sections.items():
        for key, value in values.items():
            if key == "sim_data":
                continue
            arrays[f"{section_name}_{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"saved {path}", flush=True)


def summarize_history(history):
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def compositional_sample_batched(workflow, conditions, **sample_kwargs):
    """Sample small groups of datasets while summarizing each group efficiently."""
    n_datasets = np.asarray(next(iter(conditions.values()))).shape[0]
    posterior_batches = []

    for start in range(0, n_datasets, COMPOSITIONAL_DATASET_BATCH_SIZE):
        stop = min(start + COMPOSITIONAL_DATASET_BATCH_SIZE, n_datasets)
        print(
            f"[compositional] dataset batch {start + 1}-{stop}/{n_datasets}",
            flush=True,
        )
        batch_conditions = {key: np.asarray(value)[start:stop] for key, value in conditions.items()}
        posterior_batches.append(
            workflow.compositional_sample(
                conditions=batch_conditions,
                batch_size=COMPOSITIONAL_SUMMARY_BATCH_SIZE,
                **sample_kwargs,
            )
        )

    return {
        key: np.concatenate([posterior[key] for posterior in posterior_batches], axis=0)
        for key in posterior_batches[0]
    }


def fit_or_load_global(workflow_global, save_dir: Path):
    model_path = save_dir / "global_approximator.keras"
    if REUSE_SAVED_MODELS and model_path.exists():
        print(f"[global] loading saved approximator from {model_path}", flush=True)
        workflow_global.approximator = keras.saving.load_model(model_path)
        return

    n_samples_per_epoch = ONLINE_BATCHES_PER_EPOCH * BATCH_SIZE
    simulation_elapsed = 0.0

    if ONLINE_TRAINING:
        print(
            f"[global] online training with {n_samples_per_epoch} fresh datasets per epoch",
            flush=True,
        )
        print(
            f"[global] per-dataset sim_data shape=({N_LOCAL_SUBJECTS * N_TRIALS}, 3)",
            flush=True,
        )
        workflow_global.simulator = BufferedOnlineSimulator("global", ONLINE_BATCHES_PER_EPOCH)
        started = time.monotonic()
        history = workflow_global.fit_online(
            epochs=GLOBAL_EPOCHS,
            num_batches_per_epoch=ONLINE_BATCHES_PER_EPOCH,
            batch_size=BATCH_SIZE,
            callbacks=[EpochProgressCallback("global").callback],
            verbose=0,
        )
    else:
        n_samples = N_TRAINING_BATCHES * BATCH_SIZE
        started = time.monotonic()
        print(f"[global] simulating {n_samples} training datasets", flush=True)
        training_data = make_global_training_data(n_samples)
        print(f"[global] sim_data shape={training_data['sim_data'].shape}", flush=True)
        simulation_elapsed = time.monotonic() - started

        started = time.monotonic()
        history = workflow_global.fit_offline(
            training_data,
            epochs=GLOBAL_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[EpochProgressCallback("global").callback],
            verbose=0,
        )

    training_elapsed = time.monotonic() - started
    metrics = summarize_history(history)

    save_json(
        save_dir / "global_history.json",
        {
            "metrics": metrics,
            "online_training": ONLINE_TRAINING,
            "online_batches_per_epoch": ONLINE_BATCHES_PER_EPOCH if ONLINE_TRAINING else None,
            "samples_per_epoch": n_samples_per_epoch if ONLINE_TRAINING else N_TRAINING_BATCHES * BATCH_SIZE,
            "planned_unique_samples": n_samples_per_epoch * GLOBAL_EPOCHS
            if ONLINE_TRAINING
            else N_TRAINING_BATCHES * BATCH_SIZE,
            "simulation_elapsed_seconds": simulation_elapsed,
            "training_elapsed_seconds": training_elapsed,
        },
    )
    keras.saving.save_model(workflow_global.approximator, model_path)
    print(f"saved {model_path}", flush=True)


def fit_or_load_local(workflow_local, save_dir: Path):
    model_path = save_dir / "local_approximator.keras"
    if REUSE_SAVED_MODELS and model_path.exists():
        print(f"[local] loading saved approximator from {model_path}", flush=True)
        workflow_local.approximator = keras.saving.load_model(model_path)
        return

    n_samples_per_epoch = ONLINE_BATCHES_PER_EPOCH * BATCH_SIZE
    simulation_elapsed = 0.0

    if ONLINE_TRAINING:
        print(
            f"[local] online training with {n_samples_per_epoch} fresh datasets per epoch",
            flush=True,
        )
        print(f"[local] per-dataset sim_data shape=({N_TRIALS}, 3)", flush=True)
        workflow_local.simulator = BufferedOnlineSimulator("local", ONLINE_BATCHES_PER_EPOCH)
        started = time.monotonic()
        history = workflow_local.fit_online(
            epochs=LOCAL_EPOCHS,
            num_batches_per_epoch=ONLINE_BATCHES_PER_EPOCH,
            batch_size=BATCH_SIZE,
            callbacks=[EpochProgressCallback("local").callback],
            verbose=0,
        )
    else:
        n_samples = N_TRAINING_BATCHES * BATCH_SIZE
        started = time.monotonic()
        print(f"[local] simulating {n_samples} training datasets", flush=True)
        training_data = make_local_training_data(n_samples)
        print(f"[local] sim_data shape={training_data['sim_data'].shape}", flush=True)
        simulation_elapsed = time.monotonic() - started

        started = time.monotonic()
        history = workflow_local.fit_offline(
            training_data,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[EpochProgressCallback("local").callback],
            verbose=0,
        )

    training_elapsed = time.monotonic() - started
    metrics = summarize_history(history)

    save_json(
        save_dir / "local_history.json",
        {
            "metrics": metrics,
            "online_training": ONLINE_TRAINING,
            "online_batches_per_epoch": ONLINE_BATCHES_PER_EPOCH if ONLINE_TRAINING else None,
            "samples_per_epoch": n_samples_per_epoch if ONLINE_TRAINING else N_TRAINING_BATCHES * BATCH_SIZE,
            "planned_unique_samples": n_samples_per_epoch * LOCAL_EPOCHS
            if ONLINE_TRAINING
            else N_TRAINING_BATCHES * BATCH_SIZE,
            "simulation_elapsed_seconds": simulation_elapsed,
            "training_elapsed_seconds": training_elapsed,
        },
    )
    keras.saving.save_model(workflow_local.approximator, model_path)
    print(f"saved {model_path}", flush=True)


def run_tutorial():
    load_runtime()
    save_dir = SAVE_DIR
    figures_dir = save_dir / "figures"
    arrays_dir = save_dir / "arrays"
    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        save_dir / "run_config.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "tensorflow",
            "model": "hierarchical_conflict_ddm",
            "n_training_batches": N_TRAINING_BATCHES,
            "batch_size": BATCH_SIZE,
            "online_training": ONLINE_TRAINING,
            "online_batches_per_epoch": ONLINE_BATCHES_PER_EPOCH if ONLINE_TRAINING else None,
            "global_samples_per_epoch": ONLINE_BATCHES_PER_EPOCH * BATCH_SIZE
            if ONLINE_TRAINING
            else N_TRAINING_BATCHES * BATCH_SIZE,
            "global_planned_unique_samples": ONLINE_BATCHES_PER_EPOCH * BATCH_SIZE * GLOBAL_EPOCHS
            if ONLINE_TRAINING
            else N_TRAINING_BATCHES * BATCH_SIZE,
            "local_planned_unique_samples": ONLINE_BATCHES_PER_EPOCH * BATCH_SIZE * LOCAL_EPOCHS
            if ONLINE_TRAINING
            else N_TRAINING_BATCHES * BATCH_SIZE,
            "global_epochs": GLOBAL_EPOCHS,
            "local_epochs": LOCAL_EPOCHS,
            "global_summary_network_kwargs": GLOBAL_SUMMARY_NETWORK_KWARGS,
            "global_inference_network_kwargs": GLOBAL_INFERENCE_NETWORK_KWARGS,
            "n_local_subjects": N_LOCAL_SUBJECTS,
            "n_trials": N_TRIALS,
            "n_subjects": N_SUBJECTS,
            "n_test": N_TEST,
            "n_samples": N_SAMPLES,
            "compositional_summary_batch_size": COMPOSITIONAL_SUMMARY_BATCH_SIZE,
            "compositional_dataset_batch_size": COMPOSITIONAL_DATASET_BATCH_SIZE,
            "reuse_saved_models": REUSE_SAVED_MODELS,
            "parallel_workers": PARALLEL_WORKERS,
        },
    )

    workflow_global, workflow_local = make_workflows()

    fit_or_load_global(workflow_global, save_dir)
    fit_or_load_local(workflow_local, save_dir)

    print("[same-scale] simulating test data", flush=True)
    test_data_single = sample_data(N_TEST, n_subjects=N_LOCAL_SUBJECTS, n_trials=N_TRIALS)
    test_data_single["sim_data"] = test_data_single["sim_data"].reshape(
        N_TEST,
        N_LOCAL_SUBJECTS * N_TRIALS,
        3,
    )

    print("[same-scale] sampling global posterior", flush=True)
    test_posterior = workflow_global.sample(
        num_samples=N_SAMPLES,
        conditions={"sim_data": test_data_single["sim_data"]},
        batch_size=BATCH_SIZE,
    )
    fig = bf.diagnostics.recovery(
        estimates=test_posterior,
        targets=test_data_single,
        variable_names=PRETTY_PARAM_NAMES_GLOBAL,
    )
    save_figure(figures_dir / "same_scale_global_recovery.png", fig)
    fig = bf.diagnostics.calibration_ecdf(
        estimates=test_posterior,
        targets=test_data_single,
        variable_names=PRETTY_PARAM_NAMES_GLOBAL,
    )
    save_figure(figures_dir / "same_scale_global_calibration_ecdf.png", fig)
    save_npz_dict(
        arrays_dir / "same_scale_global.npz",
        estimates=test_posterior,
        targets={name: test_data_single[name] for name in PARAM_NAMES_GLOBAL},
    )

    print("[compositional] simulating large test data", flush=True)
    test_data = sample_data(N_TEST, n_subjects=N_SUBJECTS, n_trials=N_TRIALS)
    test_data["sim_data"] = test_data["sim_data"].reshape(
        N_TEST,
        N_SUBJECTS // N_LOCAL_SUBJECTS,
        N_TRIALS * N_LOCAL_SUBJECTS,
        3,
    )

    print("[compositional] sampling global posterior", flush=True)
    global_posterior = compositional_sample_batched(
        workflow_global,
        conditions={"sim_data": test_data["sim_data"]},
        num_samples=N_SAMPLES,
        compute_prior_score=prior_global_score,
        method="two_step_adaptive",
        steps="adaptive",
        mini_batch_size=MINI_BATCH_SIZE,
    )
    fig = bf.diagnostics.recovery(
        estimates=global_posterior,
        targets=test_data,
        variable_names=PRETTY_PARAM_NAMES_GLOBAL,
    )
    save_figure(figures_dir / "compositional_global_recovery.png", fig)
    save_npz_dict(
        arrays_dir / "compositional_global.npz",
        estimates=global_posterior,
        targets={name: test_data[name] for name in PARAM_NAMES_GLOBAL},
    )

    print("[local] sampling ancestral posterior", flush=True)
    smaller_test_data = {
        key: value[:, :N_LOCAL_CONDITIONS]
        for key, value in test_data.items()
        if key in PARAM_NAMES_LOCAL
    }
    smaller_test_data["sim_data"] = test_data["sim_data"].reshape(
        N_TEST,
        N_SUBJECTS,
        N_TRIALS,
        3,
    )[:, :N_LOCAL_CONDITIONS]

    local_posterior = workflow_local.ancestral_sample(
        conditions={"sim_data": smaller_test_data["sim_data"]},
        ancestral_conditions=global_posterior,
        batch_size=BATCH_SIZE * N_SAMPLES,
    )
    flat_local_posterior = {
        key: value.reshape(N_TEST * N_LOCAL_CONDITIONS, N_SAMPLES, 1)
        for key, value in local_posterior.items()
    }
    flat_test_data = {
        key: value.reshape(N_TEST * N_LOCAL_CONDITIONS, 1)
        for key, value in smaller_test_data.items()
        if key != "sim_data"
    }
    fig = bf.diagnostics.recovery(
        estimates=flat_local_posterior,
        targets=flat_test_data,
        variable_names=PRETTY_PARAM_NAMES_LOCAL,
    )
    save_figure(figures_dir / "local_recovery.png", fig)
    save_npz_dict(
        arrays_dir / "local_ancestral.npz",
        estimates=flat_local_posterior,
        targets=flat_test_data,
    )

    print(f"complete. Results saved in {save_dir}", flush=True)


if __name__ == "__main__":
    run_tutorial()
