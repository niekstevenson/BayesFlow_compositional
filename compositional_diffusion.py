"""Run the compositional diffusion tutorial with TensorFlow on the GPU.

Usage:
    conda run --no-capture-output -n tf-gpu python -u compositional_diffusion.py
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
from scipy.stats import beta as beta_dist
from scipy.stats import norm as norm_dist


# Tutorial defaults. Edit these constants if you want a smaller or larger run.
N_TRAINING_BATCHES = 256
BATCH_SIZE = 64
GLOBAL_EPOCHS = 1_000
LOCAL_EPOCHS = 50
N_LOCAL_SUBJECTS = 5
N_TRIALS = 30
N_SUBJECTS = 10_000
N_TEST = 100
N_SAMPLES = 100
N_LOCAL_CONDITIONS = 10
MINI_BATCH_SIZE = 10
COMPOSITIONAL_BATCH_SIZE = BATCH_SIZE * 10
PARALLEL_WORKERS = 8
SAVE_DIR = Path("results/compositional_diffusion_tutorial")

GLOBAL_PRIOR = {
    "mu_nu": (0.5, 0.3),
    "mu_alpha": (0.0, 0.05),
    "mu_t0": (-1.0, 0.3),
    "log_sigma_nu": (-1.0, 1.0),
    "log_sigma_alpha": (-3.0, 1.0),
    "log_sigma_t0": (-1.0, 0.3),
    "beta_raw": (0.0, 1.0),
}

PARAM_NAMES_GLOBAL = list(GLOBAL_PRIOR)
PARAM_NAMES_LOCAL = ["nu", "alpha", "t0"]
PRETTY_PARAM_NAMES_GLOBAL = [
    r"$\mu_\nu$",
    r"$\mu_\alpha$",
    r"$\mu_{t_0}$",
    r"$\log \sigma_\nu$",
    r"$\log \sigma_\alpha$",
    r"$\log \sigma_{t_0}$",
    r"$\beta_\text{raw}$",
]
PRETTY_PARAM_NAMES_LOCAL = [r"$\nu_p$", r"$\alpha_p$", r"$t_{0,p}$"]

bf = None
keras = None


@njit
def seed_numba(seed: int) -> None:
    np.random.seed(seed)


@njit
def simulate_ddm_trial(nu, alpha, t0, beta, dt=1e-3, scale=1.0, max_time=10.0):
    y = beta * alpha
    rt = t0
    const = scale * np.sqrt(dt)

    while (alpha >= y >= 0.0) and rt <= max_time:
        y += nu * dt + const * np.random.randn()
        rt += dt

    choice = 1.0 if y >= alpha else 0.0
    return choice, rt


def simulate_ddm(nu, alpha, t0, beta, n_subjects=1, n_trials=1):
    if np.isscalar(nu):
        nu = np.ones((n_subjects,)) * nu
        alpha = np.ones((n_subjects,)) * alpha
        t0 = np.ones((n_subjects,)) * t0

    data = np.zeros((n_subjects, n_trials, 2), dtype=np.float64)
    for j_subject in range(n_subjects):
        for i_trial in range(n_trials):
            data[j_subject, i_trial] = simulate_ddm_trial(
                nu[j_subject],
                alpha[j_subject],
                t0[j_subject],
                beta,
            )

    if n_subjects == 1 and n_trials == 1:
        data = data[0, 0]
    elif n_subjects == 1:
        data = data[0]
    elif n_trials == 1:
        data = data[:, 0]

    return {"sim_data": data}


def score_normal(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return -(x - mean) / sd**2


def beta_from_normal(z, a, b):
    return beta_dist.ppf(norm_dist.cdf(z), a, b)


def sample_hierarchical_priors(n_subjects=1):
    mu_nu = np.random.normal(*GLOBAL_PRIOR["mu_nu"])
    mu_alpha = np.random.normal(*GLOBAL_PRIOR["mu_alpha"])
    mu_t0 = np.random.normal(*GLOBAL_PRIOR["mu_t0"])

    log_sigma_nu = np.random.normal(*GLOBAL_PRIOR["log_sigma_nu"])
    log_sigma_alpha = np.random.normal(*GLOBAL_PRIOR["log_sigma_alpha"])
    log_sigma_t0 = np.random.normal(*GLOBAL_PRIOR["log_sigma_t0"])

    beta_raw = np.random.normal(*GLOBAL_PRIOR["beta_raw"])
    beta = beta_from_normal(beta_raw, a=50, b=50)

    nu = np.random.normal(mu_nu, np.exp(log_sigma_nu), size=n_subjects)
    alpha = np.exp(np.random.normal(mu_alpha, np.exp(log_sigma_alpha), size=n_subjects))
    t0 = np.exp(np.random.normal(mu_t0, np.exp(log_sigma_t0), size=n_subjects))

    return {
        "mu_nu": mu_nu,
        "mu_alpha": mu_alpha,
        "mu_t0": mu_t0,
        "log_sigma_nu": log_sigma_nu,
        "log_sigma_alpha": log_sigma_alpha,
        "log_sigma_t0": log_sigma_t0,
        "beta_raw": beta_raw,
        "beta": beta,
        "nu": nu,
        "alpha": alpha,
        "t0": t0,
    }


def prior_global_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: score_normal(x[name], mean=GLOBAL_PRIOR[name][0], sd=GLOBAL_PRIOR[name][1])
        for name in PARAM_NAMES_GLOBAL
    }


def simulate_one_dataset(n_subjects=1, n_trials=1):
    params = sample_hierarchical_priors(n_subjects=n_subjects)
    data = simulate_ddm(
        params["nu"],
        params["alpha"],
        params["t0"],
        params["beta"],
        n_subjects=n_subjects,
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


def sample_serial(n_samples: int, n_subjects=1, n_trials=1, seed: int | None = None):
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


def sample_parallel(n_samples: int, n_subjects=1, n_trials=1, workers=PARALLEL_WORKERS):
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


def sample_data(n_samples: int, n_subjects=1, n_trials=1):
    return sample_parallel(
        n_samples,
        n_subjects=n_subjects,
        n_trials=n_trials,
        workers=PARALLEL_WORKERS,
    )


def make_global_training_data(n_samples: int):
    data = sample_data(n_samples, n_subjects=N_LOCAL_SUBJECTS, n_trials=N_TRIALS)
    data["sim_data"] = data["sim_data"].reshape(n_samples, N_LOCAL_SUBJECTS * N_TRIALS, 2)
    return data


def make_local_training_data(n_samples: int):
    return sample_data(n_samples, n_trials=N_TRIALS)


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
        summary_network=bf.networks.SetTransformer(summary_dim=16),
        inference_network=bf.networks.DiffusionModel(),
    )

    adapter_local = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .log(["alpha", "t0"])
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


def fit_global(workflow_global, save_dir: Path):
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
            "simulation_elapsed_seconds": simulation_elapsed,
            "training_elapsed_seconds": training_elapsed,
        },
    )
    keras.saving.save_model(workflow_global.approximator, save_dir / "global_approximator.keras")
    print(f"saved {save_dir / 'global_approximator.keras'}", flush=True)


def fit_local(workflow_local, save_dir: Path):
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
            "simulation_elapsed_seconds": simulation_elapsed,
            "training_elapsed_seconds": training_elapsed,
        },
    )
    keras.saving.save_model(workflow_local.approximator, save_dir / "local_approximator.keras")
    print(f"saved {save_dir / 'local_approximator.keras'}", flush=True)


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
            "n_training_batches": N_TRAINING_BATCHES,
            "batch_size": BATCH_SIZE,
            "global_epochs": GLOBAL_EPOCHS,
            "local_epochs": LOCAL_EPOCHS,
            "n_local_subjects": N_LOCAL_SUBJECTS,
            "n_trials": N_TRIALS,
            "n_subjects": N_SUBJECTS,
            "n_test": N_TEST,
            "n_samples": N_SAMPLES,
            "parallel_workers": PARALLEL_WORKERS,
        },
    )

    workflow_global, workflow_local = make_workflows()

    fit_global(workflow_global, save_dir)
    fit_local(workflow_local, save_dir)

    print("[same-scale] simulating test data", flush=True)
    test_data_single = sample_data(N_TEST, n_subjects=N_LOCAL_SUBJECTS, n_trials=N_TRIALS)
    test_data_single["sim_data"] = test_data_single["sim_data"].reshape(
        N_TEST,
        N_LOCAL_SUBJECTS * N_TRIALS,
        2,
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
        2,
    )

    print("[compositional] sampling global posterior", flush=True)
    global_posterior = workflow_global.compositional_sample(
        num_samples=N_SAMPLES,
        conditions={"sim_data": test_data["sim_data"]},
        compute_prior_score=prior_global_score,
        method="two_step_adaptive",
        steps="adaptive",
        mini_batch_size=MINI_BATCH_SIZE,
        batch_size=COMPOSITIONAL_BATCH_SIZE,
        return_summaries=True,
    )
    global_posterior.pop("_summaries", None)
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
        2,
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
