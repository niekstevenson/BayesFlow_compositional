"""Variable-trial hierarchical DDM training script.

This file intentionally contains one model only: a hierarchical diffusion
decision model with subject-level condition effects and variable trial counts.

Generative structure:

    Group level:
        eta_k ~ Normal(m_k, s_k)

    Subject level:
        v_p        ~ Normal(mu_v, exp(log_sigma_v))
        v_cond_p   ~ Normal(mu_v_cond, exp(log_sigma_v_cond))
        log_a_p    ~ Normal(mu_log_a, exp(log_sigma_log_a))
        log_t0_p   ~ Normal(mu_log_t0, exp(log_sigma_log_t0))
        log_sv_p   ~ Normal(mu_log_sv, exp(log_sigma_log_sv))

    Trial level:
        x_pt in {-0.5, +0.5}, balanced per subject
        trial_v_pt = v_p + v_cond_p * x_pt + exp(log_sv_p) * Normal(0, 1)

The DDM uses fixed diffusion scale 1 and fixed starting point z = 0.5.
Each observation is (choice, reaction_time, condition). Variable-length trial
sets are padded to max_trials and accompanied by valid_mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

logging.disable(logging.WARNING)
import bayesflow as bf
logging.disable(logging.NOTSET)
logging.getLogger("bayesflow").setLevel(logging.ERROR)
logging.getLogger("numexpr").setLevel(logging.WARNING)
import keras
import numpy as np
from numba import njit


DEFAULT_BATCH_SIZE = 8
DEFAULT_GLOBAL_EPOCHS = 1_000
DEFAULT_LOCAL_EPOCHS = 50

DEFAULT_MIN_TRIALS = 60
DEFAULT_MAX_TRIALS = 320
DEFAULT_N_TRAIN = 5_120
DEFAULT_N_VAL = 2_048
DEFAULT_N_SUBJECTS = 5


GROUP_PRIOR = {
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

GROUP_PARAMETERS = list(GROUP_PRIOR)
SUBJECT_PARAMETERS = ["v", "v_cond", "log_a", "log_t0", "log_sv"]


@njit
def seed_numba(seed):
    np.random.seed(seed)


@njit
def simulate_ddm_trial(drift, boundary, non_decision_time, start_fraction=0.5, dt=1e-3, max_time=10.0):
    """Simulate one DDM trial."""
    evidence = start_fraction * boundary
    rt = non_decision_time
    noise_scale = np.sqrt(dt)

    while (boundary >= evidence >= 0.0) and rt <= max_time:
        evidence += drift * dt + noise_scale * np.random.randn()
        rt += dt

    choice = 1.0 if evidence >= boundary else 0.0
    return choice, rt


def score_log_normal(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Score of log Normal(mean, sd^2) for an unconstrained Gaussian variable."""
    return -(x - mean) / sd**2


def sample_hierarchical_priors(n_subjects=1):
    """Sample one full hierarchy: group parameters and subject parameters."""
    mu_v = np.random.normal(*GROUP_PRIOR["mu_v"])
    mu_v_cond = np.random.normal(*GROUP_PRIOR["mu_v_cond"])
    mu_log_a = np.random.normal(*GROUP_PRIOR["mu_log_a"])
    mu_log_t0 = np.random.normal(*GROUP_PRIOR["mu_log_t0"])
    mu_log_sv = np.random.normal(*GROUP_PRIOR["mu_log_sv"])

    log_sigma_v = np.random.normal(*GROUP_PRIOR["log_sigma_v"])
    log_sigma_v_cond = np.random.normal(*GROUP_PRIOR["log_sigma_v_cond"])
    log_sigma_log_a = np.random.normal(*GROUP_PRIOR["log_sigma_log_a"])
    log_sigma_log_t0 = np.random.normal(*GROUP_PRIOR["log_sigma_log_t0"])
    log_sigma_log_sv = np.random.normal(*GROUP_PRIOR["log_sigma_log_sv"])

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


@njit
def simulate_subjects(v, v_cond, log_a, log_t0, log_sv, n_trials, sim_data, valid_mask, dt, max_time):
    for i_subject in range(v.shape[0]):
        n = n_trials[i_subject]
        half = n // 2

        condition = np.empty(n)
        for i_trial in range(half):
            condition[i_trial] = 0.5
        for i_trial in range(half, n):
            condition[i_trial] = -0.5
        np.random.shuffle(condition)

        boundary = np.exp(log_a[i_subject])
        non_decision_time = np.exp(log_t0[i_subject])
        drift_sd = np.exp(log_sv[i_subject])

        for i_trial in range(n):
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
            valid_mask[i_subject, i_trial] = True


def draw_even_trial_counts(n_subjects, min_trials=DEFAULT_MIN_TRIALS, max_trials=DEFAULT_MAX_TRIALS):
    """Draw balanced trial counts. Even counts guarantee equal condition counts."""
    if min_trials < 2:
        raise ValueError("min_trials must be at least 2.")
    if max_trials < min_trials:
        raise ValueError("max_trials must be >= min_trials.")

    low = math.ceil(min_trials / 2)
    high = math.floor(max_trials / 2)
    if high < low:
        raise ValueError("No even trial count is available in the requested range.")

    return (2 * np.random.randint(low, high + 1, size=n_subjects)).astype(np.int32)


def simulate_hierarchical_ddm(
    v,
    v_cond,
    log_a,
    log_t0,
    log_sv,
    min_trials=DEFAULT_MIN_TRIALS,
    max_trials=DEFAULT_MAX_TRIALS,
    dt=1e-3,
    max_time=10.0,
):
    """Simulate padded subject-level trial sets for one hierarchy."""
    v = np.asarray(v)
    n_subjects = v.shape[0]
    n_trials = draw_even_trial_counts(n_subjects, min_trials=min_trials, max_trials=max_trials)
    sim_data = np.zeros((n_subjects, max_trials, 3), dtype=np.float32)
    valid_mask = np.zeros((n_subjects, max_trials), dtype=bool)

    simulate_subjects(
        v,
        np.asarray(v_cond),
        np.asarray(log_a),
        np.asarray(log_t0),
        np.asarray(log_sv),
        n_trials,
        sim_data,
        valid_mask,
        dt,
        max_time,
    )

    return {"sim_data": sim_data, "valid_mask": valid_mask, "n_trials": n_trials}


def group_prior_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Score of the independent Gaussian group prior."""
    return {
        name: score_log_normal(x[name], mean=GROUP_PRIOR[name][0], sd=GROUP_PRIOR[name][1])
        for name in GROUP_PARAMETERS
    }


simulator = bf.make_simulator([sample_hierarchical_priors, simulate_hierarchical_ddm])


@keras.saving.register_keras_serializable(package="BayesFlowComp")
class MaskedSetTransformer(bf.networks.SetTransformer):
    """SetTransformer with masks carried into pooling-by-attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, previous_mask=None):
        return None

    def call(self, x, training=False, attention_mask=None, mask=None):
        valid = None

        if mask is not None:
            valid = keras.ops.cast(mask, "bool")
            attention_mask = self._self_attention_mask(valid)
        elif attention_mask is not None:
            attention_mask = keras.ops.cast(attention_mask, "bool")
            if len(keras.ops.shape(attention_mask)) == 2:
                valid = attention_mask
                attention_mask = self._self_attention_mask(valid)
            else:
                valid = keras.ops.any(attention_mask, axis=-1)

        valid_float = None
        if valid is not None:
            valid_float = keras.ops.cast(keras.ops.expand_dims(valid, axis=-1), x.dtype)
            x = x * valid_float

        for layer in self.attention_blocks:
            x = layer(x, training=training, attention_mask=attention_mask)
            if valid_float is not None:
                x = x * valid_float

        if valid is None:
            x = self.pooling_by_attention(x, training=training)
        else:
            x = self._masked_pooling(x, valid, valid_float, training=training)

        return self.output_projector(x)

    @staticmethod
    def _self_attention_mask(valid):
        query_mask = keras.ops.expand_dims(valid, axis=-1)
        key_mask = keras.ops.expand_dims(valid, axis=-2)
        return keras.ops.logical_and(query_mask, key_mask)

    def _masked_pooling(self, x, valid, valid_float, training=False):
        transformed = self.pooling_by_attention.feedforward(x, training=training)
        transformed = transformed * valid_float

        batch_size = keras.ops.shape(x)[0]
        seeds = keras.ops.tile(
            keras.ops.expand_dims(self.pooling_by_attention.seed_vector, axis=0),
            [batch_size, 1, 1],
        )
        pool_mask = keras.ops.tile(
            keras.ops.expand_dims(valid, axis=1),
            [1, self.pooling_by_attention.num_seeds, 1],
        )
        summaries = self.pooling_by_attention.mab(
            seeds,
            transformed,
            training=training,
            attention_mask=pool_mask,
        )
        return keras.ops.reshape(summaries, (keras.ops.shape(summaries)[0], -1))


global_adapter = bf.approximators.ContinuousApproximator.build_adapter(
    inference_variables=GROUP_PARAMETERS,
    summary_variables="sim_data",
    summary_mask="valid_mask",
)

local_adapter = bf.approximators.ContinuousApproximator.build_adapter(
    inference_variables=SUBJECT_PARAMETERS,
    inference_conditions=GROUP_PARAMETERS,
    summary_variables="sim_data",
    summary_mask="valid_mask",
)

global_workflow = bf.CompositionalWorkflow(
    adapter=global_adapter,
    summary_network=MaskedSetTransformer(
        summary_dim=64,
        embed_dims=(96, 96, 96),
        num_heads=(4, 4, 4),
        num_seeds=4,
        dropout=0.05,
    ),
    inference_network=bf.networks.DiffusionModel(
        subnet_kwargs={
            "widths": (256, 256, 256, 256, 256),
            "dropout": 0.05,
        },
    ),
)

local_workflow = bf.BasicWorkflow(
    adapter=local_adapter,
    summary_network=MaskedSetTransformer(summary_dim=32, num_seeds=2),
    inference_network=bf.networks.CouplingFlow(),
)

WORKFLOWS = {
    "global": global_workflow,
    "local": local_workflow,
}


class ProgressCallback(keras.callbacks.Callback):
    """Print compact progress and stop cleanly at the time budget."""

    def __init__(
        self,
        label,
        max_seconds,
        report_every_seconds=30.0,
        report_every_epochs=1,
        start_epoch=0,
        initial_best_val_loss=math.inf,
    ):
        super().__init__()
        self.label = label
        self.max_seconds = max_seconds
        self.report_every_seconds = report_every_seconds
        self.report_every_epochs = report_every_epochs
        self.start_epoch = start_epoch
        self.started_at = None
        self.last_report_at = None
        self.first_loss = None
        self.best_val_loss = initial_best_val_loss

    def on_train_begin(self, logs=None):
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        print(f"[{self.label}] training started")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        now = time.monotonic()
        elapsed = now - self.started_at
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if loss is not None and self.first_loss is None:
            self.first_loss = float(loss)
        if val_loss is not None:
            self.best_val_loss = min(self.best_val_loss, float(val_loss))

        due_by_time = (now - self.last_report_at) >= self.report_every_seconds
        due_by_epoch = (epoch + 1) % self.report_every_epochs == 0
        due_by_end = elapsed >= self.max_seconds

        if due_by_time or due_by_epoch or due_by_end:
            parts = [f"[{self.label}] epoch={self.start_epoch + epoch + 1}", f"elapsed={elapsed:.1f}s"]
            if loss is not None:
                parts.append(f"loss={float(loss):.4g}")
                parts.append(f"loss_delta={float(loss) - self.first_loss:+.4g}")
            if val_loss is not None:
                parts.append(f"val_loss={float(val_loss):.4g}")
                parts.append(f"best_val={self.best_val_loss:.4g}")
            print(" ".join(parts))
            self.last_report_at = now

        if loss is not None and not np.isfinite(float(loss)):
            print(f"[{self.label}] stopping: non-finite loss")
            self.model.stop_training = True
        elif elapsed >= self.max_seconds:
            print(f"[{self.label}] stopping: reached {self.max_seconds:.0f}s budget")
            self.model.stop_training = True


class EpochCheckpointCallback(keras.callbacks.Callback):
    """Persist enough state to resume after every completed epoch."""

    def __init__(
        self,
        workflow,
        label,
        checkpoint_path,
        state_path,
        history_path,
        state_factory,
        start_epoch,
        best_checkpoint_path=None,
        best_state_path=None,
        initial_best_val_loss=math.inf,
    ):
        super().__init__()
        self.workflow = workflow
        self.label = label
        self.checkpoint_path = Path(checkpoint_path)
        self.state_path = Path(state_path)
        self.history_path = Path(history_path)
        self.state_factory = state_factory
        self.start_epoch = start_epoch
        self.best_checkpoint_path = Path(best_checkpoint_path) if best_checkpoint_path else None
        self.best_state_path = Path(best_state_path) if best_state_path else None
        self.best_val_loss = initial_best_val_loss
        self.saved_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        completed_epoch = self.start_epoch + epoch
        val_loss = logs.get("val_loss") if logs else None
        is_best = False
        if val_loss is not None and np.isfinite(float(val_loss)) and float(val_loss) < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            is_best = True

        append_history(self.history_path, self.label, completed_epoch, logs or {})
        save_training_state(self.state_path, self.state_factory(completed_epoch, self.best_val_loss), quiet=True)
        save_checkpoint(self.workflow, self.checkpoint_path, quiet=True)
        if is_best and self.best_checkpoint_path is not None and self.best_state_path is not None:
            save_training_state(self.best_state_path, self.state_factory(completed_epoch, self.best_val_loss), quiet=True)
            save_checkpoint(self.workflow, self.best_checkpoint_path, quiet=True)
        clear_torch_mps_cache()
        self.saved_epochs += 1


def workflow_for_label(label):
    return WORKFLOWS[label]


def default_checkpoint_path(label):
    return Path("checkpoints") / f"hierarchical_{label}_latest.keras"


def default_best_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")


def default_data_cache_path(checkpoint_path):
    return Path(checkpoint_path).with_suffix(".data.npz")


def default_validation_cache_path(checkpoint_path):
    return Path(checkpoint_path).with_suffix(".val.npz")


def default_state_path(checkpoint_path):
    return Path(checkpoint_path).with_suffix(".state.json")


def default_history_path(checkpoint_path):
    return Path(checkpoint_path).with_suffix(".history.csv")


def save_checkpoint(workflow, checkpoint_path, quiet=False):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    workflow.approximator.save(checkpoint_path)
    if not quiet:
        print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(workflow, checkpoint_path, keep_optimizer):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    workflow.approximator = keras.saving.load_model(checkpoint_path, compile=keep_optimizer)
    if keep_optimizer:
        workflow.optimizer = getattr(workflow.approximator, "optimizer", None)
        workflow._needs_compile = workflow.optimizer is None
    else:
        workflow.optimizer = None
        workflow._needs_compile = True
    print(f"Loaded checkpoint: {checkpoint_path}")


def save_training_state(state_path, state, quiet=False):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    if not quiet:
        print(f"Saved training state: {state_path}")


def load_training_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        return {}

    with state_path.open() as f:
        state = json.load(f)
    print(f"Loaded training state: {state_path}")
    return state


def save_data_cache(data_cache_path, train_data, val_data, metadata):
    data_cache_path = Path(data_cache_path)
    data_cache_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {f"train__{key}": value for key, value in train_data.items()}
    if val_data is not None:
        arrays.update({f"val__{key}": value for key, value in val_data.items()})
    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez_compressed(data_cache_path, **arrays)
    print(f"Saved data cache: {data_cache_path}")


def load_data_cache(data_cache_path):
    data_cache_path = Path(data_cache_path)
    if not data_cache_path.exists():
        raise FileNotFoundError(f"Data cache does not exist: {data_cache_path}")

    with np.load(data_cache_path, allow_pickle=False) as arrays:
        train_data = {
            key.removeprefix("train__"): arrays[key]
            for key in arrays.files
            if key.startswith("train__")
        }
        val_data = {
            key.removeprefix("val__"): arrays[key]
            for key in arrays.files
            if key.startswith("val__")
        }
        metadata = json.loads(arrays["metadata"].item()) if "metadata" in arrays.files else {}

    print(f"Loaded data cache: {data_cache_path}")
    return train_data, val_data or None, metadata


def append_history(history_path, label, completed_epoch, logs):
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = [name for name, value in logs.items() if is_float_like(value)]
    if not metric_names:
        return

    write_header = not history_path.exists()
    if write_header:
        fieldnames = ["workflow", "epoch", *metric_names]
    else:
        with history_path.open(newline="") as f:
            reader = csv.reader(f)
            fieldnames = next(reader, None)
        if not fieldnames:
            write_header = True
            fieldnames = ["workflow", "epoch", *metric_names]

    with history_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = {"workflow": label, "epoch": completed_epoch}
        row.update({name: float(logs[name]) for name in metric_names})
        writer.writerow(row)


def is_float_like(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def clear_torch_mps_cache():
    if keras.backend.backend() != "torch":
        return

    try:
        import torch
    except ImportError:
        return

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sample_data(n_samples, parallel=True, **kwargs):
    sample_fn = simulator.sample_parallel if parallel else simulator.sample
    return sample_fn(n_samples, **kwargs)


def make_global_data(
    n_samples,
    n_subjects=DEFAULT_N_SUBJECTS,
    min_trials=DEFAULT_MIN_TRIALS,
    max_trials=DEFAULT_MAX_TRIALS,
    parallel=True,
    dt=1e-3,
    max_time=10.0,
):
    data = sample_data(
        n_samples,
        parallel=parallel,
        n_subjects=n_subjects,
        min_trials=min_trials,
        max_trials=max_trials,
        dt=dt,
        max_time=max_time,
    )
    data["sim_data"] = data["sim_data"].reshape(n_samples, n_subjects * max_trials, 3)
    data["valid_mask"] = data["valid_mask"].reshape(n_samples, n_subjects * max_trials)
    return data


def make_local_data(
    n_samples,
    min_trials=DEFAULT_MIN_TRIALS,
    max_trials=DEFAULT_MAX_TRIALS,
    parallel=True,
    dt=1e-3,
    max_time=10.0,
):
    data = sample_data(
        n_samples,
        parallel=parallel,
        n_subjects=1,
        min_trials=min_trials,
        max_trials=max_trials,
        dt=dt,
        max_time=max_time,
    )
    data["sim_data"] = data["sim_data"][:, 0]
    data["valid_mask"] = data["valid_mask"][:, 0]
    data["n_trials"] = data["n_trials"][:, 0]
    return data


class BufferedHierarchicalSimulator:
    """Generate fresh epoch-sized data buffers for BayesFlow online training."""

    def __init__(self, label, args, n_batches_per_epoch):
        self.label = label
        self.args = args
        self.n_batches_per_epoch = n_batches_per_epoch
        self.buffer = None
        self.cursor = 0

    def sample(self, batch_size):
        if self.buffer is None or self.cursor + batch_size > self._buffer_size():
            n_samples = self.n_batches_per_epoch * batch_size
            make_data = make_global_data if self.label == "global" else make_local_data
            kwargs = {
                "n_samples": n_samples,
                "min_trials": self.args.min_trials,
                "max_trials": self.args.max_trials,
                "parallel": False,
                "dt": self.args.dt,
                "max_time": self.args.max_time,
            }
            if self.label == "global":
                kwargs["n_subjects"] = self.args.n_subjects
            self.buffer = make_data(**kwargs)
            self.cursor = 0

        start = self.cursor
        stop = start + batch_size
        self.cursor = stop
        return {key: value[start:stop] for key, value in self.buffer.items()}

    def _buffer_size(self):
        return len(next(iter(self.buffer.values())))


def make_training_data(label, args):
    make_data = make_global_data if label == "global" else make_local_data
    kwargs = {
        "n_samples": args.n_train,
        "min_trials": args.min_trials,
        "max_trials": args.max_trials,
        "parallel": not args.serial_simulation,
        "dt": args.dt,
        "max_time": args.max_time,
    }
    if label == "global":
        kwargs["n_subjects"] = args.n_subjects
    train_data = make_data(**kwargs)

    val_data = None
    if args.n_val > 0:
        kwargs["n_samples"] = args.n_val
        val_data = make_data(**kwargs)

    return train_data, val_data


def summarize_data(label, data):
    trial_counts = data["n_trials"]
    condition_sums = np.sum(data["sim_data"][..., 2] * data["valid_mask"], axis=-1)
    print(
        f"[{label}] sim_data={data['sim_data'].shape}; "
        f"valid_mask={data['valid_mask'].shape}; "
        f"n_trials min/mean/max={int(trial_counts.min())}/{float(trial_counts.mean()):.1f}/{int(trial_counts.max())}; "
        f"max_abs_condition_sum={np.max(np.abs(condition_sums)):.3g}"
    )


def summarize_history(label, history):
    metrics = {key: [float(value) for value in values] for key, values in history.history.items()}
    print(f"[{label}] history={metrics}")

    loss = metrics.get("loss", [])
    val_loss = metrics.get("val_loss", [])
    if not loss or not all(np.isfinite(loss)):
        print(f"[{label}] result=FAIL loss is missing or non-finite")
        return metrics

    loss_delta = loss[-1] - loss[0]
    if val_loss and all(np.isfinite(val_loss)):
        print(f"[{label}] result=OK loss_delta={loss_delta:+.4g}; val_delta={val_loss[-1] - val_loss[0]:+.4g}")
    else:
        print(f"[{label}] result=OK loss_delta={loss_delta:+.4g}")
    return metrics


def state_from_args(label, args, completed_epochs, best_val_loss=None):
    return {
        "workflow": f"hierarchical_{label}",
        "completed_epochs": completed_epochs,
        "target_epochs": args.epochs,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_subjects": args.n_subjects if label == "global" else 1,
        "min_trials": args.min_trials,
        "max_trials": args.max_trials,
        "batch_size": args.batch_size,
        "online": args.online,
        "best_val_loss": best_val_loss,
        "keep_optimizer": args.keep_optimizer,
        "backend": keras.backend.backend(),
        "dt": args.dt,
        "max_time": args.max_time,
    }


def metadata_from_args(label, args):
    return {
        "workflow": f"hierarchical_{label}",
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_subjects": args.n_subjects if label == "global" else 1,
        "min_trials": args.min_trials,
        "max_trials": args.max_trials,
        "dt": args.dt,
        "max_time": args.max_time,
    }


def validation_cache_matches(label, args, metadata):
    if not metadata:
        return False

    expected = metadata_from_args(label, args)
    for key in ["workflow", "n_val", "n_subjects", "min_trials", "max_trials", "dt", "max_time"]:
        if metadata.get(key) != expected.get(key):
            return False
    return True


def make_validation_data(label, args):
    if args.n_val <= 0:
        return None

    make_data = make_global_data if label == "global" else make_local_data
    kwargs = {
        "n_samples": args.n_val,
        "min_trials": args.min_trials,
        "max_trials": args.max_trials,
        "parallel": False,
        "dt": args.dt,
        "max_time": args.max_time,
    }
    if label == "global":
        kwargs["n_subjects"] = args.n_subjects
    return make_data(**kwargs)


def train_hierarchical(args):
    label = args.which
    workflow = workflow_for_label(label)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(label)
    data_cache_path = Path(args.data_cache) if args.data_cache else default_data_cache_path(checkpoint_path)
    val_cache_path = Path(args.data_cache) if args.data_cache else default_validation_cache_path(checkpoint_path)
    state_path = default_state_path(checkpoint_path)
    best_checkpoint_path = default_best_checkpoint_path(checkpoint_path)
    best_state_path = default_state_path(best_checkpoint_path)
    history_path = default_history_path(checkpoint_path)
    state = load_training_state(state_path) if args.resume else {}
    best_state = load_training_state(best_state_path) if args.resume and best_state_path.exists() else {}

    if args.epochs is None:
        default_epochs = DEFAULT_GLOBAL_EPOCHS if label == "global" else DEFAULT_LOCAL_EPOCHS
        args.epochs = int(state.get("target_epochs", default_epochs))
    if args.n_train is None:
        args.n_train = int(state.get("n_train", DEFAULT_N_TRAIN))
    if args.n_val is None:
        args.n_val = int(state.get("n_val", DEFAULT_N_VAL))
    if args.batch_size is None:
        args.batch_size = int(state.get("batch_size", DEFAULT_BATCH_SIZE))

    if checkpoint_path.exists() and args.resume:
        load_checkpoint(workflow, checkpoint_path, keep_optimizer=args.keep_optimizer)
    elif checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint exists: {checkpoint_path}. Use --resume or --overwrite.")
    elif args.resume:
        raise FileNotFoundError(f"Cannot resume because checkpoint does not exist: {checkpoint_path}")

    completed_epochs = int(state.get("completed_epochs", 0)) if args.resume else 0
    initial_best_val_loss = best_state.get("best_val_loss", state.get("best_val_loss", math.inf))
    if initial_best_val_loss is None:
        initial_best_val_loss = math.inf
    initial_best_val_loss = float(initial_best_val_loss)
    remaining_epochs = max(args.epochs - completed_epochs, 0)
    if remaining_epochs == 0:
        print(f"[hierarchical_{label}] target epochs already reached: {completed_epochs}/{args.epochs}")
        return

    started = time.monotonic()
    train_data = None
    if args.seed is not None:
        np.random.seed(args.seed)
        seed_numba(args.seed + 1)

    if args.online:
        val_data = None
        if val_cache_path.exists() and args.resume:
            _, cached_val_data, metadata = load_data_cache(val_cache_path)
            if cached_val_data is not None and validation_cache_matches(label, args, metadata):
                val_data = cached_val_data
                print(f"[hierarchical_{label}] validation metadata={metadata}")
            else:
                print(f"[hierarchical_{label}] regenerating validation cache: metadata mismatch")
        if val_data is None:
            val_data = make_validation_data(label, args)
            save_data_cache(val_cache_path, {}, val_data, metadata_from_args(label, args))

        n_batches_per_epoch = math.ceil(args.n_train / args.batch_size)
        workflow.simulator = BufferedHierarchicalSimulator(label, args, n_batches_per_epoch)
    else:
        if data_cache_path.exists() and args.resume:
            train_data, val_data, metadata = load_data_cache(data_cache_path)
            if metadata:
                print(f"[hierarchical_{label}] data metadata={metadata}")
        else:
            train_data, val_data = make_training_data(label, args)
            save_data_cache(data_cache_path, train_data, val_data, metadata_from_args(label, args))

    print(f"Backend: {keras.backend.backend()}")
    print(f"[hierarchical_{label}] checkpoint={checkpoint_path}")
    print(f"[hierarchical_{label}] best_checkpoint={best_checkpoint_path}")
    if args.online:
        print(f"[hierarchical_{label}] validation_cache={val_cache_path}")
    else:
        print(f"[hierarchical_{label}] data_cache={data_cache_path}")
    print(f"[hierarchical_{label}] history={history_path}")
    print(f"[hierarchical_{label}] simulation elapsed={time.monotonic() - started:.1f}s")
    if args.online:
        print(
            f"[hierarchical_{label}] training_data=fresh; "
            f"samples_per_epoch={n_batches_per_epoch * args.batch_size}; "
            f"batches_per_epoch={n_batches_per_epoch}"
        )
    else:
        summarize_data(f"hierarchical_{label}_train", train_data)
    if val_data is not None:
        summarize_data(f"hierarchical_{label}_val", val_data)
    else:
        print(f"[hierarchical_{label}] validation disabled")
    print(
        f"[hierarchical_{label}] fixed diffusion scale=1, z=0.5; "
        f"epochs completed={completed_epochs}/{args.epochs}; "
        f"this_run={remaining_epochs}; batch_size={args.batch_size}; "
        f"keep_optimizer={args.keep_optimizer}; online={args.online}"
    )

    progress = ProgressCallback(
        label=f"hierarchical_{label}",
        max_seconds=args.minutes * 60.0,
        report_every_seconds=args.report_every_seconds,
        report_every_epochs=args.report_every_epochs,
        start_epoch=completed_epochs,
        initial_best_val_loss=initial_best_val_loss,
    )
    checkpoint = EpochCheckpointCallback(
        workflow=workflow,
        label=f"hierarchical_{label}",
        checkpoint_path=checkpoint_path,
        state_path=state_path,
        history_path=history_path,
        state_factory=lambda epoch, best_val_loss=None: state_from_args(label, args, epoch, best_val_loss),
        start_epoch=completed_epochs + 1,
        best_checkpoint_path=best_checkpoint_path,
        best_state_path=best_state_path,
        initial_best_val_loss=initial_best_val_loss,
    )

    if args.online:
        history = workflow.fit_online(
            validation_data=val_data,
            epochs=remaining_epochs,
            num_batches_per_epoch=n_batches_per_epoch,
            batch_size=args.batch_size,
            keep_optimizer=args.keep_optimizer,
            verbose=0,
            callbacks=[progress, checkpoint],
        )
    else:
        history = workflow.fit_offline(
            train_data,
            validation_data=val_data,
            epochs=remaining_epochs,
            batch_size=args.batch_size,
            keep_optimizer=args.keep_optimizer,
            verbose=0,
            callbacks=[progress, checkpoint],
        )

    metrics = summarize_history(f"hierarchical_{label}", history)
    epochs_run = checkpoint.saved_epochs or len(next(iter(metrics.values()), []))
    completed_epochs += epochs_run
    if checkpoint.saved_epochs == 0:
        save_training_state(state_path, state_from_args(label, args, completed_epochs, checkpoint.best_val_loss))
        save_checkpoint(workflow, checkpoint_path)


def smoke_check():
    sample = simulator.sample(2, n_subjects=2, min_trials=60, max_trials=64)
    print("Backend:", keras.backend.backend())
    print("sim_data shape:", np.asarray(sample["sim_data"]).shape)
    print("valid_mask shape:", np.asarray(sample["valid_mask"]).shape)
    print("n_trials shape:", np.asarray(sample["n_trials"]).shape)
    print("global workflow:", type(global_workflow).__name__)
    print("local workflow:", type(local_workflow).__name__)


def main():
    parser = argparse.ArgumentParser(description="Train the variable-trial hierarchical DDM.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("smoke", help="Run a short simulator/workflow smoke check.")

    train_parser = subparsers.add_parser("train-hierarchical", help="Train the global or local workflow.")
    train_parser.add_argument("--which", choices=["global", "local"], default="global")
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument("--data-cache", default=None)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument("--keep-optimizer", dest="keep_optimizer", action="store_true")
    train_parser.add_argument("--reset-optimizer", dest="keep_optimizer", action="store_false")
    train_parser.add_argument("--minutes", type=float, default=5.0)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--n-train", type=int, default=None)
    train_parser.add_argument("--n-val", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--n-subjects", type=int, default=DEFAULT_N_SUBJECTS)
    train_parser.add_argument("--min-trials", type=int, default=DEFAULT_MIN_TRIALS)
    train_parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    train_parser.add_argument("--dt", type=float, default=1e-3)
    train_parser.add_argument("--max-time", type=float, default=10.0)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--serial-simulation", action="store_true")
    train_parser.add_argument("--online", dest="online", action="store_true")
    train_parser.add_argument("--offline", dest="online", action="store_false")
    train_parser.add_argument("--report-every-seconds", type=float, default=30.0)
    train_parser.add_argument("--report-every-epochs", type=int, default=1)
    train_parser.set_defaults(keep_optimizer=True, online=True)

    args = parser.parse_args()
    if args.command is None or args.command == "smoke":
        smoke_check()
    elif args.command == "train-hierarchical":
        train_hierarchical(args)


if __name__ == "__main__":
    main()
