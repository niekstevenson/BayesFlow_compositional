"""Train the hierarchical conflict DDM on groups of 50 subjects.

This is a separate 50-subject variant of ``compositional_diffusion_conflict``.
The priors, simulator, 120 trials per subject, diffusion inference network,
online training schedule, and local subject model are unchanged. The global
summary network is hierarchical: it first summarizes each subject's 120
trials, then summarizes the set of 50 subject embeddings. This avoids applying
quadratic self-attention directly to 6,000 observations and preserves subject
boundaries.

The default full run trains the global model, reuses the existing compatible
local model, and evaluates direct recovery at 50 subjects and compositional
recovery at the planned 5,000-subject scale.

Usage:
    conda run --no-capture-output -n tf-gpu python -u \
        compositional_diffusion_conflict_50subjects.py

    conda run --no-capture-output -n tf-gpu python -u \
        compositional_diffusion_conflict_50subjects.py --smoke-test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import compositional_diffusion_conflict as conflict

import tensorflow as tf

# Configure the allocator before importing Keras or BayesFlow, both of which
# may initialize TensorFlow's GPU context during import.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

import bayesflow as bf
import keras
import numpy as np

conflict.bf = bf
conflict.keras = keras


N_LOCAL_SUBJECTS = 50
N_SUBJECTS = 5_000
GLOBAL_BATCH_SIZE = 8
LOCAL_BATCH_SIZE = 64
COMPOSITIONAL_SUMMARY_BATCH_SIZE = 8
SAVE_DIR = Path("results/compositional_diffusion_conflict_120trials_50subjects_online_nested")
SOURCE_LOCAL_MODEL = conflict.SAVE_DIR / "local_approximator.keras"

TRIAL_SUMMARY_NETWORK_KWARGS = {
    "summary_dim": 64,
    "embed_dims": (96, 96, 96),
    "num_heads": (4, 4, 4),
    "num_seeds": 2,
    "dropout": 0.05,
}
SUBJECT_SUMMARY_NETWORK_KWARGS = {
    "summary_dim": 64,
    "embed_dims": (96, 96, 96),
    "num_heads": (4, 4, 4),
    "num_seeds": 4,
    "dropout": 0.05,
}

_original_fit_or_load_local = conflict.fit_or_load_local
_original_save_json = conflict.save_json
_runtime_reported = False


def require_training_gpu() -> None:
    """Fail before a long run if TensorFlow cannot access a GPU."""
    global _runtime_reported
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise SystemExit(
            "TensorFlow cannot see a GPU. Run this in the tf-gpu conda environment."
        )
    if not _runtime_reported:
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"Keras backend: {keras.backend.backend()}", flush=True)
        print(f"TensorFlow version: {tf.__version__}", flush=True)
        print(f"TensorFlow GPU devices: {logical_gpus or physical_gpus}", flush=True)
        _runtime_reported = True


@keras.saving.register_keras_serializable(package="BayesFlowComp")
class HierarchicalSetTransformer(bf.networks.SummaryNetwork):
    """Summarize trials within subjects, then subjects within a group."""

    def __init__(
        self,
        n_subjects=N_LOCAL_SUBJECTS,
        n_trials=conflict.N_TRIALS,
        trial_network_kwargs=None,
        subject_network_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_subjects = int(n_subjects)
        self.n_trials = int(n_trials)
        self.trial_network_kwargs = dict(
            trial_network_kwargs or TRIAL_SUMMARY_NETWORK_KWARGS
        )
        self.subject_network_kwargs = dict(
            subject_network_kwargs or SUBJECT_SUMMARY_NETWORK_KWARGS
        )
        self.trial_summary_dim = int(self.trial_network_kwargs["summary_dim"])
        self.trial_network = bf.networks.SetTransformer(**self.trial_network_kwargs)
        self.subject_network = bf.networks.SetTransformer(**self.subject_network_kwargs)

    def call(self, x, training=False, mask=None):
        if len(x.shape) != 3:
            raise ValueError(
                "HierarchicalSetTransformer expects (batch, subjects * trials, features); "
                f"received shape {x.shape}."
            )

        expected_observations = self.n_subjects * self.n_trials
        observed = x.shape[-2]
        if observed is not None and observed != expected_observations:
            raise ValueError(
                f"Expected {expected_observations} observations "
                f"({self.n_subjects} subjects x {self.n_trials} trials), got {observed}."
            )

        feature_dim = keras.ops.shape(x)[-1]
        subject_trials = keras.ops.reshape(x, (-1, self.n_trials, feature_dim))
        subject_summaries = self.trial_network(subject_trials, training=training)
        subject_summaries = keras.ops.reshape(
            subject_summaries,
            (-1, self.n_subjects, self.trial_summary_dim),
        )
        return self.subject_network(subject_summaries, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(self.subject_network_kwargs["summary_dim"])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_subjects": self.n_subjects,
                "n_trials": self.n_trials,
                "trial_network_kwargs": self.trial_network_kwargs,
                "subject_network_kwargs": self.subject_network_kwargs,
            }
        )
        return config


def make_workflows(require_gpu=True):
    """Build the original workflows with the nested global summary network."""
    if require_gpu:
        require_training_gpu()

    adapter_global = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(conflict.PARAM_NAMES_GLOBAL, into="inference_variables")
        .rename("sim_data", "summary_variables")
    )
    workflow_global = bf.CompositionalWorkflow(
        adapter=adapter_global,
        summary_network=HierarchicalSetTransformer(),
        inference_network=bf.networks.DiffusionModel(
            **conflict.GLOBAL_INFERENCE_NETWORK_KWARGS
        ),
    )

    adapter_local = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(conflict.PARAM_NAMES_LOCAL, into="inference_variables")
        .concatenate(conflict.PARAM_NAMES_GLOBAL, into="inference_conditions")
        .rename("sim_data", "summary_variables")
    )
    workflow_local = bf.BasicWorkflow(
        adapter=adapter_local,
        summary_network=bf.networks.SetTransformer(summary_dim=16),
        inference_network=bf.networks.StableConsistencyModel(),
    )
    return workflow_global, workflow_local


def fit_or_reuse_local(workflow_local, save_dir: Path):
    """Reuse the unchanged local model, or train it with its original batch size."""
    target_path = save_dir / "local_approximator.keras"
    if conflict.REUSE_SAVED_MODELS and target_path.exists():
        print(f"[local] loading saved approximator from {target_path}", flush=True)
        workflow_local.approximator = keras.saving.load_model(target_path)
        return

    if SOURCE_LOCAL_MODEL.exists():
        print(f"[local] reusing compatible approximator from {SOURCE_LOCAL_MODEL}", flush=True)
        workflow_local.approximator = keras.saving.load_model(SOURCE_LOCAL_MODEL)
        keras.saving.save_model(workflow_local.approximator, target_path)
        print(f"saved {target_path}", flush=True)
        return

    previous_batch_size = conflict.BATCH_SIZE
    conflict.BATCH_SIZE = LOCAL_BATCH_SIZE
    try:
        _original_fit_or_load_local(workflow_local, save_dir)
    finally:
        conflict.BATCH_SIZE = previous_batch_size


def save_json_with_architecture(path: Path, payload) -> None:
    if path.name == "run_config.json":
        payload = dict(payload)
        payload.update(
            {
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "local_batch_size": LOCAL_BATCH_SIZE,
                "global_summary_architecture": "hierarchical_set_transformer",
                "trial_summary_network_kwargs": TRIAL_SUMMARY_NETWORK_KWARGS,
                "subject_summary_network_kwargs": SUBJECT_SUMMARY_NETWORK_KWARGS,
                "source_local_model": str(SOURCE_LOCAL_MODEL),
            }
        )
    _original_save_json(path, payload)


def configure_base_module() -> None:
    """Apply this entry point's isolated configuration to the shared implementation."""
    conflict.N_LOCAL_SUBJECTS = N_LOCAL_SUBJECTS
    conflict.N_SUBJECTS = N_SUBJECTS
    conflict.BATCH_SIZE = GLOBAL_BATCH_SIZE
    conflict.COMPOSITIONAL_SUMMARY_BATCH_SIZE = COMPOSITIONAL_SUMMARY_BATCH_SIZE
    conflict.SAVE_DIR = SAVE_DIR
    conflict.GLOBAL_SUMMARY_NETWORK_KWARGS = {
        "architecture": "hierarchical_set_transformer",
        "trial": TRIAL_SUMMARY_NETWORK_KWARGS,
        "subject": SUBJECT_SUMMARY_NETWORK_KWARGS,
    }
    conflict.make_workflows = make_workflows
    conflict.fit_or_load_local = fit_or_reuse_local
    conflict.save_json = save_json_with_architecture


def print_plan() -> None:
    samples_per_epoch = conflict.ONLINE_BATCHES_PER_EPOCH * GLOBAL_BATCH_SIZE
    observations_per_dataset = N_LOCAL_SUBJECTS * conflict.N_TRIALS
    print("50-subject hierarchical training plan")
    print(f"  global training subjects per dataset: {N_LOCAL_SUBJECTS}")
    print(f"  trials per subject: {conflict.N_TRIALS}")
    print(f"  flattened adapter input: ({observations_per_dataset}, 3)")
    print("  summary path: trials -> subject embedding -> group embedding")
    print(f"  global batch size: {GLOBAL_BATCH_SIZE}")
    print(f"  online batches per epoch: {conflict.ONLINE_BATCHES_PER_EPOCH}")
    print(f"  global datasets per epoch: {samples_per_epoch}")
    print(f"  global epochs: {conflict.GLOBAL_EPOCHS}")
    print(f"  compositional evaluation subjects: {N_SUBJECTS}")
    print(f"  compositional groups: {N_SUBJECTS // N_LOCAL_SUBJECTS}")
    print(f"  output: {SAVE_DIR.resolve()}")


def smoke_test() -> None:
    """Verify simulation, nested summaries, and Keras serialization."""
    workflow_global, _ = make_workflows(require_gpu=False)
    data = conflict.sample_serial(
        2,
        n_subjects=N_LOCAL_SUBJECTS,
        n_trials=conflict.N_TRIALS,
        seed=20260713,
    )
    data["sim_data"] = data["sim_data"].reshape(
        2,
        N_LOCAL_SUBJECTS * conflict.N_TRIALS,
        3,
    )
    workflow_global.approximator.adapter(data, strict=True)
    summaries = workflow_global.approximator.summarize(
        {"sim_data": data["sim_data"]},
        batch_size=1,
    )
    if tuple(summaries.shape) != (2, SUBJECT_SUMMARY_NETWORK_KWARGS["summary_dim"]):
        raise AssertionError(f"Unexpected summary shape: {summaries.shape}")

    history = workflow_global.fit_offline(
        data,
        epochs=1,
        batch_size=1,
        verbose=0,
    )
    loss = float(history.history["loss"][-1])
    if not np.isfinite(loss):
        raise AssertionError(f"Non-finite smoke-training loss: {loss}")

    before = workflow_global.approximator.summarize(
        {"sim_data": data["sim_data"]},
        batch_size=1,
    )
    smoke_path = Path("/tmp/compositional_diffusion_conflict_50subjects_smoke.keras")
    keras.saving.save_model(workflow_global.approximator, smoke_path, overwrite=True)
    restored = keras.saving.load_model(smoke_path)
    after = restored.summarize({"sim_data": data["sim_data"]}, batch_size=1)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
    print(
        f"smoke test passed: data={data['sim_data'].shape}, summaries={summaries.shape}, "
        f"loss={loss:.5g}, serialized_output={after.shape}",
        flush=True,
    )


def train_only() -> None:
    """Train or load only the new global approximator."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_json_with_architecture(
        SAVE_DIR / "run_config.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "tensorflow",
            "model": "hierarchical_conflict_ddm",
            "n_local_subjects": N_LOCAL_SUBJECTS,
            "n_trials": conflict.N_TRIALS,
            "global_epochs": conflict.GLOBAL_EPOCHS,
            "online_training": conflict.ONLINE_TRAINING,
            "online_batches_per_epoch": conflict.ONLINE_BATCHES_PER_EPOCH,
        },
    )
    workflow_global, _ = make_workflows()
    conflict.fit_or_load_global(workflow_global, SAVE_DIR)
    print(f"global training complete. Model saved in {SAVE_DIR}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument(
        "--train-only",
        action="store_true",
        help="Train the global model without running recovery evaluations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_base_module()
    print_plan()
    if args.plan_only:
        return
    if args.smoke_test:
        smoke_test()
        return
    require_training_gpu()
    if args.train_only:
        train_only()
        return
    conflict.run_tutorial()


if __name__ == "__main__":
    main()
