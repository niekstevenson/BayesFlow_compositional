"""Serializable task-marked hierarchical summary network."""

from __future__ import annotations

import bayesflow as bf
import keras


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


@keras.saving.register_keras_serializable(package="BayesFlowComp")
class TaskMarkedHierarchicalSetTransformer(bf.networks.SummaryNetwork):
    """Summarize trials, attach task identity, then summarize participants."""

    def __init__(
        self,
        n_subjects: int = 50,
        n_trials: int = 120,
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
        self.subject_summary_dim = int(self.subject_network_kwargs["summary_dim"])
        self.trial_network = bf.networks.SetTransformer(**self.trial_network_kwargs)
        self.subject_network = bf.networks.SetTransformer(**self.subject_network_kwargs)

    def call(self, x, training=False, mask=None):
        if len(x.shape) != 3:
            raise ValueError(
                "TaskMarkedHierarchicalSetTransformer expects "
                "(batch, subjects * trials, 6); "
                f"received shape {x.shape}."
            )
        expected_observations = self.n_subjects * self.n_trials
        observed = x.shape[-2]
        if observed is not None and observed != expected_observations:
            raise ValueError(
                f"Expected {expected_observations} observations "
                f"({self.n_subjects} subjects x {self.n_trials} trials), got {observed}."
            )
        feature_dim = x.shape[-1]
        if feature_dim is not None and feature_dim != 6:
            raise ValueError(
                "Expected (choice, rt, condition, task_flanker, task_simon, "
                f"task_stroop), got {feature_dim} features."
            )

        subject_trials = keras.ops.reshape(
            x,
            (-1, self.n_subjects, self.n_trials, 6),
        )
        task_one_hot = subject_trials[:, :, 0, 3:6]
        trial_inputs = keras.ops.reshape(
            subject_trials[..., :3],
            (-1, self.n_trials, 3),
        )
        subject_summaries = self.trial_network(trial_inputs, training=training)
        subject_summaries = keras.ops.reshape(
            subject_summaries,
            (-1, self.n_subjects, self.trial_summary_dim),
        )
        marked_subjects = keras.ops.concatenate(
            (subject_summaries, task_one_hot),
            axis=-1,
        )
        group_summary = self.subject_network(marked_subjects, training=training)

        # Explicit proportions prevent the outer attention pool from having to
        # reconstruct the task counts from softmax-normalized attention alone.
        task_proportions = keras.ops.mean(task_one_hot, axis=1)
        return keras.ops.concatenate((group_summary, task_proportions), axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.subject_summary_dim + 3

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
