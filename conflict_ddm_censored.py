"""Conflict-DDM simulation with a 100-second response deadline.

As requested for this benchmark, a timeout is represented as ``choice=0`` and
``rt=100``. Genuine lower-boundary responses also remain encoded as choice 0.
"""

from __future__ import annotations

import numpy as np
from numba import njit

import compositional_diffusion_conflict as conflict


TIMEOUT_CHOICE = 0.0
DEFAULT_MAX_TIME = 100.0


@njit
def simulate_ddm_trial(
    drift,
    boundary,
    non_decision_time,
    start_fraction=0.5,
    dt=1e-3,
    max_time=DEFAULT_MAX_TIME,
):
    """Simulate one trial and encode a deadline timeout as ``(choice=0, rt=100)``."""
    evidence = start_fraction * boundary
    rt = non_decision_time

    while boundary > evidence > 0.0 and rt < max_time:
        step = min(dt, max_time - rt)
        evidence += drift * step + np.sqrt(step) * np.random.randn()
        rt += step

    if evidence >= boundary:
        return 1.0, rt
    if evidence <= 0.0:
        return 0.0, rt
    return TIMEOUT_CHOICE, max_time


def count_timeouts(sim_data: np.ndarray) -> int:
    """Count deadline responses by RT, since choice 0 is not unique to timeouts."""
    data = np.asarray(sim_data)
    return int(
        np.count_nonzero(
            (data[..., 0] == TIMEOUT_CHOICE) & (data[..., 1] == DEFAULT_MAX_TIME)
        )
    )


@njit
def simulate_subjects(
    v, v_cond, log_a, log_t0, log_sv, n_trials, sim_data, dt, max_time
):
    for i_subject in range(v.shape[0]):
        half = n_trials // 2
        condition = np.empty(n_trials)
        condition[:half] = 0.5
        condition[half:] = -0.5
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
                dt,
                max_time,
            )
            sim_data[i_subject, i_trial, 0] = choice
            sim_data[i_subject, i_trial, 1] = rt
            sim_data[i_subject, i_trial, 2] = condition[i_trial]


def simulate_one_dataset(n_subjects=1, n_trials=conflict.N_TRIALS):
    params = conflict.sample_hierarchical_priors(n_subjects=n_subjects)
    sim_data = np.zeros((n_subjects, n_trials, 3), dtype=np.float64)
    simulate_subjects(
        params["v"],
        params["v_cond"],
        params["log_a"],
        params["log_t0"],
        params["log_sv"],
        n_trials,
        sim_data,
        1e-3,
        DEFAULT_MAX_TIME,
    )
    return params | {"sim_data": sim_data}


def sample_serial(
    n_samples: int,
    n_subjects=1,
    n_trials=conflict.N_TRIALS,
    seed: int | None = None,
):
    if seed is not None:
        np.random.seed(seed)
        conflict.seed_numba(seed + 1)
    return conflict.stack_simulation_results(
        [
            simulate_one_dataset(n_subjects=n_subjects, n_trials=n_trials)
            for _ in range(n_samples)
        ]
    )


def sample_parallel(
    n_samples: int,
    n_subjects=1,
    n_trials=conflict.N_TRIALS,
    workers=conflict.PARALLEL_WORKERS,
    seed: int | None = None,
):
    from joblib import Parallel, delayed

    # Keep the seeded simulation stream invariant to the number of workers.
    counts = conflict.split_counts(n_samples, 32)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = [
        int(child.generate_state(1)[0]) for child in seed_sequence.spawn(len(counts))
    ]
    batches = Parallel(n_jobs=workers)(
        delayed(sample_serial)(
            count,
            n_subjects=n_subjects,
            n_trials=n_trials,
            seed=child_seed,
        )
        for count, child_seed in zip(counts, child_seeds)
        if count > 0
    )
    return conflict.merge_batches(batches)


def sample_data(
    n_samples: int,
    n_subjects=1,
    n_trials=conflict.N_TRIALS,
    workers=conflict.PARALLEL_WORKERS,
    seed: int | None = None,
):
    return sample_parallel(
        n_samples,
        n_subjects=n_subjects,
        n_trials=n_trials,
        workers=workers,
        seed=seed,
    )


def make_global_training_data(
    n_samples: int,
    n_subjects: int,
    n_trials: int,
    workers: int,
):
    data = sample_data(
        n_samples,
        n_subjects=n_subjects,
        n_trials=n_trials,
        workers=workers,
    )
    data["sim_data"] = data["sim_data"].reshape(
        n_samples,
        n_subjects * n_trials,
        3,
    )
    return data
