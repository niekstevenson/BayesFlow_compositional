"""Task-aware hierarchical conflict DDM used by the K=50 benchmark.

Each participant performs exactly one of flanker, Simon, or Stroop. Task is an
observed participant-level covariate. Proportion-centered treatment contrasts
make ``mu_v_cond`` and ``mu_log_a`` population-weighted means while the Simon
and Stroop coefficients remain differences from Flanker.
"""

from __future__ import annotations

import numpy as np

import compositional_diffusion_conflict as conflict
import conflict_ddm_censored as censored


TASK_NAMES = ("flanker", "simon", "stroop")
TASK_PROPORTIONS = np.asarray((0.3438247, 0.2318725, 0.4243028), dtype=np.float64)
if np.any(TASK_PROPORTIONS <= 0.0) or not np.isclose(TASK_PROPORTIONS.sum(), 1.0):
    raise ValueError(f"Invalid task probabilities: {TASK_PROPORTIONS}")

# The contrast scales equal the median participant heterogeneity in the
# corresponding original hierarchy: exp(E[log sigma]).
V_COND_CONTRAST_SD = float(np.exp(conflict.GLOBAL_PRIOR["log_sigma_v_cond"][0]))
LOG_A_CONTRAST_SD = float(np.exp(conflict.GLOBAL_PRIOR["log_sigma_log_a"][0]))

GLOBAL_PRIOR = {
    "mu_v": conflict.GLOBAL_PRIOR["mu_v"],
    "mu_v_cond": conflict.GLOBAL_PRIOR["mu_v_cond"],
    "beta_v_cond_simon": (0.0, V_COND_CONTRAST_SD),
    "beta_v_cond_stroop": (0.0, V_COND_CONTRAST_SD),
    "mu_log_a": conflict.GLOBAL_PRIOR["mu_log_a"],
    "beta_log_a_simon": (0.0, LOG_A_CONTRAST_SD),
    "beta_log_a_stroop": (0.0, LOG_A_CONTRAST_SD),
    "mu_log_t0": conflict.GLOBAL_PRIOR["mu_log_t0"],
    "mu_log_sv": conflict.GLOBAL_PRIOR["mu_log_sv"],
    "log_sigma_v": conflict.GLOBAL_PRIOR["log_sigma_v"],
    "log_sigma_v_cond": conflict.GLOBAL_PRIOR["log_sigma_v_cond"],
    "log_sigma_log_a": conflict.GLOBAL_PRIOR["log_sigma_log_a"],
    "log_sigma_log_t0": conflict.GLOBAL_PRIOR["log_sigma_log_t0"],
    "log_sigma_log_sv": conflict.GLOBAL_PRIOR["log_sigma_log_sv"],
}
PARAM_NAMES_GLOBAL = list(GLOBAL_PRIOR)


def sample_global_parameters() -> dict[str, float]:
    return {
        name: float(np.random.normal(location, scale))
        for name, (location, scale) in GLOBAL_PRIOR.items()
    }


def centered_task_contrasts(task_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Simon and Stroop treatment indicators centered by population share."""
    task_index = np.asarray(task_index)
    x_simon = (task_index == 1).astype(np.float64) - TASK_PROPORTIONS[1]
    x_stroop = (task_index == 2).astype(np.float64) - TASK_PROPORTIONS[2]
    return x_simon, x_stroop


def task_level_means(parameters: dict[str, float]) -> dict[str, np.ndarray]:
    """Return implied means ordered as flanker, Simon, Stroop."""
    task_index = np.arange(3)
    x_simon, x_stroop = centered_task_contrasts(task_index)
    return {
        "v_cond": parameters["mu_v_cond"]
        + parameters["beta_v_cond_simon"] * x_simon
        + parameters["beta_v_cond_stroop"] * x_stroop,
        "log_a": parameters["mu_log_a"]
        + parameters["beta_log_a_simon"] * x_simon
        + parameters["beta_log_a_stroop"] * x_stroop,
    }


def simulate_one_dataset(
    n_subjects=1, n_trials=conflict.N_TRIALS
) -> dict[str, np.ndarray]:
    if n_trials % 2:
        raise ValueError("n_trials must be even to balance the two trial conditions.")

    global_parameters = sample_global_parameters()
    task_index = np.random.choice(3, size=n_subjects, p=TASK_PROPORTIONS)
    task_one_hot = np.eye(3, dtype=np.float64)[task_index]
    x_simon, x_stroop = centered_task_contrasts(task_index)

    v = np.random.normal(
        global_parameters["mu_v"],
        np.exp(global_parameters["log_sigma_v"]),
        size=n_subjects,
    )
    v_cond_mean = (
        global_parameters["mu_v_cond"]
        + global_parameters["beta_v_cond_simon"] * x_simon
        + global_parameters["beta_v_cond_stroop"] * x_stroop
    )
    v_cond = np.random.normal(
        v_cond_mean,
        np.exp(global_parameters["log_sigma_v_cond"]),
    )
    log_a_mean = (
        global_parameters["mu_log_a"]
        + global_parameters["beta_log_a_simon"] * x_simon
        + global_parameters["beta_log_a_stroop"] * x_stroop
    )
    log_a = np.random.normal(
        log_a_mean,
        np.exp(global_parameters["log_sigma_log_a"]),
    )
    log_t0 = np.random.normal(
        global_parameters["mu_log_t0"],
        np.exp(global_parameters["log_sigma_log_t0"]),
        size=n_subjects,
    )
    log_sv = np.random.normal(
        global_parameters["mu_log_sv"],
        np.exp(global_parameters["log_sigma_log_sv"]),
        size=n_subjects,
    )

    trial_data = np.zeros((n_subjects, n_trials, 3), dtype=np.float64)
    censored.simulate_subjects(
        v,
        v_cond,
        log_a,
        log_t0,
        log_sv,
        n_trials,
        trial_data,
        1e-3,
        censored.DEFAULT_MAX_TIME,
    )
    task_features = np.broadcast_to(
        task_one_hot[:, None, :],
        (n_subjects, n_trials, 3),
    )
    sim_data = np.concatenate((trial_data, task_features), axis=-1)
    return global_parameters | {"sim_data": sim_data}


def sample_serial(
    n_samples: int,
    n_subjects=1,
    n_trials=conflict.N_TRIALS,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
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
) -> dict[str, np.ndarray]:
    from joblib import Parallel, delayed

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
) -> dict[str, np.ndarray]:
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
) -> dict[str, np.ndarray]:
    data = sample_data(
        n_samples,
        n_subjects=n_subjects,
        n_trials=n_trials,
        workers=workers,
    )
    data["sim_data"] = data["sim_data"].reshape(
        n_samples,
        n_subjects * n_trials,
        6,
    )
    return data


def validate_sim_data(sim_data: np.ndarray) -> None:
    data = np.asarray(sim_data)
    if data.shape[-1] != 6:
        raise ValueError(f"Expected six simulation features, got {data.shape}.")
    task = data[..., 3:]
    if not np.all((task == 0.0) | (task == 1.0)):
        raise ValueError("Task indicators must be binary one-hot values.")
    if not np.all(task.sum(axis=-1) == 1.0):
        raise ValueError("Every trial must contain exactly one task indicator.")
    if data.ndim >= 4 and not np.all(task == task[..., :1, :]):
        raise ValueError(
            "Task identity must be constant across a participant's trials."
        )
