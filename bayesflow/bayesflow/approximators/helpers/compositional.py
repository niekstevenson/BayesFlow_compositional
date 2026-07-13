from collections.abc import Callable
import inspect

import keras

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor


def build_prior_score_fn(
    compute_prior_score: Callable,
    adapter: Adapter,
    standardizer,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a prior score step function with all fixed state captured once.

    Inspects ``compute_prior_score`` signature, pre-computes the standardization std
    correction, and returns a ``(samples, time) -> Tensor`` closure.

    The callback must evaluate the score of the *diffused* prior at the supplied
    state. A callback without a ``time`` argument is interpreted as a
    time-independent noisy-prior score (for example, a standardized Gaussian
    prior under a variance-preserving diffusion). BayesFlow does not attempt to
    turn a generic clean-prior score into a noisy-prior score.

    Parameters
    ----------
    compute_prior_score : Callable
        Function that computes noisy-prior scores. May or may not accept a
        ``time`` keyword argument if the score is time-independent.
    adapter : Adapter
        Adapter used to perform inverse transformation of samples.
        Must have zero log_det_jac for all keys.
    standardizer : object
        Fitted standardizer with ``standardize`` dict and ``standardize_layers``.

    Returns
    -------
    Callable[[Tensor, Tensor], Tensor]
        Step function with signature ``(samples, time) -> Tensor``.

    Raises
    ------
    NotImplementedError
        If the adapter has non-zero log_det_jac for any key.
    """

    # Capture fixed states
    prior_has_time = "time" in inspect.signature(compute_prior_score).parameters

    if "inference_variables" in standardizer.standardize:
        standardize_layer = standardizer.standardize_layers["inference_variables"]
        std_components = [standardize_layer.moving_std(idx) for idx in range(len(standardize_layer.moving_mean))]
        std = std_components[0] if len(std_components) == 1 else keras.ops.concatenate(std_components, axis=-1)
        std_expanded = keras.ops.expand_dims(std, 0)
    else:
        std_expanded = None

    def _step(samples: Tensor, time: Tensor) -> Tensor:
        samples = keras.tree.map_structure(
            lambda s: standardizer.maybe_standardize(s, key="inference_variables", stage="inference", forward=False),
            samples,
        )

        if keras.backend.backend() != "torch":  # samples cannot be converted to numpy, otherwise it breaks
            adapted_samples, log_det_jac = keras.tree.map_structure(
                lambda s: adapter({"inference_variables": s}, inverse=True, strict=False, log_det_jac=True),
                samples,
            )
        else:  # samples need to be converted to numpy, adapter cannot use torch tensors
            adapted_samples, log_det_jac = keras.tree.map_structure(
                lambda s: adapter(
                    {"inference_variables": keras.ops.convert_to_numpy(s)},
                    inverse=True,
                    strict=False,
                    log_det_jac=True,
                ),
                samples,
            )

        if len(log_det_jac) > 0:
            problematic_keys = [key for key in log_det_jac if log_det_jac[key] != 0.0]
            raise NotImplementedError(
                f"Cannot use compositional sampling with adapters "
                f"that have non-zero log_det_jac. Problematic keys: {problematic_keys}"
            )

        if prior_has_time:
            callback_time = keras.ops.convert_to_numpy(time) if keras.backend.backend() == "torch" else time
            prior_score = compute_prior_score(adapted_samples, time=callback_time)
        else:
            prior_score = compute_prior_score(adapted_samples)

        for key in adapted_samples:
            prior_score[key] = keras.ops.cast(prior_score[key], "float32")
        prior_score = keras.tree.map_structure(keras.ops.convert_to_tensor, prior_score)
        out = keras.ops.concatenate([prior_score[key] for key in adapted_samples], axis=-1)

        # Apply Jacobian correction from standardization:
        # For T^{-1}(z) = z * std + mean the score transforms as score_z = score_x * std
        if std_expanded is not None:
            out = out * std_expanded

        return out

    return _step
