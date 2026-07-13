from collections.abc import Sequence, Mapping
from typing import Any, Literal, Callable
import math

import keras
from keras import ops

from bayesflow.types import Tensor, Shape
from bayesflow.utils import (
    expand_right_as,
    find_network,
    integrate,
    integrate_stochastic,
    jacobian_trace,
    layer_kwargs,
    logging,
    linsolve_batched,
    maybe_mask_tensor,
    random_mask,
    randomly_mask_along_axis,
    resolve_seed,
    weighted_mean,
    DETERMINISTIC_METHODS,
    STOCHASTIC_METHODS,
)
from bayesflow.utils.serialization import serialize, serializable

from .schedules.noise_schedule import NoiseSchedule
from .dispatch import find_noise_schedule

from ...inference import InferenceNetwork
from ...defaults import TIME_MLP_DEFAULTS, DIFFUSION_INTEGRATE_DEFAULTS


@serializable("bayesflow.networks")
class DiffusionModel(InferenceNetwork):
    """Score-based diffusion model for simulation-based inference (SBI).

    Implements a score-based diffusion model with configurable subnet architecture,
    noise schedule, and prediction/loss types for amortized SBI as described in [1].
    The diffusion model allows for post-hoc guidance (see [1]) and composition [2],
    implementing, stabilizing, and scaling initial ideas from [3] and [4].

    Note that score-based diffusion is the most sluggish of all available samplers,
    so expect slower inference times than flow matching and much slower than
    normalizing flows.

    Parameters
    ----------
    subnet : str, type, or keras.Layer, optional
        A neural network type for the diffusion model, will be instantiated using
        *subnet_kwargs*.  If a string is provided, it should be a registered name
        (e.g., ``"time_mlp"``).  If a type or ``keras.Layer`` is provided, it will
        be directly instantiated with the given *subnet_kwargs*.  Any subnet must
        accept a tuple of tensors ``(target, time, conditions)``.
    noise_schedule : {'edm', 'cosine'} or NoiseSchedule or type, optional
        Noise schedule controlling the diffusion dynamics.  Can be a string
        identifier, a schedule class, or a pre-initialised schedule instance.
        Default is ``"edm"``.
    prediction_type : {'velocity', 'noise', 'F', 'x', 'score', 'potential'}, optional
        Output format of the model's prediction.  Default is ``"F"``.
    loss_type : {'velocity', 'noise', 'F'}, optional
        Loss function used to train the model.  Default is ``"noise"``.
    subnet_kwargs : dict[str, Any], optional
        Additional keyword arguments passed to the subnet constructor.
    schedule_kwargs : dict[str, Any], optional
        Additional keyword arguments passed to the noise schedule constructor.
    integrate_kwargs : dict[str, Any], optional
        Configuration dictionary for the ODE/SDE integrator used at inference time.
    drop_cond_prob : float, optional
        Probability of dropping conditions during training (i.e., classifier-free guidance).
        Default is 0.0.
    drop_target_prob : float, optional
        Probability of dropping target values during training (i.e., learning arbitrary
        distributions). Default is 0.0.
    **kwargs
        Additional keyword arguments passed to the base ``InferenceNetwork``.

    References
    ----------
    [1] Arruda, J., Bracher, N., Köthe, U., Hasenauer, J., & Radev, S. T. (2025).
        Diffusion Models in Simulation-Based Inference: A Tutorial Review.
        arXiv preprint arXiv:2512.20685.
    [2] Arruda et al., (2025). Compositional amortized inference for large-scale hierarchical Bayesian models.
        https://arxiv.org/abs/2505.14429
    [3] Geffner et al., (2023). Compositional score modeling for simulation-based inference.
        https://arxiv.org/abs/2209.14249
    [4] Linhart et al., (2024). Diffusion posterior sampling for simulation-based inference in tall data settings.
        https://arxiv.org/abs/2404.07593
    """

    def __init__(
        self,
        *,
        subnet: str | type | keras.Layer = "time_mlp",
        noise_schedule: Literal["edm", "cosine"] | NoiseSchedule | type = "edm",
        prediction_type: Literal["velocity", "noise", "F", "x", "score", "potential"] = "F",
        loss_type: Literal["velocity", "noise", "F"] = "noise",
        subnet_kwargs: dict[str, Any] = None,
        schedule_kwargs: dict[str, Any] = None,
        integrate_kwargs: dict[str, Any] = None,
        drop_cond_prob: float = 0.0,
        drop_target_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__(base_distribution="normal", **kwargs)

        if prediction_type not in ["noise", "velocity", "F", "x", "score", "potential"]:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        if loss_type not in ["noise", "velocity", "F"]:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if loss_type != "noise":
            logging.warning(
                "The standard schedules have weighting functions defined for the noise prediction loss. "
                "You might want to replace them if you are using a different loss function."
            )

        self._prediction_type = prediction_type
        self._loss_type = loss_type

        schedule_kwargs = schedule_kwargs or {}
        self.noise_schedule = find_noise_schedule(noise_schedule, **schedule_kwargs)
        self.noise_schedule.validate()

        self.integrate_kwargs = DIFFUSION_INTEGRATE_DEFAULTS | (integrate_kwargs or {})
        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "time_mlp":
            subnet_kwargs = TIME_MLP_DEFAULTS | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)

        self.output_projector = None
        self.drop_cond_prob = drop_cond_prob
        self.unconditional_mode = False
        self.drop_target_prob = drop_target_prob

        self.compositional_bridge_d0 = 1.0
        self.compositional_bridge_d1 = 1.0

    def compute_metrics(
        self,
        x: Tensor | Sequence[Tensor],
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        subnet_kwargs = self._collect_mask_kwargs(self._SUBNET_MASK_KEYS, kwargs)

        training = stage == "training"
        noise_schedule_training_stage = stage == "training" or stage == "validation"

        if conditions is not None:
            conditions = randomly_mask_along_axis(conditions, self.drop_cond_prob, seed_generator=self.seed_generator)

        # Sample training diffusion time as a low discrepancy sequence to decrease variance
        u0 = keras.random.uniform(shape=(1,), dtype=ops.dtype(x), seed=self.seed_generator)
        i = ops.arange(0, ops.shape(x)[0], dtype=ops.dtype(x))
        t = (u0 + i / ops.cast(ops.shape(x)[0], dtype=ops.dtype(x))) % 1

        # Calculate the noise level
        log_snr_t = self.noise_schedule.get_log_snr(t, training=noise_schedule_training_stage)
        log_snr_t = expand_right_as(log_snr_t, x)

        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)
        weights_for_snr = self.noise_schedule.get_weights_for_snr(log_snr_t=log_snr_t)

        # Generate noise vector
        eps_t = keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # Diffuse x to get noisy input to the network
        diffused_x = alpha_t * x + sigma_t * eps_t

        # Generate optional target dropout mask
        mask_x = random_mask(ops.shape(x), self.drop_target_prob, self.seed_generator)
        diffused_x = maybe_mask_tensor(diffused_x, mask=mask_x, replacement=x)

        # Obtain output of the network and transform to prediction of the clean signal x
        norm_log_snr_t = self._transform_log_snr(log_snr_t)
        if self._prediction_type == "potential":
            pred = self._compute_grad_of_potential(
                xz=diffused_x, norm_log_snr=norm_log_snr_t, conditions=conditions, training=training, **subnet_kwargs
            )
            x_pred = self.convert_prediction_to_x(
                pred=pred,
                z=diffused_x,
                alpha_t=alpha_t,
                sigma_t=sigma_t,
                log_snr_t=log_snr_t,
                prediction_type="velocity",
            )
        else:
            subnet_out = self.subnet((diffused_x, norm_log_snr_t, conditions), training=training, **subnet_kwargs)
            pred = self.output_projector(subnet_out)
            x_pred = self.convert_prediction_to_x(
                pred=pred,
                z=diffused_x,
                alpha_t=alpha_t,
                sigma_t=sigma_t,
                log_snr_t=log_snr_t,
                prediction_type=self._prediction_type,
            )

        # Finally, compute the loss according to the configured loss type.  Note that the standard weighting
        # functions are defined for the noise prediction loss, so if you use a different loss type, you might want
        # to adjust the weighting accordingly.
        match self._loss_type:
            case "noise":
                noise_pred = (diffused_x - alpha_t * x_pred) / sigma_t
                loss = weights_for_snr * ops.mean(mask_x * (noise_pred - eps_t) ** 2, axis=-1)

            case "velocity":
                velocity_pred = (alpha_t * diffused_x - x_pred) / sigma_t
                v_t = alpha_t * eps_t - sigma_t * x
                loss = weights_for_snr * ops.mean(mask_x * (velocity_pred - v_t) ** 2, axis=-1)

            case "F":
                sigma_data = self.noise_schedule.sigma_data if hasattr(self.noise_schedule, "sigma_data") else 1.0
                x1 = ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2) / (ops.exp(-log_snr_t / 2) * sigma_data)
                x2 = (sigma_data * alpha_t) / (ops.exp(-log_snr_t / 2) * ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2))
                f_pred = x1 * x_pred - x2 * diffused_x
                f_t = x1 * x - x2 * diffused_x
                loss = weights_for_snr * ops.mean(mask_x * (f_pred - f_t) ** 2, axis=-1)

            case _:
                raise ValueError(f"Unknown loss type: {self._loss_type}")

        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}

    def build(self, xz_shape: Shape, conditions_shape: Shape = None):
        if self.built:
            return

        self.base_distribution.build(xz_shape)

        units = 1 if self._prediction_type == "potential" else xz_shape[-1]
        self.output_projector = keras.layers.Dense(units=units, bias_initializer="zeros")

        # construct input shape for subnet and subnet projector
        time_shape = (xz_shape[0], 1)
        self.subnet.build((xz_shape, time_shape, conditions_shape))
        out_shape = self.subnet.compute_output_shape((xz_shape, time_shape, conditions_shape))

        self.output_projector.build(out_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "noise_schedule": self.noise_schedule,
            "prediction_type": self._prediction_type,
            "loss_type": self._loss_type,
            "integrate_kwargs": self.integrate_kwargs,
            "drop_cond_prob": self.drop_cond_prob,
            "drop_target_prob": self.drop_target_prob,
        }
        return base_config | serialize(config)

    def guidance_function(
        self,
        x_pred: Tensor,
        time: Tensor,
        score: Tensor,
        constraints: Callable | Sequence[Callable] = None,
        guidance_strength: float = 1.0,
        scaling_function: Callable = None,
        reduce: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        """
        Takes the current denoised estimate and the diffusion time. By default, it computes a guidance constraint term
         if guidance constraints are provided, and returns the score otherwise. Function can be overwritten by the
         user to implement custom guidance.

        Custom guidance:
        def guidance_function(x_pred, time, score):
            # do something here, needs to return the new score
            return score
        workflow.approximator.inference_network.guidance_function = guidance_function

        Parameters
        ----------
        x_pred : Tensor
            The denoised target at time t.
        time : Tensor
            The time corresponding to x.
        score: Tensor
            The score at the current time point, which should be guided.
        constraints : Callable or Sequence[Callable], optional
            A single constraint function or a list/tuple of constraint functions.
            Each function should take x as input and return a tensor of constraint values.
        guidance_strength : float
            A positive scaling factor for the guidance term. Default is 1.0.
        scaling_function : Callable, optional
            A function that takes time t as input and returns a scaling factor s(t).
            If None, a default scaling based on the noise schedule is used. Default is None.
        reduce : {'sum', 'mean'}
            Method to reduce the log-probabilities from multiple constraints. Default is 'sum'.

        Returns
        -------
        Tensor
            The computed guidance term of the same shape as zt.
        """
        if constraints is None:
            return score

        if not isinstance(constraints, Sequence):
            constraints = [constraints]

        if scaling_function is None:

            def scaling_function(t: Tensor) -> Tensor:
                log_snr = self.noise_schedule.get_log_snr(t, training=False)
                alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr)
                return ops.square(alpha_t) / ops.square(sigma_t)

        def objective_fn(x: Tensor) -> Tensor:
            st = scaling_function(time)
            logp = keras.ops.zeros((), dtype=x.dtype)
            for c in constraints:
                ck = c(x)
                logp = logp - keras.ops.softplus(st * ck)
            return keras.ops.sum(logp) if reduce == "sum" else keras.ops.mean(logp)

        backend = keras.backend.backend()

        match backend:
            case "jax":
                import jax

                grad = jax.grad(objective_fn)(x_pred)

            case "tensorflow":
                import tensorflow as tf

                with tf.GradientTape() as tape:
                    tape.watch(x_pred)
                    objective = objective_fn(x_pred)
                grad = tape.gradient(objective, x_pred)

            case "torch":
                import torch

                with torch.enable_grad():
                    x_grad = x_pred.clone().detach().requires_grad_(True)
                    objective = objective_fn(x_grad)
                    grad = torch.autograd.grad(
                        outputs=objective,
                        inputs=x_grad,
                    )[0]

            case _:
                raise NotImplementedError(f"Unsupported backend: {backend}")

        return score + guidance_strength * grad

    def convert_prediction_to_x(
        self, pred: Tensor, z: Tensor, alpha_t: Tensor, sigma_t: Tensor, log_snr_t: Tensor, prediction_type: str
    ) -> Tensor:
        """
        Converts the neural network prediction into the denoised data `x`, depending on
        the prediction type configured for the model.

        Parameters
        ----------
        pred : Tensor
            The output prediction from the neural network, typically representing noise,
            velocity, or a transformation of the clean signal.
        z : Tensor
            The noisy latent variable `z` to be denoised.
        alpha_t : Tensor
            The noise schedule's scaling factor for the clean signal at time `t`.
        sigma_t : Tensor
            The standard deviation of the noise at time `t`.
        log_snr_t : Tensor
            The log signal-to-noise ratio at time `t`.
        prediction_type : str
            The type of prediction made by the model, which determines how to convert it to `x`.
            Must be one of {'velocity', 'noise', 'F', 'x', 'score'}.

        Returns
        -------
        Tensor
            The reconstructed clean signal `x` from the model prediction.
        """

        match prediction_type:
            case "velocity":
                return alpha_t * z - sigma_t * pred

            case "noise":
                return (z - sigma_t * pred) / alpha_t

            case "F":
                sigma_data = getattr(self.noise_schedule, "sigma_data", 1.0)
                x1 = (sigma_data**2 * alpha_t) / (ops.exp(-log_snr_t) + sigma_data**2)
                x2 = ops.exp(-log_snr_t / 2) * sigma_data / ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2)
                return x1 * z + x2 * pred

            case "x":
                return pred

            case "score":
                return (z + sigma_t**2 * pred) / alpha_t

            case "potential":
                # Score is computed as ∇_z phi via _compute_grad_of_potential.
                raise RuntimeError("convert_prediction_to_x should not be called for prediction_type='potential'. ")

            case _:
                raise ValueError(f"Unknown prediction type {prediction_type}.")

    def score(
        self,
        xz: Tensor,
        time: float | Tensor = None,
        log_snr_t: Tensor = None,
        conditions: Tensor = None,
        training: bool = False,
        guidance_kwargs: Mapping[str, Any] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the score of the target or latent variable `xz`.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable `z`, typically of shape (..., D),
            where D is the dimensionality of the latent space.
        time : float or Tensor
            Scalar or tensor representing the time (or noise level) at which the velocity
            should be computed. Will be broadcasted to xz. If None, log_snr_t must be provided.
        log_snr_t : Tensor
            The log signal-to-noise ratio at time `t`. If None, time must be provided.
        conditions : Tensor, optional
            Conditional inputs to the network, such as conditioning variables
            or encoder outputs. Shape must be broadcastable with `xz`. Default is None.
        training : bool, optional
            Whether the model is in training mode. Affects behavior of dropout, batch norm,
            or other stochastic layers. Default is False.
        guidance_kwargs : Mapping[str, Any], optional
            A dictionary of parameters for computing a guidance constraint term, which is
            added to the score for guided sampling. The specific keys and values depend on
            the implementation of `guidance_constraint_term`.
        **kwargs
            Subnet kwargs (e.g., attention_mask, mask) for the subnet layer.
            Pass built-in constraint settings through ``guidance_kwargs`` or override
            :meth:`guidance_function` for custom guidance.

        Returns
        -------
        Tensor
            The velocity tensor of the same shape as `xz`, representing the right-hand
            side of the probability-flow SDE or ODE at the given `time`.
        """
        subnet_kwargs = self._collect_mask_kwargs(self._SUBNET_MASK_KEYS, kwargs)

        if log_snr_t is None:
            log_snr_t = self.noise_schedule.get_log_snr(t=time, training=training)
            log_snr_t = expand_right_as(log_snr_t, xz)
            log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        if time is None:
            time = self.noise_schedule.get_t_from_log_snr(log_snr_t, training=training)

        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

        norm_log_snr = self._transform_log_snr(log_snr_t)

        if self._prediction_type == "potential":
            pred = self._compute_grad_of_potential(
                xz=xz, norm_log_snr=norm_log_snr, conditions=conditions, training=training, **subnet_kwargs
            )
            x_pred = self.convert_prediction_to_x(
                pred=pred, z=xz, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t, prediction_type="velocity"
            )
        else:
            subnet_out = self.subnet((xz, norm_log_snr, conditions), training=training, **subnet_kwargs)
            pred = self.output_projector(subnet_out)

            x_pred = self.convert_prediction_to_x(
                pred=pred,
                z=xz,
                alpha_t=alpha_t,
                sigma_t=sigma_t,
                log_snr_t=log_snr_t,
                prediction_type=self._prediction_type,
            )
        score = (alpha_t * x_pred - xz) / ops.square(sigma_t)

        score = self.guidance_function(x_pred=x_pred, time=time, score=score, **(guidance_kwargs or {}))
        return score

    def velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        stochastic_solver: bool,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Computes the velocity (i.e., time derivative) of the target or latent variable `xz` for either
        a stochastic differential equation (SDE) or ordinary differential equation (ODE).

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable `z`, typically of shape (..., D),
            where D is the dimensionality of the latent space.
        time : float or Tensor
            Scalar or tensor representing the time (or noise level) at which the velocity
            should be computed. Will be broadcasted to xz.
        stochastic_solver : bool
            If True, computes the velocity for the stochastic formulation (SDE).
            If False, uses the deterministic formulation (ODE).
        conditions : Tensor, optional
            Conditional inputs to the network, such as conditioning variables
            or encoder outputs. Shape must be broadcastable with `xz`. Default is None.
        training : bool, optional
            Whether the model is in training mode. Affects behavior of dropout, batch norm,
            or other stochastic layers. Default is False.

        Returns
        -------
        Tensor
            The velocity tensor of the same shape as `xz`, representing the right-hand
            side of the SDE or ODE at the given `time`.
        """

        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        score = self.score(xz, log_snr_t=log_snr_t, conditions=conditions, training=training, **kwargs)

        # compute velocity f, g of the SDE or ODE
        f, g_squared = self.noise_schedule.get_drift(log_snr_t=log_snr_t, x=xz, training=training)

        if stochastic_solver:
            # for the SDE: d(z) = [f(z, t) - g(t) ^ 2 * score(z, lambda )] dt + g(t) dW
            out = f - g_squared * score
        else:
            # for the ODE: d(z) = [f(z, t) - 0.5 * g(t) ^ 2 * score(z, lambda )] dt
            out = f - 0.5 * g_squared * score

        # Zero out velocity where target is fixed (during inference only)
        if not training:
            target_mask = kwargs.get("target_mask", None)
            out = maybe_mask_tensor(out, mask=target_mask)

        return out

    def diffusion_term(
        self,
        xz: Tensor,
        time: float | Tensor,
        training: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Compute the diffusion term (standard deviation of the noise) at a given time.

        Parameters
        ----------
        xz : Tensor
            Input tensor of shape (..., D), typically representing the target or latent variables at given time.
        time : float or Tensor
            The diffusion time step(s). Can be a scalar or a tensor broadcastable to the shape of `xz`.
        training : bool, optional
            Whether to use the training noise schedule (default is False).

        Returns
        -------
        Tensor
            The diffusion term tensor with shape matching `xz` except for the last dimension, which is set to 1.
        """
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))
        g_squared = self.noise_schedule.get_drift(log_snr_t=log_snr_t)
        g = ops.sqrt(g_squared)

        # Zero out diffusion where target is fixed (during inference only)
        if not training:
            target_mask = kwargs.get("target_mask", None)
            g = maybe_mask_tensor(g, mask=target_mask)

        return g

    def _velocity_trace(
        self,
        xz: Tensor,
        time: Tensor,
        conditions: Tensor = None,
        max_steps: int = None,
        training: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        def f(x):
            return self.velocity(
                x, time=time, stochastic_solver=False, conditions=conditions, training=training, **kwargs
            )

        v, trace = jacobian_trace(f, xz, max_steps=max_steps, seed=self.seed_generator, return_output=True)

        return v, ops.expand_dims(trace, axis=-1)

    def _compute_grad_of_potential(
        self,
        xz: Tensor,
        norm_log_snr: Tensor,
        conditions: Tensor = None,
        training: bool = False,
        **subnet_kwargs,
    ) -> Tensor:
        """
        Compute the gradient of the predicted potential phi w.r.t. xz.

        The network outputs a scalar potential phi(z, t, cond) per sample. The output is
        then defined as ∇_z phi, i.e. the gradient of the potential w.r.t. the noisy input.

        Parameters
        ----------
        xz : Tensor
            The noisy latent variable z at which to evaluate the score.
        norm_log_snr : Tensor
            Normalised log-SNR fed to the subnet as the time embedding.
        conditions : Tensor, optional
            Conditioning variables passed to the subnet. Default is None.
        training : bool, optional
            Whether the subnet is in training mode.
        **subnet_kwargs
            Additional keyword arguments forwarded to the subnet (e.g. attention masks).

        Returns
        -------
        Tensor
            Output of the same shape as *xz*, equal to some parameterization of ∇_xz phi(xz, t, cond).
        """
        backend = keras.backend.backend()

        match backend:
            case "jax":
                import jax
                import jax.numpy as jnp

                def phi_fn(x):
                    subnet_out = self.subnet(
                        (x, norm_log_snr, conditions),
                        training=training,
                        **subnet_kwargs,
                    )
                    return self.output_projector(subnet_out)

                phi, vjp_fn = jax.vjp(phi_fn, xz)
                out = vjp_fn(jnp.ones_like(phi))[0]
                return out

            case "tensorflow":
                import tensorflow as tf

                with tf.GradientTape() as tape:
                    tape.watch(xz)
                    subnet_out = self.subnet(
                        (xz, norm_log_snr, conditions),
                        training=training,
                        **subnet_kwargs,
                    )
                    phi = self.output_projector(subnet_out)

                out = tape.gradient(
                    target=phi,
                    sources=xz,
                    output_gradients=tf.ones_like(phi),
                )
                return out

            case "torch":
                import torch

                with torch.enable_grad():
                    xz_leaf = xz.requires_grad_(True)

                    subnet_out = self.subnet(
                        (xz_leaf, norm_log_snr, conditions),
                        training=training,
                        **subnet_kwargs,
                    )
                    phi = self.output_projector(subnet_out)

                    out = torch.autograd.grad(
                        outputs=phi,
                        inputs=xz_leaf,
                        grad_outputs=torch.ones_like(phi),
                        create_graph=training,
                        retain_graph=training,
                    )[0]

                return out

            case _:
                raise NotImplementedError(f"Unsupported backend for potential prediction type: {backend}")

    def _transform_log_snr(self, log_snr: Tensor) -> Tensor:
        """Transform the log_snr to the range [-1, 1] for the diffusion process."""
        log_snr_min = self.noise_schedule.log_snr_min
        log_snr_max = self.noise_schedule.log_snr_max
        normalized_snr = (log_snr - log_snr_min) / (log_snr_max - log_snr_min)
        scaled_value = 2 * normalized_snr - 1
        return scaled_value

    def _forward(
        self,
        x: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Note: integrators will cherry-pick necessary kwargs, so
        # we can be general (i.e., sloppy) here
        integrate_kwargs = {"start_time": 0.0, "stop_time": 1.0}
        integrate_kwargs |= self.integrate_kwargs
        integrate_kwargs |= kwargs

        if integrate_kwargs["method"] in STOCHASTIC_METHODS:
            logging.warning(
                "Stochastic methods are not supported for density evaluation."
                " Falling back to tsit5 ODE solver."
                " To suppress this warning, explicitly pass a method from " + str(DETERMINISTIC_METHODS) + "."
            )
            integrate_kwargs["method"] = "tsit5"

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(x))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(x))
            x = maybe_mask_tensor(x, target_mask, replacement=targets_fixed)

        if self.unconditional_mode and conditions is not None:
            conditions = keras.ops.zeros_like(conditions)
            logging.info("Condition masking is applied: conditions are set to zero.")

        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training, **kwargs)
                return {"xz": v, "trace": trace}

            state = {
                "xz": x,
                "trace": ops.zeros(ops.shape(x)[:-1] + (1,), dtype=ops.dtype(x)),
            }
            state = integrate(deltas, state, **integrate_kwargs)

            z = state["xz"]
            log_density = self.base_distribution.log_prob(z) + ops.squeeze(state["trace"], axis=-1)

            return z, log_density

        def deltas(time, xz):
            return {
                "xz": self.velocity(
                    xz, time=time, stochastic_solver=False, conditions=conditions, training=training, **kwargs
                )
            }

        state = {"xz": x}
        state = integrate(deltas, state, **integrate_kwargs)
        z = state["xz"]
        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        seed = resolve_seed(kwargs.pop("seed", None)) or self.seed_generator

        # Build integrate kwargs: hardcoded defaults -> instance config -> call-time overrides
        integrate_kwargs = {"start_time": 1.0, "stop_time": 0.0}
        integrate_kwargs |= self.integrate_kwargs
        integrate_kwargs |= kwargs

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(z))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(z))
            z = maybe_mask_tensor(z, target_mask, replacement=targets_fixed)

        if self.unconditional_mode and conditions is not None:
            conditions = keras.ops.zeros_like(conditions)
            logging.info("Condition masking is applied: conditions are set to zero.")

        if density:
            if integrate_kwargs["method"] in STOCHASTIC_METHODS:
                logging.warning(
                    "Stochastic methods are not supported for density computation."
                    " Falling back to ODE solver."
                    " Use one of the deterministic methods: " + str(DETERMINISTIC_METHODS) + "."
                )
                integrate_kwargs["method"] = "tsit5"

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training, **kwargs)
                return {"xz": v, "trace": trace}

            state = {
                "xz": z,
                "trace": ops.zeros(ops.shape(z)[:-1] + (1,), dtype=ops.dtype(z)),
            }
            state = integrate(deltas, state, **integrate_kwargs)

            x = state["xz"]
            log_density = self.base_distribution.log_prob(z) - ops.squeeze(state["trace"], axis=-1)

            return x, log_density

        state = {"xz": z}

        if integrate_kwargs["method"] in STOCHASTIC_METHODS:

            def deltas(time, xz):
                return {
                    "xz": self.velocity(
                        xz,
                        time=time,
                        stochastic_solver=integrate_kwargs["method"] != "glass",
                        conditions=conditions,
                        training=training,
                        **kwargs,
                    )
                }

            def diffusion(time, xz):
                return {"xz": self.diffusion_term(xz, time=time, training=training, **kwargs)}

            score_fn = None
            if "corrector_steps" in integrate_kwargs or integrate_kwargs["method"] == "langevin":

                def score_fn(time, xz):
                    return {"xz": self.score(xz, time=time, conditions=conditions, training=training, **kwargs)}

            state = integrate_stochastic(
                drift_fn=deltas,
                diffusion_fn=diffusion,
                score_fn=score_fn,
                noise_schedule=self.noise_schedule,
                state=state,
                seed=seed,
                **integrate_kwargs,
            )
        else:

            def deltas(time, xz):
                return {
                    "xz": self.velocity(
                        xz, time=time, stochastic_solver=False, conditions=conditions, training=training, **kwargs
                    )
                }

            state = integrate(deltas, state, **integrate_kwargs)

        x = state["xz"]
        return x

    def compositional_bridge(
        self,
        time: Tensor,
        d0: float | None = None,
        d1: float | None = None,
    ) -> Tensor:
        """Exponential damping schedule for the compositional score.

        Sampling-time overrides are passed explicitly and never mutate the model.
        The model attributes remain available as configurable defaults.

        Parameters
        ----------
        time: Tensor
            Time step for the diffusion process.
        d0, d1 : float, optional
            Positive damping values at clean and noisy time, respectively.

        Returns
        -------
        Tensor
            Bridge function value with same shape as time.

        """
        d0 = self.compositional_bridge_d0 if d0 is None else d0
        d1 = self.compositional_bridge_d1 if d1 is None else d1
        for name, value in (("d0", d0), ("d1", d1)):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        return d0 * ops.exp(-ops.log(d0 / d1) * time)

    def compositional_velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        stochastic_solver: bool,
        conditions: Tensor,
        seed: keras.random.SeedGenerator,
        compute_prior_score: Callable[[Tensor], Tensor] | Literal["standard_normal"] | None = None,
        mini_batch_size: int | None = None,
        training: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Computes the compositional velocity using the error-damped mini-batch
        score estimator of Arruda et al. (2026).

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (num_datasets, num_items, ...)
        time : float or Tensor
            Time step for the diffusion process
        stochastic_solver : bool
            Whether to use stochastic (SDE) or deterministic (ODE) formulation
        conditions : Tensor
            Conditional inputs with compositional structure (num_datasets, num_items, ...)
        seed: SeedGenerator
            For reproducibility.
        compute_prior_score: Callable or {"standard_normal"}, optional
            Function computing the noisy-prior score. Otherwise, a trained
            unconditional score is used.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode
        **kwargs
            Additional keyword arguments passed to the individual score computation.

        Returns
        -------
        Tensor
            Compositional velocity of same shape as input xz
        """
        compositional_score = self.compositional_score(
            xz=xz,
            time=time,
            conditions=conditions,
            compute_prior_score=compute_prior_score,
            mini_batch_size=mini_batch_size,
            training=training,
            seed=seed,
            **kwargs,
        )

        # Calculate standard noise schedule components
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        # Compute velocity using standard drift-diffusion formulation
        f, g_squared = self.noise_schedule.get_drift(log_snr_t=log_snr_t, x=xz, training=training)

        if stochastic_solver:
            # for the SDE: d(z) = [f(z, t) - g(t) ^ 2 * score(z, lambda )] dt + g(t) dW
            out = f - g_squared * compositional_score
        else:
            # for the ODE: d(z) = [f(z, t) - 0.5 * g(t) ^ 2 * score(z, lambda )] dt
            out = f - 0.5 * g_squared * compositional_score

        # Zero out velocity where target is fixed (during inference only)
        if not training:
            target_mask = kwargs.get("target_mask", None)
            out = maybe_mask_tensor(out, mask=target_mask)

        return out

    def compositional_score(
        self,
        xz: Tensor,
        time: float | Tensor,
        conditions: Tensor,
        seed: keras.random.SeedGenerator | None,
        compute_prior_score: Callable[[Tensor], Tensor] | Literal["standard_normal"] | None = None,
        mini_batch_size: int = None,
        training: bool = False,
        clip: tuple[float, float] | None = None,
        use_jac: bool = False,
        guidance_kwargs: Mapping[str, Any] = None,
        item_indices: Tensor | None = None,
        compositional_bridge_d0: float | None = None,
        compositional_bridge_d1: float | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the compositional score for multiple datasets.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (num_datasets, num_items, ...)
        time : float or Tensor
            Time step for the diffusion process
        conditions : Tensor
            Conditional inputs with compositional structure (num_datasets, num_items, ...)
        seed: keras.random.SeedGenerator or None
            Optional seed for reproducibility.
        compute_prior_score: Callable or {"standard_normal"}, optional
            Function computing the noisy-prior score. Use ``"standard_normal"`` for the exact diffused score of a
            standard-normal clean prior. Otherwise, an unconditional score is used and must have been trained with
            condition dropout.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode
        clip: (float, float), optional
            Optional clean-state clipping interval. Disabled by default because clipping changes the composed score.
        use_jac: bool, optional
            Whether to use the Jacobian-based compositional score instead of the direct sum.
        guidance_kwargs : Mapping[str, Any], optional
            A dictionary of parameters for computing a guidance constraint term, which is
            added to the score for guided sampling. The specific keys and values depend on
            the implementation of `guidance_function`.
        **kwargs
            Additional keyword arguments passed to the individual score computation

        Returns
        -------
        Tensor
            Compositional velocity of same shape as input xz
        """
        if conditions is None:
            raise ValueError("Conditions are required for compositional sampling")

        if compute_prior_score is None and self.drop_cond_prob <= 0:
            raise ValueError(
                "Cannot estimate the prior score from an unconditional network when drop_cond_prob=0. "
                "Pass compute_prior_score='standard_normal', provide an exact noisy-prior score callback, "
                "or train the diffusion model with condition dropout."
            )

        if use_jac and callable(compute_prior_score):
            raise ValueError(
                "use_jac=True cannot infer the prior-score Jacobian from an adapted callback. "
                "Use compute_prior_score='standard_normal' or a model trained with condition dropout."
            )

        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)
        time = ops.cast(time, dtype=ops.dtype(xz))

        if not use_jac:
            compositional_score = self._compositional_score_direct(
                xz=xz,
                time=time,
                log_snr_t=log_snr_t,
                conditions=conditions,
                compute_prior_score=compute_prior_score,
                mini_batch_size=mini_batch_size,
                training=training,
                seed=seed,
                item_indices=item_indices,
                **kwargs,
            )
        else:
            # uses a Gaussian approximation, can be very slow
            compositional_score = self._compositional_score_jac(
                xz=xz,
                time=time,
                log_snr_t=log_snr_t,
                sigma_t=sigma_t,
                conditions=conditions,
                compute_prior_score=compute_prior_score,
                mini_batch_size=mini_batch_size,
                training=training,
                seed=seed,
                item_indices=item_indices,
                **kwargs,
            )

        damping = self.compositional_bridge(
            time,
            d0=compositional_bridge_d0,
            d1=compositional_bridge_d1,
        )
        compositional_score = expand_right_as(damping, compositional_score) * compositional_score

        x_pred = (xz + sigma_t**2 * compositional_score) / alpha_t
        compositional_score = self.guidance_function(
            x_pred=x_pred, time=time, score=compositional_score, **(guidance_kwargs or {})
        )

        compositional_score = self._maybe_clip_score(compositional_score, clip, alpha_t, sigma_t, xz)
        return compositional_score

    @staticmethod
    def _select_compositional_conditions(
        conditions: Tensor,
        mini_batch_size: int | None,
        item_indices: Tensor | None,
        seed: keras.random.SeedGenerator | None,
    ) -> tuple[Tensor, int | Tensor]:
        """Select a factor mini-batch, optionally using solver-scoped indices."""
        batch_size, num_items = ops.shape(conditions)[:2]

        if item_indices is not None:
            item_indices = ops.cast(item_indices, "int32")
            if ops.ndim(item_indices) == 1:
                conditions_batch = ops.take(conditions, item_indices, axis=1)
                return conditions_batch, ops.shape(item_indices)[0]
            if ops.ndim(item_indices) == 2:
                expanded_indices = item_indices
                for _ in range(ops.ndim(conditions) - 2):
                    expanded_indices = ops.expand_dims(expanded_indices, axis=-1)
                expanded_indices = ops.broadcast_to(
                    expanded_indices,
                    (batch_size, ops.shape(item_indices)[1]) + tuple(ops.shape(conditions)[2:]),
                )
                conditions_batch = ops.take_along_axis(conditions, expanded_indices, axis=1)
                return conditions_batch, ops.shape(item_indices)[1]
            raise ValueError("item_indices must have shape (M,) or (batch_size, M).")

        if mini_batch_size is not None and mini_batch_size < num_items:
            ranks = keras.random.uniform((batch_size, num_items), seed=seed)
            _, per_row_idx = ops.top_k(-ranks, mini_batch_size)
            expanded_indices = per_row_idx
            for _ in range(ops.ndim(conditions) - 2):
                expanded_indices = ops.expand_dims(expanded_indices, axis=-1)
            expanded_indices = ops.broadcast_to(
                expanded_indices,
                (batch_size, mini_batch_size) + tuple(ops.shape(conditions)[2:]),
            )
            return ops.take_along_axis(conditions, expanded_indices, axis=1), mini_batch_size

        return conditions, num_items

    def _standard_normal_prior_score(self, xz: Tensor, log_snr_t: Tensor) -> Tensor:
        """Exact noisy score for a standard-normal clean prior."""
        _, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)
        if self.noise_schedule._variance_type == "preserving":
            variance_t = ops.ones_like(sigma_t)
        else:
            variance_t = 1 + ops.square(sigma_t)
        return -xz / variance_t

    @staticmethod
    def _minimum_symmetric_eigenvalue(matrix: Tensor) -> Tensor:
        """Return the smallest eigenvalue, with a CPU fallback for torch MPS."""
        if keras.backend.backend() == "torch" and matrix.device.type == "mps":
            import torch

            return torch.linalg.eigvalsh(matrix.cpu())[..., 0].to(matrix.device)
        eigenvalues, _ = ops.linalg.eigh(matrix)
        return eigenvalues[..., 0]

    def _compositional_score_direct(
        self,
        xz: Tensor,
        time: float | Tensor,
        log_snr_t: Tensor,
        conditions: Tensor,
        seed: keras.random.SeedGenerator = None,
        compute_prior_score: Callable[[Tensor], Tensor] | Literal["standard_normal"] | None = None,
        mini_batch_size: int = None,
        training: bool = False,
        item_indices: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the undamped mini-batch estimator of Arruda et al. (2026),

        ``(1-J)(1-t)s0 + (J/M) sum_m s_m``.

        The public score method subsequently multiplies this expression by the
        optional damping schedule ``d(t)``.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (num_datasets, num_items, ...)
        time : float or Tensor
            Time step for the diffusion process.
        log_snr_t : Tensor
            Log SNR at time t, broadcastable to shape of xz.
        conditions : Tensor
            Conditional inputs with compositional structure (num_datasets, num_items, ...)
        seed: keras.random.SeedGenerator or None
            Optional seed for reproducibility.
        compute_prior_score: Callable or {"standard_normal"}, optional
            Function computing the noisy-prior score. Otherwise, the unconditional score is estimated.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode.
        **kwargs
            Additional keyword arguments passed to the individual score computation.

        Returns
        -------
        Tensor
            Compositional score of same shape as input xz
        """

        batch_size, num_items = ops.shape(conditions)[:2]
        conditions_batch, mini_batch_size = self._select_compositional_conditions(
            conditions=conditions,
            mini_batch_size=mini_batch_size,
            item_indices=item_indices,
            seed=seed,
        )

        # Determine scale of summed posterior score
        needs_network_prior = compute_prior_score is None
        if needs_network_prior:
            zero_cond = ops.zeros_like(ops.take(conditions, 0, axis=1))
            cond_with_prior = ops.concatenate([conditions_batch, ops.expand_dims(zero_cond, 1)], axis=1)
            num_total = mini_batch_size + 1
        else:
            cond_with_prior = conditions_batch
            num_total = mini_batch_size
        scale = num_items / mini_batch_size

        # Expand and flatten compositional dimension (i.e., num items) for score computation
        dims = tuple(ops.shape(xz)[1:])
        snr_dims = tuple(ops.shape(log_snr_t)[1:])
        conditions_dims = tuple(ops.shape(cond_with_prior)[2:])
        xz_reshaped = ops.reshape(
            ops.repeat(ops.expand_dims(xz, 1), num_total, axis=1), (batch_size * num_total,) + dims
        )
        log_snr_reshaped = ops.reshape(
            ops.repeat(ops.expand_dims(log_snr_t, 1), num_total, axis=1),
            (batch_size * num_total,) + snr_dims,
        )
        conditions_flat = ops.reshape(cond_with_prior, (batch_size * num_total,) + conditions_dims)
        scores_flat = self.score(
            xz_reshaped,
            log_snr_t=log_snr_reshaped,
            conditions=conditions_flat,
            training=training,
            **kwargs,
        )
        all_scores = ops.reshape(scores_flat, (batch_size, num_total) + dims)
        individual_scores = all_scores[:, :mini_batch_size]

        if needs_network_prior:
            prior_score = all_scores[:, -1]
        elif compute_prior_score == "standard_normal":
            prior_score = self._standard_normal_prior_score(xz, log_snr_t)
        else:
            prior_score = compute_prior_score(xz, time)

        # Eq. (8) of Arruda et al. (2026), written as a residual sum to
        # reduce cancellation near clean time.
        weighted_prior_score = expand_right_as(1 - time, prior_score) * prior_score
        delta = individual_scores - ops.expand_dims(weighted_prior_score, axis=1)
        update_delta = scale * ops.sum(delta, axis=1)
        compositional_score = weighted_prior_score + update_delta

        return compositional_score

    def _compositional_score_jac(
        self,
        xz: Tensor,
        time: float | Tensor,
        log_snr_t: Tensor,
        sigma_t: Tensor,
        conditions: Tensor,
        seed: keras.random.SeedGenerator = None,
        compute_prior_score: Callable[[Tensor], Tensor] | Literal["standard_normal"] | None = None,
        mini_batch_size: int = None,
        training: bool = False,
        regularize_precision: float = 1e-6,
        item_indices: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Stabilized version of the JAC compositional score (Linhart et al. 2026).

        Precision-weighted compositional posterior score where both individual posterior precisions
        and the prior precision are estimated via the Jacobian of the score network.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (num_datasets, num_items, ...)
        time : float or Tensor
            Time step for the diffusion process.
        log_snr_t : Tensor
            Log SNR at time t, broadcastable to shape of xz.
        sigma_t : Tensor
            Sigma component of noise schedule at time t, broadcastable to shape of xz.
        conditions : Tensor
            Conditional inputs with compositional structure (num_datasets, num_items, ...)
        seed: SeedGenerator or None
            Optional seed for reproducibility.
        compute_prior_score: Callable or {"standard_normal"}, optional
            Prior specification used by the composition. Callable priors are rejected because JAC also needs their
            Jacobian.
        mini_batch_size : int or None, optional
            Must be ``None`` or include all conditions. Subsampling is not
            supported because inversion makes the JAC mini-batch estimator biased.
        training : bool
            Whether in training mode.
        regularize_precision : float
            Minimum eigenvalue used when stabilizing precision systems.
        **kwargs
            Additional keyword arguments passed to the individual score computation.

        Returns
        -------
        Tensor
            Compositional score of same shape as input xz
        """
        batch_size, num_items = ops.shape(conditions)[:2]
        num_items_int = int(ops.convert_to_numpy(num_items))
        m = ops.shape(xz)[-1]
        upsilon_t = ops.maximum(ops.reshape(sigma_t, (batch_size, -1))[:, 0] ** 2, 1e-8)

        if callable(compute_prior_score):
            raise ValueError(
                "JAC composition requires the prior-score Jacobian. "
                "Use compute_prior_score='standard_normal' or a condition-dropped network prior."
            )

        conditions_batch, mini_batch_size = self._select_compositional_conditions(
            conditions=conditions,
            mini_batch_size=mini_batch_size,
            item_indices=item_indices,
            seed=seed,
        )
        if int(ops.convert_to_numpy(mini_batch_size)) != num_items_int:
            raise ValueError(
                "JAC composition does not support factor mini-batching because solving a subsampled "
                "random precision system is biased. Set mini_batch_size=None."
            )

        uses_network_prior = compute_prior_score is None
        if uses_network_prior:
            zero_cond = ops.zeros_like(ops.take(conditions, 0, axis=1))
            cond_for_network = ops.concatenate([conditions_batch, ops.expand_dims(zero_cond, 1)], axis=1)
            num_total = mini_batch_size + 1
        else:
            cond_for_network = conditions_batch
            num_total = mini_batch_size

        dims = tuple(ops.shape(xz)[1:])
        snr_dims = tuple(ops.shape(log_snr_t)[1:])
        cond_dims = tuple(ops.shape(cond_for_network)[2:])

        # Repeat xz and log_snr_t for each observation+prior
        xz_rep = ops.reshape(
            ops.repeat(ops.expand_dims(xz, 1), num_total, axis=1),
            (batch_size * num_total,) + dims,
        )
        lsnr_rep = ops.reshape(
            ops.repeat(ops.expand_dims(log_snr_t, 1), num_total, axis=1),
            (batch_size * num_total,) + snr_dims,
        )
        cond_flat = ops.reshape(
            cond_for_network,
            (batch_size * num_total,) + cond_dims,
        )

        all_scores, all_jacs = self._compute_score_and_jacobian(
            xz=xz_rep,
            log_snr_t=lsnr_rep,
            conditions=cond_flat,
            training=training,
            **kwargs,
        )

        # Reshape: (B, num_total, m) and (B, num_total, m, m)
        all_scores = ops.reshape(all_scores, (batch_size, num_total, m))
        all_jacs = ops.reshape(all_jacs, (batch_size, num_total, m, m))
        factor_mean_score = ops.mean(all_scores[:, :mini_batch_size], axis=1)
        if num_items_int == 1:
            return factor_mean_score

        # compute P_k = (I + υ J_k)⁻¹ for all k at once
        I_m = ops.eye(m, dtype=ops.dtype(xz))
        I_4d = ops.broadcast_to(
            ops.reshape(I_m, (1, 1, m, m)),
            (batch_size, num_total, m, m),
        )
        all_jacs = 0.5 * (all_jacs + ops.swapaxes(all_jacs, -1, -2))
        A_all = I_4d + upsilon_t[:, None, None, None] * all_jacs
        min_eigenvalue = self._minimum_symmetric_eigenvalue(A_all)
        diagonal_shift = ops.maximum(regularize_precision - min_eigenvalue, 0.0)
        A_all = A_all + diagonal_shift[..., None, None] * I_4d
        P_all = ops.inv(A_all)

        # P_k @ s_k for all k: (B, num_total, m, 1) → (B, num_total, m)
        scores_col = ops.expand_dims(all_scores, -1)
        Ps_all = ops.squeeze(ops.matmul(P_all, scores_col), axis=-1)  # (B, num_total, m)

        # Split observations and, when needed, the learned unconditional prior.
        P_obs = P_all[:, :mini_batch_size]  # (B, n_obs, m, m)
        Ps_obs = Ps_all[:, :mini_batch_size]  # (B, n_obs, m)

        if uses_network_prior:
            prior_score = all_scores[:, -1]
            P_lambda = P_all[:, -1]
            P_lambda_s = Ps_all[:, -1]  # already computed
        else:
            prior_score = self._standard_normal_prior_score(xz, log_snr_t)
            if self.noise_schedule._variance_type == "preserving":
                prior_variance = ops.ones_like(upsilon_t)
            else:
                prior_variance = 1 + upsilon_t
            prior_jacobian_scale = -1 / prior_variance
            prior_backward_scale = ops.maximum(1 + upsilon_t * prior_jacobian_scale, 1e-8)
            P_lambda = I_m[None, ...] / prior_backward_scale[:, None, None]
            P_lambda_s = ops.squeeze(ops.matmul(P_lambda, ops.expand_dims(prior_score, -1)), axis=-1)

        # Algebraically equivalent to sum(P_i) + (1-J)P_0, but avoids
        # cancellation between O(J / alpha_t^2) terms near the noisy endpoint.
        I_m_batch = ops.broadcast_to(I_m, (batch_size, m, m))
        delta_P = P_obs - P_lambda[:, None, :, :]
        delta_Ps = Ps_obs - P_lambda_s[:, None, :]
        lambda_matrix = P_lambda + ops.sum(delta_P, axis=1)
        s_tilde = P_lambda_s + ops.sum(delta_Ps, axis=1)

        lambda_matrix = 0.5 * (lambda_matrix + ops.swapaxes(lambda_matrix, -1, -2))
        min_eigenvalue = self._minimum_symmetric_eigenvalue(lambda_matrix)
        final_shift = ops.maximum(regularize_precision - min_eigenvalue, 0.0)
        lambda_matrix = lambda_matrix + final_shift[:, None, None] * I_m_batch
        s_tilde = s_tilde + final_shift[:, None] * factor_mean_score
        compositional_score = linsolve_batched(lambda_matrix, s_tilde)
        return compositional_score

    def _compute_score_and_jacobian(
        self,
        xz: Tensor,
        log_snr_t: Tensor,
        conditions: Tensor,
        training: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Compute a score and its full Jacobian ``J = ∇_{θ_t} s(θ_t)`` for one condition.

        Returns
        -------
        s : Tensor (B, m)
        jacobian : Tensor (B, m, m)
            Full Jacobian of the score with respect to ``xz``.
        """
        m = ops.shape(xz)[-1]
        backend = keras.backend.backend()

        match backend:
            case "jax":
                import jax
                import jax.numpy as jnp

                def phi_fn(x):
                    return self.score(
                        xz=x,
                        log_snr_t=log_snr_t,
                        conditions=conditions,
                        training=training,
                        **kwargs,
                    )

                score, vjp_fn = jax.vjp(phi_fn, xz)

                rows = []
                for i in range(m):
                    e_i = jnp.zeros_like(score).at[:, i].set(1.0)
                    (row_i,) = vjp_fn(e_i)
                    rows.append(jnp.expand_dims(row_i, axis=1))

                jacobian = jnp.concatenate(rows, axis=1)

            case "tensorflow":
                import tensorflow as tf

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(xz)
                    score = self.score(
                        xz=xz,
                        log_snr_t=log_snr_t,
                        conditions=conditions,
                        training=training,
                        **kwargs,
                    )

                if m <= 32:  # for larger dimension
                    rows = []
                    for i in range(m):
                        e_i = tf.one_hot(tf.fill([tf.shape(score)[0]], i), depth=m, dtype=score.dtype)
                        row_i = tape.gradient(score, xz, output_gradients=e_i)
                        rows.append(tf.expand_dims(row_i, axis=1))
                    jacobian = tf.concat(rows, axis=1)
                else:
                    jacobian = tape.batch_jacobian(score, xz, experimental_use_pfor=False)

                del tape

            case "torch":
                import torch

                with torch.enable_grad():
                    xz_leaf = xz.detach().requires_grad_(True)

                    score = self.score(
                        xz=xz_leaf,
                        log_snr_t=log_snr_t,
                        conditions=conditions,
                        training=training,
                        **kwargs,
                    )

                    rows = []
                    for i in range(m):
                        e_i = torch.zeros_like(score)
                        e_i[:, i] = 1.0
                        (row_i,) = torch.autograd.grad(
                            outputs=score,
                            inputs=xz_leaf,
                            grad_outputs=e_i,
                            create_graph=False,
                            retain_graph=(i < m - 1),
                        )
                        rows.append(row_i.unsqueeze(1))

                    jacobian = torch.cat(rows, dim=1)

                score = score.detach()

            case _:
                raise NotImplementedError(f"JAC Jacobian not implemented for backend '{backend}'. ")
        return score, jacobian

    @staticmethod
    def _maybe_clip_score(compositional_score, clip, alpha_t, sigma_t, xz):
        if clip is not None:
            min_clip, max_clip = clip
            x_pred = (xz + sigma_t**2 * compositional_score) / alpha_t
            x_pred = ops.clip(x_pred, min_clip, max_clip)
            compositional_score = (x_pred * alpha_t - xz) / (sigma_t**2)
        return compositional_score

    def _inverse_compositional(
        self,
        z: Tensor,
        conditions: Tensor,
        compute_prior_score: Callable[[Tensor], Tensor] | Literal["standard_normal"] | None = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Inverse pass for compositional diffusion sampling.
        """
        seed = resolve_seed(kwargs.pop("seed", None)) or self.seed_generator
        integrate_kwargs = {"start_time": 1.0, "stop_time": 0.0}
        integrate_kwargs |= self.integrate_kwargs
        integrate_kwargs |= kwargs

        num_items = ops.shape(conditions)[1]
        num_items_int = int(ops.convert_to_numpy(num_items))
        use_jac = bool(integrate_kwargs.get("use_jac", False))
        default_mini_batch_size = num_items_int if use_jac else max(int(num_items_int * 0.1), 2)
        mini_batch_size_arg = integrate_kwargs.pop("mini_batch_size", default_mini_batch_size)
        if mini_batch_size_arg is None:
            mini_batch_size = num_items_int
        else:
            mini_batch_size = int(mini_batch_size_arg)
            if mini_batch_size <= 0:
                raise ValueError("mini_batch_size must be positive or None.")
            mini_batch_size = min(mini_batch_size, num_items_int)

        score_kwargs = dict(kwargs)
        score_kwargs.pop("mini_batch_size", None)

        if keras.backend.backend() == "jax" and mini_batch_size != num_items_int:
            mini_batch_size = num_items_int
            logging.warning("Setting mini_batch_size to num_items as jax does not support mini-batching.")

        compositional_bridge_d0 = float(integrate_kwargs.pop("compositional_bridge_d0", self.compositional_bridge_d0))
        compositional_bridge_d1 = float(integrate_kwargs.pop("compositional_bridge_d1", self.compositional_bridge_d1))
        score_kwargs.pop("compositional_bridge_d0", None)
        score_kwargs.pop("compositional_bridge_d1", None)
        for name, value in (
            ("compositional_bridge_d0", compositional_bridge_d0),
            ("compositional_bridge_d1", compositional_bridge_d1),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        if compositional_bridge_d0 != 1.0:
            logging.warning(
                "compositional_bridge_d0 != 1 deliberately tempers the clean posterior; "
                "use d0=1 for the exact clean-time factorization."
            )

        item_index_offsets = None
        item_index_stride = None
        if mini_batch_size < num_items_int:
            if integrate_kwargs.get("method") == "scipy":
                raise ValueError("The scipy solver does not support compositional mini-batching; use all factors.")

            num_rows = ops.shape(conditions)[0]
            item_index_offsets = keras.random.randint(
                shape=(num_rows, 1),
                minval=0,
                maxval=num_items_int,
                dtype="int32",
                seed=seed,
            )

            # A prime greater than J is coprime to J, so modular stepping visits
            # each factor exactly once before repeating.
            item_index_stride = num_items_int + 1
            while any(item_index_stride % divisor == 0 for divisor in range(2, math.isqrt(item_index_stride) + 1)):
                item_index_stride += 1

        def item_indices_for_step(solver_step):
            if item_index_offsets is None:
                return None
            if solver_step is None:
                raise RuntimeError("The selected solver does not expose a step index required for mini-batching.")
            solver_step = ops.cast(solver_step, "int32")
            positions = solver_step * mini_batch_size + ops.arange(mini_batch_size, dtype="int32")
            return ops.mod(item_index_offsets + positions[None, :] * item_index_stride, num_items_int)

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(z))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(z))
            z = maybe_mask_tensor(z, target_mask, replacement=targets_fixed)

        if density:
            raise NotImplementedError(
                "Compositional log densities are not implemented: the previous zero-trace result was incorrect."
            )

        state = {"xz": z}

        if integrate_kwargs["method"] in STOCHASTIC_METHODS:
            stochastic_solver = integrate_kwargs["method"] != "glass"

            def deltas(time, xz, solver_step=None):
                return {
                    "xz": self.compositional_velocity(
                        xz,
                        time=time,
                        stochastic_solver=stochastic_solver,
                        conditions=conditions,
                        compute_prior_score=compute_prior_score,
                        mini_batch_size=mini_batch_size,
                        training=training,
                        seed=seed,
                        item_indices=item_indices_for_step(solver_step),
                        compositional_bridge_d0=compositional_bridge_d0,
                        compositional_bridge_d1=compositional_bridge_d1,
                        **score_kwargs,
                    )
                }

            def diffusion(time, xz):
                return {"xz": self.diffusion_term(xz, time=time, training=training, **kwargs)}

            score_fn = None
            if "corrector_steps" in integrate_kwargs or integrate_kwargs["method"] == "langevin":

                def score_fn(time, xz, solver_step=None):
                    return {
                        "xz": self.compositional_score(
                            xz=xz,
                            time=time,
                            conditions=conditions,
                            compute_prior_score=compute_prior_score,
                            mini_batch_size=mini_batch_size,
                            training=training,
                            seed=seed,
                            item_indices=item_indices_for_step(solver_step),
                            compositional_bridge_d0=compositional_bridge_d0,
                            compositional_bridge_d1=compositional_bridge_d1,
                            **score_kwargs,
                        )
                    }

            state = integrate_stochastic(
                drift_fn=deltas,
                diffusion_fn=diffusion,
                score_fn=score_fn,
                noise_schedule=self.noise_schedule,
                state=state,
                seed=seed,
                **integrate_kwargs,
            )
        else:

            def deltas(time, xz, solver_step=None):
                return {
                    "xz": self.compositional_velocity(
                        xz,
                        time=time,
                        stochastic_solver=False,
                        conditions=conditions,
                        compute_prior_score=compute_prior_score,
                        mini_batch_size=mini_batch_size,
                        training=training,
                        seed=seed,
                        item_indices=item_indices_for_step(solver_step),
                        compositional_bridge_d0=compositional_bridge_d0,
                        compositional_bridge_d1=compositional_bridge_d1,
                        **score_kwargs,
                    )
                }

            state = integrate(deltas, state, **integrate_kwargs)

        x = state["xz"]
        return x
