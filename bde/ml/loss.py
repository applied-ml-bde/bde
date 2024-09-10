r"""Loss Functions for Bayesian Neural Networks.

This module contains implementations of loss functions and their wrappers
used in training Bayesian Neural Networks within the Bayesian Deep Ensembles (BDE) framework.

Classes
-------
- `Loss`: Defines the API used by loss-related classes.
- `LossMSE`: A callable class for computing MSE loss.
- `LogLikelihoodLoss`: A callable class for computing the log-likelihood loss.

Functions
---------
- `flax_training_loss_wrapper_regression`: Wraps a regression loss function for training.
- `flax_training_loss_wrapper_classification`: Wraps a classification loss function for training.

"""  # noqa: E501

import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import optax
import pytest
from flax.struct import dataclass
from flax.training.train_state import TrainState
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

# from bde.utils import configs as cnfg


@dataclass
class Loss(ABC):
    r"""An abstract class for implementing the API of loss functions."""

    @abstractmethod
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> Array:
        r"""Evaluate the loss.

        Returns an unreduced evaluation of the loss.
        i.e. the loss is calculated separately for each item in the batch.

        Parameters
        ----------
        y_true
            The ground truth.
        y_pred
            The prediction.

        Returns
        -------
        Array
            The unreduced loss value.
        """
        ...

    @jax.jit
    def apply_reduced(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        r"""Evaluate reduced the loss.

        The loss is evaluated separately for each item in the batch,
        and the mean of these values is returned.

        Parameters
        ----------
        y_true
            The ground truth.
        y_pred
            The prediction.
        **kwargs
            Other keywords that may be passed to the unreduced loss function.

        Returns
        -------
        Array
            The reduced loss value.
        """
        return self(y_true=y_true, y_pred=y_pred, **kwargs).mean()


@dataclass
class LossMSE(Loss):
    r"""A class wrapper for MSE loss."""

    @jax.jit
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> Array:
        r"""Evaluate the loss.

        Returns an unreduced MSE loss,
        i.e. the loss is calculated separately for each item in the batch.

        Parameters
        ----------
        y_true
            The ground truth.
        y_pred
            The prediction.

        Returns
        -------
        Array
            The unreduced loss value.
        """
        res = optax.losses.squared_error(y_pred, y_true)
        return res.mean(axis=tuple(range(1, res.ndim)))


@dataclass
class LogLikelihoodLoss(Loss):
    r"""A callable jax-supported class for computing the log-likelihood of the given predictions and labels.

    This class implements the log-likelihood loss,
    which is commonly used in probabilistic models to quantify the difference between the predicted
    probability distribution and the true labels.

    Mathematically, it is defined as:

    .. math::
        \ell_{\text{log-likelihood}} = log(max(\sigma, \epsilon)) +
        \frac{\omega}{2} \cdot (\frac{\hat\mu - \mu}{max(\sigma, \epsilon)})^2

    Attributes
    ----------
    epsilon : float
        A small constant added to prevent division by zero.
    mean_weight : float
        The weight applied to the mean squared error term.

    Methods
    -------
    __call__(y_true, y_pred)
        Computes the log-likelihood loss for the given predictions and labels.
    _split_pred(y_true, y_pred)
        Splits the predicted values into predictions and their corresponding uncertainties.
    apply_reduced()
        Evaluates the reduced loss (inherited from base class).
    """  # noqa: E501

    epsilon: float = 1e-6
    mean_weight: float = 1.0

    @jax.jit
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> Array:
        r"""Compute the log-likelihood of the given predictions and labels.

        # ADD: Mathematical description

        Parameters
        ----------
        y_true
            The true labels. An array of the shape ``(n_samples, ..., n_features)``.
        y_pred
            The predicted values.
            An array of the shape ``(n_samples, ..., n_features + n_uncertainty)``.
            If ``n_uncertainty < n_features``, the last ``n_features - n_uncertainty``
            will be assigned an uncertainty of 1, which effectively calculates their MSE
            (as long as ``\epsilon\se1``).

        Returns
        -------
        Array
            Returns the log-likelihood of the given values.
        """  # noqa: E501
        mean_pred, std_pred = self._split_pred(y_true=y_true, y_pred=y_pred)
        res = jnp.log(std_pred)
        weight_factor = jnp.array(self.mean_weight).reshape(-1) / 2
        res += weight_factor * (((y_true - mean_pred) / std_pred) ** 2)
        return res.mean(axis=tuple(range(1, res.ndim)))

    @jax.jit
    def _split_pred(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> tuple[Array, Array]:
        r"""Split the predicted values into 2 arrays of the same shape: predictions and uncertainties.

        The number of expected values is inferred based on the number of features
        (size of last axis) of the true labels:
            - The 1st array corresponds a prediction to each label.
            - The 2nd array corresponds an uncertainty to each label.
              If there are not enough items, they would be substituted with 1:
              $$y_true=(\\el_1, \\el_2), y_pred=(p_1, p_2, p_3) \\to
              mean_pred=(\\mu_1, \\mu_2), std_pred=(\\sigma_1, 1)
              $$

              If there are too many items predicted, they would be cut-off.
              # NOTE: Is this the desired behavior?

        Parameters
        ----------
        y_true
            The true labels.
        y_pred
            The predicted values. The last axis includes both the labels and the uncertainties.

        Returns
        -------
        tuple[Array, Array]
            A tuple containing the predicted labels and the predicted uncertainty.
        """  # noqa: E501
        # TODO: Make sure that the prediction is not too large or too small.
        n_mean = y_true.shape[-1]
        mean_pred = y_pred[..., :n_mean]
        std_pred = y_pred[..., n_mean : (2 * n_mean)]
        std_pred = jnp.clip(std_pred, a_min=jnp.array(self.epsilon), a_max=None)
        n_std = std_pred.shape[-1]

        padding = [
            (0, 0) if ax != y_true.ndim - 1 else (0, n_mean - n_std)
            for ax in range(y_true.ndim)
        ]
        std_pred = jnp.pad(
            std_pred,
            pad_width=padding,
            mode="constant",
            constant_values=1,
        )
        return mean_pred, std_pred


def flax_training_loss_wrapper_regression(
    f_loss: Callable[[ArrayLike, ArrayLike], float],
) -> Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]:  # noqa: D202
    r"""Wrap a regression loss function for use in Flax training.

    This function wraps a regression loss function so that it can be used in
    the training loop of a Flax model.

    Parameters
    ----------
    f_loss
        The loss function to wrap.
        It should take the true labels and predicted labels as input and return the computed loss value.

    Returns
    -------
    Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]
        A function that can be used in the training loop,
        taking the model state, parameters, and a batch of data as input and returning the loss.
    """  # noqa: E501

    @jax.jit
    def sub_f(state, params, batch):
        x, y = batch
        preds = state.apply_fn(params, x)
        loss = f_loss(y, preds)
        return jnp.mean(loss)

    return sub_f


def flax_training_loss_wrapper_classification(
    f_loss: Callable[[ArrayLike, ArrayLike], float],
) -> Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]:  # noqa: D202
    r"""Wrap a classification loss function for use in Flax training.

    This function wraps a classification loss function so that it can be used in
    the training loop of a Flax model.

    Parameters
    ----------
    f_loss
        The loss function to wrap.
        It should take the true labels and predicted labels as input and return the computed loss value.

    Returns
    -------
    Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]
        A function that can be used in the training loop,
        taking the model state, parameters, and a batch of data as input and returning the loss.
    """  # noqa: E501

    @jax.jit
    def sub_f(state, params, batch):
        x, y = batch
        preds = state.apply_fn(params, x)
        preds = (preds > 0).astype(jnp.float32)
        loss = f_loss(y, preds)
        return jnp.mean(loss)

    return sub_f


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_loss"
    pytest.main([str(tests_path)])
