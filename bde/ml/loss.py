r"""Loss Functions for Bayesian Neural Networks.

This module contains implementations of loss functions and their wrappers
used in training Bayesian Neural Networks within the Bayesian Deep Ensembles (BDE)
framework.

Classes
-------
- `Loss`: Defines the API used by loss-related classes.
- `LossMSE`: A callable class for computing MSE loss.
- `NLLLoss`: Base class for Negative Log Likelihood Loss functions.
- `GaussianNLLLoss`: A callable class for computing the Gaussian negative
    log-likelihood loss.

Functions
---------
- `flax_training_loss_wrapper_regression`: Wraps a regression loss function for
    training.
- `flax_training_loss_wrapper_classification`: Wraps a classification loss function
    for training.

"""

import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)

import chex
import jax
import optax
import pytest
from flax.training.train_state import TrainState
from jax import Array
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike


@register_pytree_node_class
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

    @abstractmethod
    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX pytree.

        Returns
        -------
        A tuple with 2 elements:
         - The `children`, containing arrays & pytrees
         - The `aux_data`, containing static and hashable data.
        """
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(
        cls,
        aux_data: Optional[Tuple],
        children: Tuple,
    ) -> "Loss":
        r"""Specify how to build a module from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & pytrees.

        Returns
        -------
        Loss
            Reconstructed loss function.
        """
        ...


@register_pytree_node_class
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

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX pytree.

        Returns
        -------
        A tuple with 2 elements:
         - The `children`, containing arrays & pytrees
         - The `aux_data`, containing static and hashable data.
        """
        return tuple(), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Optional[Tuple],
        children: Tuple,
    ) -> "LossMSE":
        r"""Specify how to build a module from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & pytrees.

        Returns
        -------
        LossMSE
            Reconstructed loss function.
        """
        return LossMSE()


@register_pytree_node_class
class NLLLoss(Loss, ABC):
    r"""Negative log likelihood loss.

    A base class for loss classes representing the negative log likelihood loss from a
    certain probability distribution.

    .. math::
        \ell_{\text{NLL-loss}} = -\log{\mathcal{P}(\text{data} | \text{model})}
    """

    params: dict[str, Any]

    @abstractmethod
    def _split_pred(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ): ...


@register_pytree_node_class
class GaussianNLLLoss(NLLLoss):
    r"""Gaussian negative log likelihood loss.

    A callable jax-supported class for computing the negative log likelihood loss of a
    Gaussian distribution.
    This loss is commonly used in probabilistic models to quantify the difference
    between the predicted probability distribution and the true labels.

    Mathematically, it is defined as:

    .. math::
        \ell_{\text{Gaussian NLLLoss}} = \frac{1}{2}[
            \log{(var)} +
            \frac{(\hat\mu - \mu)^2}{var} +
            \log{(2\pi)}
        ]

    This implementation includes the following parameters:

    .. math::
        \ell_{\text{Gaussian NLLLoss}} = \frac{1}{2}[
            \log{(var)} +
            \omega_{\text{mean weight}} \cdot
                \frac{(\hat\mu - \mu)^2}{var} +
            \begin{cases}
                \log{(2\pi)} && \text{"is full" is True } \\
                0 && \text{"is full" is False }
            \end{cases}
        ]

    where

    .. math::
        var = max(\sigma^2, \epsilon)

    Attributes
    ----------
    params : dict[str, ...]
        Defines loss-related parameters:
        - epsilon : float
            A stability factor for the variance.
        - mean_weight : float
            A scale factor for the mean.
        - is_full : bool
            If true include constant loss value, otherwise ignored.


    Methods
    -------
    __call__(y_true, y_pred)
        Computes the log-likelihood loss for the given predictions and labels.
    _split_pred(y_true, y_pred)
        Splits the predicted values into predictions and their corresponding
        uncertainties.
    apply_reduced()
        Evaluates the reduced loss (inherited from base class).
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        mean_weight: float = 1.0,
        is_full: bool = False,
    ):
        r"""Set parameters for loss function.

        Parameters
        ----------
        epsilon
            A stability factor for the variance.
        mean_weight
            A scale factor for the mean.
        is_full
            If true include constant loss value, otherwise ignored.
        """
        self.params = {
            "epsilon": epsilon,
            "mean_weight": mean_weight,
            "is_full": is_full,
        }

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX pytree.

        Returns
        -------
        A tuple with 2 elements:
         - The `children`, containing arrays & pytrees
         - The `aux_data`, containing static and hashable data.
        """
        children = (self.params,)
        return children, None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Optional[Tuple],
        children: Tuple,
    ) -> "GaussianNLLLoss":
        r"""Specify how to build a module from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & pytrees.

        Returns
        -------
        GaussianNLLLoss
            Reconstructed loss function.
        """
        return GaussianNLLLoss(
            epsilon=children[0]["epsilon"],
            mean_weight=children[0]["mean_weight"],
            is_full=children[0]["is_full"],
        )

    @staticmethod
    @jax.jit
    def _f_true_for_call(x):
        return x + jnp.log(2 * jnp.pi)

    @staticmethod
    @jax.jit
    def _f_false_for_call(x):
        return x

    @jax.jit
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> Array:
        r"""Compute the Gaussian-NLLLoss for the given predictions and labels.

        Parameters
        ----------
        y_true
            The true labels. An array of the shape ``(n_samples, ..., n_features)``.
        y_pred
            The predicted values.
            An array of the shape ``(n_samples, ..., n_features + n_uncertainty)``.
            If ``n_uncertainty < n_features``, the last ``n_features - n_uncertainty``
            will be assigned an uncertainty of 1 which effectively calculates their MSE
            (as long as ``\epsilon\se1``).

        Returns
        -------
        Array
            Returns the log-likelihood of the given values.
        """
        mean_pred, std_pred = self._split_pred(y_true=y_true, y_pred=y_pred)
        var_pred = jnp.clip(
            std_pred**2,
            a_min=self.params["epsilon"],
            a_max=None,
        )
        res = (y_true - mean_pred) ** 2
        res = self.params["mean_weight"] * res / var_pred
        res = 0.5 * jax.lax.cond(
            self.params["is_full"],
            self._f_true_for_call,
            self._f_false_for_call,
            res + jnp.log(var_pred),
        )
        return res.mean(axis=tuple(range(1, res.ndim)))

    @jax.jit
    def _split_pred(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> tuple[Array, Array]:
        r"""Split the predicted values into predictions and uncertainties.

        Split the predicted values into 2 arrays of the same shape: predictions and
        uncertainties.
        The number of expected values is inferred based on the number of features
        (size of last axis) of `y_true`:
            - The 1st array predicts a mean value for each feature in `y_true`.
            - The 2st array predicts an uncertainty value (std) for each feature in
              `y_true`.
              If there are not enough items, they would be substituted with 1:

              .. math::
                  y_true=(\el_1, \el_2): y_pred=(p_1, p_2, p_3) \to
                  mean_pred=(\mu_1, \mu_2), std_pred=(\sigma_1, 1)

              If there are too many items predicted, they would be cut-off.
              # NOTE: Is this the desired behavior?

        Parameters
        ----------
        y_true
            The true labels.
        y_pred
            The predicted values. The last axis includes both the labels and the
            uncertainties.

        Returns
        -------
        tuple[Array, Array]
            A tuple containing the predicted labels and the predicted uncertainty.
        """
        n_mean = y_true.shape[-1]
        chex.assert_scalar_non_negative(y_pred.shape[-1] - n_mean)

        # NOTE: Right now the following behavior can be handled as a feature,
        #  but it might be better to treat it as a bad input.
        # chex.assert_scalar_non_negative((n_mean * 2) - y_pred.shape[-1])

        mean_pred, std_pred = jnp.split(y_pred, [n_mean], axis=-1)
        std_pred = std_pred[..., :n_mean]
        n_std = std_pred.shape[-1]

        padding = [(0, 0)] * (y_true.ndim - 1)
        padding += [(0, n_mean - n_std)]
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
        It should take the true labels and predicted labels as input and return the
        computed loss value.

    Returns
    -------
    Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]
        A function that can be used in the training loop,
        taking the model state, parameters, and a batch of data as input and
        returning the loss.
    """

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
        It should take the true labels and predicted labels as input and return the
        computed loss value.

    Returns
    -------
    Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]
        A function that can be used in the training loop,
        taking the model state, parameters, and a batch of data as input and returning
        the loss.
    """

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
