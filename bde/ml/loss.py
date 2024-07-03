from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections.abc import Iterable, Generator, Callable
from flax.training.train_state import TrainState
from flax.struct import dataclass
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import pathlib
import pytest

# from bde.utils import configs as cnfg


@dataclass
class LogLikelihoodLoss:
    sigma: float = 1e-6,
    mean_weight: float = 1.0,
    do_reduce: bool = False,

    @jax.jit
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        # TODO: Complete docstring
        """
        :param y_true:
        :param y_pred:
        :return:
        """
        mean_pred, std_pred = self._split_pred(y_true=y_true, y_pred=y_pred)
        res = jnp.log(std_pred)
        weight_factor = jnp.array(self.mean_weight).reshape(-1) / 2
        res += weight_factor * (((y_true - mean_pred) / std_pred) ** 2)
        # ADD: An option for reduction=True:
        return res.mean(axis=tuple(range(1, res.ndim)))
        # return res.mean()

    @jax.jit
    def _split_pred(self, y_true: ArrayLike, y_pred: ArrayLike) -> tuple[Array, Array]:
        """
        Splits the predicted values into 2 arrays of the same shape.
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
        :param y_true: The true labels.
        :param y_pred: The predicted values. The last axis includes the labels, and the uncertainties.
        :return: A tuple containing the predicted labels, and the predicted uncertainty.
        """
        n_mean = y_true.shape[-1]
        mean_pred = y_pred[..., :n_mean]
        std_pred = y_pred[..., n_mean:(2 * n_mean)]
        std_pred = jnp.clip(std_pred, a_min=jnp.array(self.sigma), a_max=None)
        n_std = std_pred.shape[-1]

        padding = [(0, 0) if ax != y_true.ndim - 1 else (0, n_mean - n_std) for ax in range(y_true.ndim)]
        std_pred = jnp.pad(
            std_pred,
            pad_width=padding,
            mode="constant",
            constant_values=1,
        )
        return mean_pred, std_pred


def flax_training_loss_wrapper_regression(
        f_loss: Callable[[ArrayLike, ArrayLike], float],
) -> Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]:
    @jax.jit
    def sub_f(state, params, batch):
        x, y = batch
        preds = state.apply_fn(params, x)
        loss = f_loss(y, preds)
        return jnp.mean(loss)
    return sub_f


def flax_training_loss_wrapper_classification(
        f_loss: Callable[[ArrayLike, ArrayLike], float],
) -> Callable[[TrainState, dict, tuple[ArrayLike, ArrayLike]], float]:
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
