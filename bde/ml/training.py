r"""Training Utilities for Bayesian Neural Networks.

This module provides functionality for training Bayesian Neural Networks within the
Bayesian Deep Ensembles (BDE) framework.

Functions
---------
- `train_step`: Executes a single optimization step for the neural network.

"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional, Callable, Dict, Tuple, List, Type
from collections.abc import Iterable, Sequence
import flax
from flax.training import train_state
from flax.training.train_state import TrainState
from flax import linen as nn
from flax.struct import dataclass, field
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import optax
import pathlib
# import bde
import pytest

import bde.ml
from bde.ml.datasets import BasicDataset
from bde.ml.loss import Loss


@jax.jit
def train_step(
        state: TrainState,
        batch: tuple[ArrayLike, ArrayLike],
        f_loss: Loss,
) -> tuple[TrainState, float]:
    r"""Perform an optimization step for the network.

    This function updates the model parameters by performing a single
    optimization step using the provided loss function.

    :param state: The training-state of the network.
    :param batch: Input data-points for the training set, containing 2 items:
        - A set of training data-points.
        - The corresponding labels.
    :param f_loss: The loss function used while training. Should have the following signature:
        (y_true, y_pred)
    :return: Updated state of the network and the loss.
    """
    grad_fn = jax.value_and_grad(bde.ml.loss.flax_training_loss_wrapper_regression(f_loss=f_loss),
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=False  # Function has additional outputs, here accuracy
                                 )
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


# @jax.jit
def jitted_training_over_multiple_models(
        model: nn.Module,
        params,
        optimizer,
        epochs: int,
        f_loss: Loss,
        metrics: List,
        train: BasicDataset,
        valid: BasicDataset,
        history: Array,
):
    return jax.pmap(
        jitted_training,
        axis_name='params',
    )(
        model=model,
        params=params,
        optimizer=optimizer,
        epochs=epochs,
        f_loss=f_loss,
        metrics=metrics,
        train=train,
        valid=valid,
        history=history,
    )


@jax.jit
def jitted_training(
        # model_class: Type[nn.Module],
        # model_kwargs: Dict[str, Any],
        model_state: TrainState,
        # params,
        # optimizer_class: Type[optax._src.base.GradientTransformation],
        # optimizer_kwargs: Dict[str, Any],
        epochs: int,
        f_loss: Loss,
        metrics: List,
        train: BasicDataset,
        valid: BasicDataset,
        history: Array,
):
    # model_state = train_state.TrainState.create(
    #     # apply_fn=model.apply,
    #     apply_fn=model_class(**model_kwargs).apply,
    #     params=params,
    #     tx=optimizer_class(**optimizer_kwargs),
    # )
    model_state, train, valid, history = jax.lax.fori_loop(
        lower=0,
        upper=epochs,
        body_fun=lambda i, val: jitted_training_epoch(
            state_data_history=val,
            num_epoch=i,
            f_loss=f_loss,
            metrics=metrics,
        ),
        init_val=(model_state, train, valid, history),
    )
    return model_state.params, history


@jax.jit
def jitted_training_epoch(
        state_data_history: Tuple[TrainState, BasicDataset, BasicDataset, Array],
        num_epoch: int,
        f_loss: Loss,
        metrics: List,
) -> Tuple[TrainState, BasicDataset, BasicDataset, Array]:
    model_state, train, valid, history = state_data_history

    train = train.shuffle()
    model_state, loss = jax.lax.scan(
        f=lambda cc, sx: train_step(
            cc,
            batch=sx,
            f_loss=f_loss,
        ),
        init=model_state,
        xs=train.get_scannable(),
    )
    history = history.at[0, num_epoch].set(loss.mean())

    # n_metrics = len(metrics)
    # history = jax.lax.fori_loop(
    #     lower=0,
    #     upper=n_metrics,
    #     body_fun=lambda i, val: jitted_evaluation_for_a_metric(
    #         model_state=model_state,
    #         history=val,
    #         metrics=metrics,
    #         idx_metric=i,
    #         idx_history=i + 1,
    #         idx_epoch=num_epoch,
    #         batches=train,
    #     ),
    #     init_val=history,
    # )

    valid = valid.shuffle()
    # ADD:
    #  - Validation step support.
    #  - Early stopping support.
    #  - General callbacks support (stretch goal).
    return model_state, train, valid, history


@jax.jit
def jitted_evaluation_for_a_metric(
        model_state: TrainState,
        batches: BasicDataset,
        metrics,
        history: Array,
        idx_metric: int,
        idx_history: int,
        idx_epoch: int,
):
    metric = metrics[idx_metric]
    m_val, n_batches = 0.0, len(batches)
    m_val = jax.lax.fori_loop(
        lower=0,
        upper=n_batches,
        body_fun=lambda i, val: jitted_evaluation_over_batch(
            model_state=model_state,
            batches=batches,
            f_eval=metric,
            num_batch=i,
            m_val=val,
        ),
        init_val=m_val,
    )
    history = history.at[idx_history, idx_epoch].set(m_val / n_batches)
    return history


@jax.jit
def jitted_evaluation_over_batch(
        model_state: TrainState,
        batches: BasicDataset,
        f_eval,
        num_batch: int,
        m_val: float,
  ) -> float:
    x, y_true = batches[num_batch]
    y_pred = model_state.apply_fn(model_state.params, x)
    return m_val + f_eval(y_true, y_pred)


# @jax.jit
# def validation_epoch(
#         model_state,
#         metric,
#         x,
#         y,
#   ):
#     metric_val = metric(x, y)
#
#     return metric_val


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_training"
    pytest.main([str(tests_path)])
