r"""Training Utilities for Bayesian Neural Networks.

This module provides functionality for training Bayesian Neural Networks within the
Bayesian Deep Ensembles (BDE) framework.

Functions
---------
- `train_step`: Executes a single optimization step for the neural network.

"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional, Callable, Dict, Tuple, List
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

import bde.ml.loss


@jax.jit
def train_step(
        state: TrainState,
        batch: tuple[ArrayLike, ArrayLike],
        f_loss: Callable[[ArrayLike, ArrayLike], float],
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


@jax.jit
def jitted_training_over_multiple_models(
        model: nn.Module,
        params,
        optimizer,
        epochs,
        f_loss,
        metrics,
        train,
        valid,
        history,
        **kwargs,
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
        **kwargs,
    )


@jax.jit
def jitted_training(
        model: nn.Module,
        params,
        optimizer,
        epochs,
        f_loss,
        metrics,
        train,
        valid,
        history,
        **kwargs,
):
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
    model_state, history = jax.lax.fori_loop(
        lower=0,
        upper=epochs,
        body_fun=lambda i, val: jitted_training_epoch(
            state_history=val,
            num_epoch=i,
            f_loss=f_loss,
            metrics=metrics,
            train=train,
            valid=valid,
        ),
        init_val=(model_state, history),
    )
    return model_state.params, history


@jax.jit
def jitted_training_epoch(
        state_history,
        num_epoch,
        f_loss,
        metrics,
        train,
        valid,
        **kwargs,
) -> Tuple[TrainState, Any]:
    n_batches = len(train)
    model_state, history = state_history
    loss = 0

    # ADD: Reshuffling of training data.
    model_state, loss = jax.lax.fori_loop(
        lower=0,
        upper=n_batches,
        body_fun=lambda i, val: jitted_training_over_batch(
            model_state_loss=val,
            f_loss=f_loss,
            batches=train,
            num_batch=i,
        ),
        init_val=(model_state, loss),
    )
    history[0].append(loss / n_batches)

    # ADD:
    #  - Metrics support (Iterate over each metric in the list. For each metric, repeat the above process).
    #  - Early stopping support.
    #  - General callbacks support (stretch goal).
    return model_state, history


@jax.jit
def jitted_training_over_batch(
        model_state_loss: Tuple[TrainState, float],
        f_loss,
        batches: Sequence[Tuple[ArrayLike, ArrayLike]],
        num_batch: int,
  ) -> Tuple[TrainState, float]:
    r"""Perform a training step over a single batch.

    :param model_state_loss: A tuple containing:
     - Model + params.
     - The cumulative loss over the batch.
    :param f_loss: Loss functions.
    :param batches: The training dataset, where each entry is a batch of the form (x, y).
    :param num_batch: The index of the current batch in the dataset.
    :return: A tuples with:
     - The updated model + params.
     - The updated cumulative loss.
    """
    model_state, cum_loss = model_state_loss
    batch = batches[num_batch]

    model_state, loss = train_step(
        model_state,
        batch=batch,
        f_loss=f_loss,
    )
    # params = model_state.params
    return model_state, cum_loss + loss


# @jax.jit
# def validation_epoch(
#         model_state,
#         metric,
#         x,
#         y,
#         **kwargs,
#   ):
#     metric_val = metric(x, y)
#
#     return metric_val


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_training"
    pytest.main([str(tests_path)])
