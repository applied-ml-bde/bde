r"""Training Utilities for Bayesian Neural Networks.

This module provides functionality for training Bayesian Neural Networks within the
Bayesian Deep Ensembles (BDE) framework.

Functions
---------
- `train_step`: Executes a single optimization step for the neural network.

"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional, Callable
from collections.abc import Iterable
import flax
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


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_training"
    pytest.main([str(tests_path)])
