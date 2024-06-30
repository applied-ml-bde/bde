from abc import ABC, abstractmethod
from typing import Any, Union, Optional, Callable
from collections.abc import Iterable
import flax
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


@jax.jit
def train_step(state, inp: tuple[ArrayLike, ArrayLike],
               f_loss: Callable[[Any, dict, Array], float]) -> tuple[Any, float]:
    """
    Perform an optimization step for the network
    :param state: The state of the network.
    :param inp: Input points for the training set, containing 2 items:
        - A set of training points.
        - Points that should be near-0 valued for normalization.
    :param f_loss: The loss function used while training. Should have the following signature:
        (state, params, batch: Array, boundary: Array, ...)
    :return: Updated state of the network and the loss of the applied batch/ boundary combination
    """
    # Gradient function
    grad_fn = jax.value_and_grad(f_loss,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=False  # used for functions with multiple outputs, like an extra metric
                                 )
    # Determine gradients for current model, parameters and batch
    loss, grads = grad_fn(state, state.params, inp)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_training"
    pytest.main([str(tests_path)])
