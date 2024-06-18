from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections.abc import Iterable, Generator, Callable
import flax
from flax import linen as nn
from flax.struct import dataclass, field
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import optax
import pathlib
import pytest

from sklearn.base import BaseEstimator

import bde.ml.training
from bde.utils import configs as cnfg


class BasicModule(nn.Module, ABC):
    """
    An abstract Module class for easy inheritance and API implementation
    """
    n_input_params: Union[int, list[int]]
    n_output_params: Union[int, list[int]]

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Function for performing the calculation of the module.
        """
        ...


class FullyConnectedModule(BasicModule):
    """
    A class for easy initialization of fully connected neural network with flax.
    """
    n_input_params: int
    n_output_params: int
    layer_sizes: Optional[Union[Iterable[int], int]] = None
    do_final_activation: bool = True

    @nn.compact
    def __call__(self, x):
        """
        Function for performing the calculation of the module.
        """
        if self.layer_sizes is not None:
            layer_sizes = self.layer_sizes
            if isinstance(layer_sizes, int):
                layer_sizes = (layer_sizes, )
            for layer_size in layer_sizes:
                x = nn.Dense(features=layer_size)(x)
                x = nn.relu(x)
        x = nn.Dense(features=self.n_output_params)(x)
        x = nn.softmax(x) if self.do_final_activation else x
        return x


class FullyConnectedEstimator(BaseEstimator):
    """
    A class implementing an SKlearn API for the BaseEstimator
    """
    def __init__(self, model: BasicModule, optimizer: optax._src.base.GradientTransformation, loss: Callable,
                 seed: int = cnfg.General.SEED, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.seed = seed

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """
        Fit the function to the given data.
        :param X: The input data.
        :param y: The labels. If y is None, X is assumed to include the labels as well.
        """
        self.params = None
        if y is not None:
            X = zip(X, y)
        for x, y in X:
            if self.params is None:
                self.batch_size = x.shape[0]
                self.params, _ = init_dense_model(self.model, self.batch_size, self.seed)
                self.model_state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                                 params=self.params,
                                                                 tx=self.optimizer)
            bde.ml.training.train_step(self.model_state, x, f_loss=self.loss)

            # model_state, loss_train = pde_net.ml.training.train_step(model_state, (x_train, b_train))
            # loss_test = loss_func(model_state, model_state.params, x_test, b_test)
            # train_log.append(loss_train)
            # test_log.append(loss_test)
            # states_log.append(model_state)
            #
            # if min_test is None or early_stop is None:
            #     min_test = loss_test
            #     continue
            # if loss_test < min_test:
            #     min_test = loss_test
            #     cons_worse = 0
            #     continue
            # cons_worse += 1
            # if cons_worse >= early_stop:
            #     break

    def predict(self, X: ArrayLike) -> Array:
        """
        Applies the fitted model to the input data.
        :param X: The input data
        :return: Predicted labels
        """
        return self.model.apply(self.params, X)


def init_dense_model(
        model: BasicModule,
        batch_size: int = 1,
        seed: int = cnfg.General.SEED,
) -> tuple[dict, Array]:
    """
    Fast initialization for a fully connected dense network
    :param model: A model object
    :param batch_size: The batch size for training
    :param seed: A seed for initialization
    :return: A dict with the params, and the input used for the initialization
    """
    rng = jax.random.key(seed=seed)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    if not isinstance(model.n_input_params, int):
        raise NotImplementedError(f"Only 1 input is currently supported")
    inp = jax.random.normal(inp_rng, (batch_size, model.n_input_params))
    params = model.init(init_rng, inp)
    return params, inp


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_models"
    pytest.main([str(tests_path)])
