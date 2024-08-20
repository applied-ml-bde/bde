r"""Models.

This module contains classes and functions for defining and managing various
neural network models used in the Bayesian Deep Ensembles (BDE) framework.
It includes basic building blocks like fully connected layers and estimators
that adhere to the scikit-learn API.

Classes
-------
- `BasicModule`: An abstract base class defining an API for neural network modules.
- `FullyConnectedModule`: A fully connected neural network module.
- `FullyConnectedEstimator`: A scikit-learn-compatible estimator for training models.

Functions
---------
- `init_dense_model`: Utility function for initializing a fully connected dense model.

"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional, List, Tuple, Dict, Type
import chex
from collections.abc import Iterable, Generator, Callable
import flax
from flax import linen as nn
from flax.struct import dataclass, field
from flax.training import train_state
from functools import partial
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.tree_util import register_pytree_node_class
import optax
import pathlib
import pytest

from sklearn.base import BaseEstimator

from bde.ml import loss, training
import bde.utils
from bde.utils import configs as cnfg


class BasicModule(nn.Module, ABC):
    r"""An abstract base class for easy inheritance and API implementation.

    Attributes
    ----------
    n_output_params : Union[int, list[int]]
        The number of output parameters or the shape of the output tensor(s). Similar
        to `n_input_params`, this can be an integer or a list.
    n_input_params : Optional[Union[int, list[int]]]
        The number of input parameters or the shape of the input tensor(s).
        This can be an integer for models with a single-input
        or a list of integers for multi-input models.

    Methods
    -------
    __call__(*args, **kwargs)
        Abstract method to be implemented by subclasses, defining the API of a forward pass of the module.
    """

    n_output_params: Union[int, list[int]]
    n_input_params: Optional[Union[int, list[int]]] = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        r"""Perform the calculation of the module."""
        ...


class FullyConnectedModule(BasicModule):
    r"""A class for easy initialization of fully connected neural networks with flax.

    This class allows for the creation of fully connected neural
    networks with a variable number of layers and neurons per layer.
    This class implements the API defined by `BasicModule`.

    Attributes
    ----------
    n_output_params : int
        The number of output features or neurons in the output layer.
    n_input_params : Optional[int]
        The number of input features or neurons in the input layer.
        If None, the number if determined based on the used params (usually determined by the data used for fitting).
    layer_sizes : Optional[Union[Iterable[int], int]], optional
        The number of neurons in each hidden layer.
        If an integer is provided, a single hidden layer with that many neurons is created.
        If an iterable of integers is provided, multiple hidden layers are created with the specified number of neurons.
        Default is None, which implies no hidden layers (only an input layer and an output layer).
    do_final_activation : bool, optional
        Whether to apply an activation function to the output layer.
        Default is True, meaning the final layer will have an activation function (softmax).

    Methods
    -------
    __call__(x)
        Define the forward pass of the fully connected network.
    """

    n_output_params: int
    n_input_params: Optional[int] = None
    layer_sizes: Optional[Union[Iterable[int], int]] = None
    do_final_activation: bool = True

    @nn.compact
    def __call__(self, x):
        r"""Perform a forward pass of the fully connected network.

        The forward pass processes the input data `x` through a series of fully connected layers,
        with the option to apply an activation function to the final layer.

        :param x: The input data, typically a batch of samples with shape `(batch_size, n_input_params)`.
        :return: The output of the network, with shape `(batch_size, n_output_params)`.
        """
        if self.layer_sizes is not None:
            layer_sizes = self.layer_sizes
            if isinstance(layer_sizes, int):
                layer_sizes = (layer_sizes, )
            for idx, layer_size in enumerate(layer_sizes):
                x = nn.Dense(
                    features=layer_size,
                    name=f"DenseLayer{idx}",
                )(x)
                x = nn.relu(x)
        x = nn.Dense(features=self.n_output_params, name=f"Output")(x)
        x = nn.softmax(x) if self.do_final_activation else x
        return x


class FullyConnectedEstimator(BaseEstimator):
    r"""SKlearn-compatible estimator for training fully connected neural networks with Jax.

    The `FullyConnectedEstimator` class wraps a Flax-based neural network model into an SKlearn-style estimator,
    providing a compatible interface for fitting, predicting, and evaluating models.

    Attributes
    ----------
    # TODO: List

    Methods
    -------
    fit(X, y=None)
        Fit the model to the training data.
    predict(X)
        Predict the output for the given input data using the trained model.
    _more_tags()
        Used by the SKlearn API to set model tags.
    """

    def __init__(
            self,
            model_class: Type[BasicModule] = FullyConnectedModule,
            model_kwargs: Optional[Dict] = None,
            optimizer_class: Type[optax._src.base.GradientTransformation] = optax.adam,
            optimizer_kwargs: Optional[Dict] = None,
            loss: Callable = loss.LossMSE(),
            batch_size: int = 1,
            epochs: int = 1,
            metrics: Optional[list] = None,
            validation_size: Optional[Union[float, tuple[ArrayLike, ArrayLike]]] = None,
            seed: int = cnfg.General.SEED,
            **kwargs,
    ):
        r"""Initialize the estimator architecture and training parameters.

        :param model_class: The neural network model class wrapped by the estimator.
        :param model_kwargs: The kwargs used to init the wrapped model.
        :param optimizer_class: The optimizer class used by the estimator for training.
        :param optimizer_kwargs: The kwargs used to init optimizer.
        :param loss: The loss function used during training.
        :param batch_size: The batch size for training, by default 1.
        :param epochs: Number of epochs for training, by default 1.
        :param metrics: A list of metrics to evaluate during training, by default None.
        :param validation_size: The size of the validation set,
        or a tuple containing validation data. by default None.
        :param seed: Random seed for initialization.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics = metrics
        self.validation_size = validation_size
        self.seed = seed

    def _more_tags(self):
        return {
            "_xfail_checks": {
                # Note: By default SKlearn assumes models do not support complex valued data.
                #  If we decide we want to support it, the following line should be uncommented.
                # "check_complex_data": "Complex data is supported.",
                "check_dtype_object": "Numpy input not supported. jax.numpy is required.",
                "check_fit1d": "1D data is not supported.",
            },
            "array_api_support": True,
            "multioutput_only": True,
            "X_types": ["2darray", "2dlabels"]
        }

    def fit(
            self,
            X: ArrayLike,
            y: Optional[ArrayLike] = None,
    ) -> BaseEstimator:
        r"""Fit the function to the given data.

        :param X: The input data.
        :param y: The labels. If y is None, X is assumed to include the labels as well.
        :return: The fitted estimator.
        """
        if y is None:
            y = X
        bde.utils.utils.check_fit_input(X, y)
        metrics = [] if self.metrics is None else self.metrics
        if len(metrics) > 0:
            raise ValueError(f"Metrics are not yet supported.")  # TODO: Remove after implementation
        if self.validation_size is not None:
            raise ValueError(f"Validation is not yet supported.")  # TODO: Remove after implementation

        self.params_ = None
        self.history_ = dict()
        model_kwargs: Dict = {
            "n_output_params": 1,
        } if self.model_kwargs is None else self.model_kwargs
        optimizer_kwargs: Dict = {
            "learning_rate": 1e-3,
        } if self.optimizer_kwargs is None else self.optimizer_kwargs

        self.model_ = self.model_class(**model_kwargs)
        optimizer = self.optimizer_class(**optimizer_kwargs)

        n_splits = X.shape[0] // self.batch_size
        if y is None:
            y = X
        if y.ndim == X.ndim - 1 and self.model_.n_output_params == 1:
            y = y.reshape(-1, 1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y don't match in number of samples.")
        X, y = X[:(n_splits * self.batch_size)], y[:(n_splits * self.batch_size)]
        # TODO: Implement a function which shuffles, crops, and return a list of zipped batches.
        #  Once it is done, remove the manual split above.
        #  The idea is that if there are some extra items,
        #  they would be used in some of the epochs due to the shuffling.

        self.params_, _ = init_dense_model(
            model=self.model_,
            batch_size=self.batch_size,
            n_features=X.shape[-1],
            seed=self.seed,
        )
        model_state = train_state.TrainState.create(
            apply_fn=self.model_.apply,
            params=self.params_,
            tx=optimizer,
        )
        name_base = f""  # use f"val_" for validation variations
        for epoch in range(self.epochs):
            # ADD: optional shuffling
            for xx, yy in zip(jnp.split(X, n_splits), jnp.split(y, n_splits)):
                model_state, loss = training.train_step(
                    model_state,
                    batch=(xx, yy),
                    f_loss=self.loss,
                )
                self.history_[f"{name_base}loss"] = self.history_.get(f"{name_base}loss", list()) + [loss]

                for metric_name, a_metric in metrics:
                    metric_val = a_metric  # TODO: Calculate metric properly
                    metric_key = f"{name_base}{metric_name}"
                    self.history_[metric_key] = self.history_.get(metric_key, list()) + [metric_val]

                # model_state_, loss_train = pde_net.ml.training.train_step(model_state_, (x_train, b_train))
                # loss_test = loss_func(model_state_, model_state_.params, x_test, b_test)
                # train_log.append(loss_train)
                # test_log.append(loss_test)
                # states_log.append(model_state_)
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
            # ADD: Validation stage
        self.params_ = model_state.params
        self.is_fitted_ = True  # TODO: Check if this is required by the API
        self.n_features_in_ = X.shape[-1]
        return self

    # @chex.chexify
    # @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def predict(self, X: ArrayLike) -> Array:
        r"""Apply the fitted model to the input data.

        :param X: The input data.
        :return: Predicted labels.
        """
        bde.utils.utils.check_predict_input(X)
        chex.assert_equal(self.is_fitted_, True)
        return self.model_.apply(self.params_, X)


def init_dense_model(
        model: BasicModule,
        batch_size: int = 1,
        n_features: Optional[int] = None,
        seed: int = cnfg.General.SEED,
) -> tuple[dict, Array]:
    r"""Fast initialization for a fully connected dense network.

    :param model: A model object.
    :param batch_size: The batch size for training.
    :param n_features: The size of the input layer.
    If it is set to `None`, it is inferred based on the provided model.
    :param seed: A seed for initialization.
    :return: A dict with the params, and the input used for the initialization.
    """
    rng = jax.random.key(seed=seed)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    if n_features is None:
        n_features = model.n_input_params
    if model.n_input_params is None:
        if n_features is None:
            raise ValueError("`n_features` and `model.n_input_params` can't both be `None`.")
        # model.n_input_params = n_features
    elif not isinstance(model.n_input_params, int):
        raise NotImplementedError("Only 1 input is currently supported")
    inp = jax.random.normal(inp_rng, (batch_size, n_features))
    params = model.init(init_rng, inp)
    return params, inp


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_models"
    pytest.main([str(tests_path)])
