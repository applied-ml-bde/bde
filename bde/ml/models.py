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
from typing import Any, Union, Optional, List, Tuple, Dict, Type, Sequence
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


@register_pytree_node_class
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

    def tree_flatten(
            self,
    ) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX pytree.

        :return: A tuple with 2 elements:
         - The `children`, containing arrays & pytrees
         - The `aux_data`, containing static and hashable data.
        """
        children = tuple()  # children must contain arrays & pytrees
        aux_data = (
            self.n_output_params,
            self.n_input_params,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    @abstractmethod
    def tree_unflatten(
            cls,
            aux_data: Tuple[Any, Any],
            children: Tuple,
    ) -> "FullyConnectedModule":
        r"""Specify how to build a module from a JAX pytree.

        :param aux_data: Contains static, hashable data.
        :param children: Contain arrays & pytrees.
        :return: Reconstructed Module.
        """
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        r"""Perform the calculation of the module."""
        ...


@register_pytree_node_class
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

    def tree_flatten(
            self,
    ) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX pytree.

        :return: A tuple with 2 elements:
         - The `children`, containing arrays & pytrees (empty).
         - The `aux_data`, containing static and hashable data (4 items).
        """
        children = tuple()  # children must contain arrays & pytrees
        aux_data = (
            self.n_output_params,
            self.n_input_params,
            self.layer_sizes,
            self.do_final_activation,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: Tuple[Any, Any, Any, Any],
            children: Tuple,
    ) -> "FullyConnectedModule":
        r"""Specify how to build a module from a JAX pytree.

        :param aux_data: Contains static, hashable data (4 elements).
        :param children: Contain arrays & pytrees. Not used by this class - Should be empty.
        :return: Reconstructed Module.
        """
        return cls(*aux_data)

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


@register_pytree_node_class
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
            model_kwargs: Optional[Dict[str, Any]] = None,
            optimizer_class: Type[optax._src.base.GradientTransformation] = optax.adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            loss: loss.Loss = loss.LossMSE(),
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

        self.params_ = dict()
        self.history_ = dict()
        self.model_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None

    def tree_flatten(
            self,
    ) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize estimator into a JAX pytree.

        :return: A tuple with 2 elements:
         - The `children`, containing arrays & pytrees (2 elements).
         - The `aux_data`, containing static and hashable data (13 elements).
        """
        children = (
            self.model_,
            self.params_,
        )  # children must contain arrays & pytrees
        aux_data = (
            self.model_class,
            self.model_kwargs,
            self.optimizer_class,
            self.optimizer_kwargs,
            self.loss,
            self.batch_size,
            self.epochs,
            self.metrics,
            self.validation_size,
            self.seed,

            self.is_fitted_,
            self.n_features_in_,
            self.history_,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: Tuple[Tuple, ...],
            children: Tuple[Any, Any],
    ) -> "FullyConnectedEstimator":
        r"""Specify how to build an estimator from a JAX pytree.

        :param aux_data: Contains static, hashable data (2 items).
        :param children: Contain arrays & pytrees (13 items).
        :return: Reconstructed estimator.
        """
        res = cls(
            *aux_data[:10]
        )
        res.model_ = children[0]
        res.params_ = children[1]
        res.is_fitted_ = aux_data[10]
        res.n_features_in_ = aux_data[11]
        res.history_ = aux_data[12]
        return res

    def __sklearn_is_fitted__(self) -> bool:
        r"""Check if the estimator is fitted."""
        return self.is_fitted_

    def _more_tags(self):
        return {
            "_xfail_checks": {
                # Note: By default SKlearn assumes models do not support complex valued data.
                #  If we decide we want to support it, the following line should be uncommented.
                # "check_complex_data": "Complex data is supported.",
                "check_dtype_object": "Numpy input not supported. jax.numpy is required.",
                "check_fit1d": "1D data is not supported.",
                "check_no_attributes_set_in_init": "The model must set some internal attributes like params "
                                                   "in order to to properly turn it into a pytree.",
                "check_n_features_in": "Needs to be set before fitting to allow passing when flattening pytree.",
            },
            # "array_api_support": True,
            "multioutput_only": True,
            "X_types": ["2darray", "2dlabels"]
        }

    def _make_history_container(self) -> Array:
        r"""Create an empty numpy array for recording the training process.

        Returns
        -------
            A zeros array where the 1st axis represents the type of evaluation
            and the 2nd axis represents the epoch number.
        """
        n_hist = 1
        if self.metrics is not None:
            n_hist += len(self.metrics)
        if self.validation_size is not None:
            n_hist = n_hist * 2
        return jnp.zeros((n_hist, self.epochs))

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
        metrics: List = [] if self.metrics is None else self.metrics
        if len(metrics) > 0:
            raise NotImplementedError(f"Metrics are not yet supported.")  # TODO: Remove after implementation
        if self.validation_size is not None:
            raise NotImplementedError(f"Validation is not yet supported.")  # TODO: Remove after implementation

        self.params_ = None
        self.history_ = dict()
        model_kwargs: Dict = {
            "n_output_params": 1,
        } if self.model_kwargs is None else self.model_kwargs
        optimizer_kwargs: Dict = {
            "learning_rate": 1e-3,
        } if self.optimizer_kwargs is None else self.optimizer_kwargs
        self.model_ = self.model_class(**model_kwargs)

        if y.ndim == X.ndim - 1 and self.model_.n_output_params == 1:
            y = y.reshape(-1, 1)
        ds = bde.ml.datasets.DatasetWrapper(x=X, y=y, batch_size=self.batch_size, seed=cnfg.General.SEED)

        self.params_, _ = init_dense_model(
            model=self.model_,
            batch_size=self.batch_size,
            n_features=X.shape[-1],
            seed=self.seed,
        )
        history_container = self._make_history_container()
        name_base = f""  # use f"val_" for validation variations

        if self.epochs > 0:
            self.params_, history_container = training.jitted_training(
                model=self.model_,
                # model_class=self.model_class,
                # model_kwargs=model_kwargs,
                params=self.params_,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=optimizer_kwargs,
                epochs=self.epochs,
                f_loss=self.loss,
                metrics=metrics,
                train=ds,
                valid=ds,
                history=history_container,
            )
        # TODO: Transform `history_container` to `self.history`
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[-1]
        return self

    @jax.jit
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
