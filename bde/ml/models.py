r"""Models.

This module contains classes and functions for defining and managing various
neural network models used in the Bayesian Deep Ensembles (BDE) framework.
It includes basic building blocks like fully connected layers and estimators
that adhere to the scikit-learn API.

Classes
-------
- `BasicModule`: An abstract base class defining an API for neural network modules.
- `FullyConnectedModule`: A fully connected neural network module.
- `FullyConnectedEstimator`: An SKlearn-compatible estimator for training models.
- `BDEEstimator`: An SKlearn-compatible implementation of Bayesian Deep Ensembles (BDEs).

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
from flax.core import FrozenDict
from functools import partial
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.tree_util import register_pytree_node_class
from jax._src.prng import PRNGKeyArray
import optax
import pathlib
import pytest

from sklearn.base import BaseEstimator

from bde.ml import loss, training, datasets
import bde.utils
from bde.utils import configs as cnfg

import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


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

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX pytree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
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

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & pytrees.

        Returns
        -------
        FullyConnectedModule
            Reconstructed Module.
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

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX pytree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
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

        Parameters
        ----------
        aux_data
            Contains static, hashable data (4 elements).
        children
            Contain arrays & pytrees. Not used by this class - Should be empty.

        Returns
        -------
        FullyConnectedModule
            Reconstructed Module.
        """
        return cls(*aux_data)

    @nn.compact
    def __call__(
            self,
            x,
    ) -> Array:
        r"""Perform a forward pass of the fully connected network.

        The forward pass processes the input data `x` through a series of fully connected layers,
        with the option to apply an activation function to the final layer.

        Parameters
        ----------
        x
            The input data, typically a batch of samples with shape `(batch_size, n_input_params)`.

        Returns
        -------
        Array
            The output of the network, with shape `(batch_size, n_output_params)`.
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
            validation_size: Optional[Union[float, Tuple[ArrayLike, ArrayLike], datasets.BasicDataset]] = None,
            seed: int = cnfg.General.SEED,
            **kwargs,
    ):
        r"""Initialize the estimator architecture and training parameters.

        Parameters
        ----------
        model_class
            The neural network model class wrapped by the estimator.
        model_kwargs
            The kwargs used to init the wrapped model.
        optimizer_class
            The optimizer class used by the estimator for training.
        optimizer_kwargs
            The kwargs used to init optimizer.
        loss
            The loss function used during training.
        batch_size
            The batch size for training, by default 1.
        epochs
            Number of epochs for training, by default 1.
        metrics
            A list of metrics to evaluate during training, by default None.
        validation_size
            The size of the validation set,
            or a tuple containing validation data. by default None.
        seed
            Random seed for initialization.
        **kwargs
            Additional keyword arguments.
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

        self.params_: Union[FrozenDict, Dict] = dict()
        self.history_: Optional[Array] = None
        self.model_: Optional[BasicModule] = None
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize estimator into a JAX pytree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
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

        Parameters
        ----------
        aux_data
            Contains static, hashable data (2 items).
        children
            Contain arrays & pytrees (13 items).

        Returns
        -------
        FullyConnectedEstimator
            Reconstructed estimator.
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
        r"""Define tags for SKlearn."""
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
            if self.validation_size > 0:
                n_hist = n_hist * 2
        return jnp.zeros((n_hist, self.epochs))

    def _prep_fit_params(
            self,
            x: ArrayLike,
            y: Optional[ArrayLike] = None,
            rng_key: Union[PRNGKeyArray, int] = cnfg.General.SEED,
    ) -> Tuple[List, optax._src.base.GradientTransformation, datasets.BasicDataset, datasets.BasicDataset]:
        r"""Handle model and parameter initialization before fitting."""
        if y is None:
            y = x
        bde.utils.utils.check_fit_input(x, y)
        metrics: List = [] if self.metrics is None else self.metrics
        if len(metrics) > 0:
            raise NotImplementedError(f"Metrics are not yet supported.")  # TODO: Remove after implementation
        if self.validation_size is not None:
            raise NotImplementedError(f"Validation is not yet supported.")  # TODO: Remove after implementation

        if not isinstance(rng_key, PRNGKeyArray):
            rng_key = jax.random.key(seed=rng_key)
        self.params_ = None
        model_kwargs: Dict = {
            "n_output_params": 1,
        } if self.model_kwargs is None else self.model_kwargs
        optimizer_kwargs: Dict = {
            "learning_rate": 1e-3,
        } if self.optimizer_kwargs is None else self.optimizer_kwargs
        self.model_ = self.model_class(**model_kwargs)
        optimizer = self.optimizer_class(**optimizer_kwargs)

        if y.ndim == x.ndim - 1 and self.model_.n_output_params == 1:
            y = y.reshape(-1, 1)  # TODO: Good only for 2D data. Fix

        rng_key, train_key, valid_key = rng_key, rng_key, rng_key
        val_size = 0 if self.validation_size is None else self.validation_size
        train_key = int(jax.random.randint(train_key, (), 0, jnp.iinfo(jnp.int32).max))
        valid_key = int(jax.random.randint(valid_key, (), 0, jnp.iinfo(jnp.int32).max))

        if val_size > 0:
            ...  # TODO: Split the data between train and valid
        train = bde.ml.datasets.DatasetWrapper(x=x, y=y, batch_size=self.batch_size, seed=train_key)
        valid = train
        if val_size <= 0:
            valid = train.gen_empty()
            valid.seed = valid_key
        return metrics, optimizer, train, valid

    # @jax.jit
    def init_inner_params(
            self,
            n_features,
            optimizer,
            rng_key,
    ) -> train_state.TrainState:
        r"""Create trainable model state.

        Parameters
        ----------
        n_features
            Number of input features.
        optimizer
            Optimization algorithm used for training.
        rng_key
            Randomness key.

        Returns
        -------
        train_state.TrainState
            Initialized training state.
        """
        params, _ = init_dense_model_jitted(
            model=self.model_,
            rng_key=rng_key,
            batch_size=self.batch_size,
            n_features=n_features,
        )
        model_state = train_state.TrainState.create(
            # apply_fn=model.apply,
            apply_fn=self.model_.apply,
            params=params,
            tx=optimizer,
        )
        return model_state

    def fit(
            self,
            X: ArrayLike,
            y: Optional[ArrayLike] = None,
    ) -> "FullyConnectedEstimator":
        r"""Fit the function to the given data.

        Parameters
        ----------
        X
            The input data.
        y
            The labels.
            If y is None, X is assumed to include the labels as well.

        Returns
        -------
        FullyConnectedEstimator
            The fitted estimator.
        """
        rng = jax.random.key(seed=self.seed)
        rng_init, prep_key, split_key = jax.random.split(rng, 3)
        metrics, optimizer, train, valid = self._prep_fit_params(x=X, y=y, rng_key=prep_key)
        model_state = self.init_inner_params(
            n_features=X.shape[-1],
            optimizer=optimizer,
            rng_key=rng_init,
        )
        if self.epochs > 0:
            self.params_, self.history_ = training.jitted_training(
                model_state=model_state,
                # model_class=self.model_class,
                # model_kwargs=model_kwargs,
                # params=self.params_,
                # optimizer_class=self.optimizer_class,
                # optimizer_kwargs=optimizer_kwargs,
                epochs=jnp.arange(self.epochs),
                f_loss=self.loss,
                metrics=jnp.array(metrics),
                train=train,
                valid=valid,
            )
        else:
            self.params_ = model_state.params
        # TODO: Transform `history_container` to `self.history`
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[-1]
        return self

    def history_description(self) -> Dict[str, Array]:
        r"""Make a readable version of the training history.

        Returns
        -------
        Dict
            Each key corresponds to an evaluation metric/ loss
            and each value is an array describing the values for different epochs.

        Raises
        ------
        AssertionError
            If the model is not fitted.
        """
        chex.assert_equal(self.is_fitted_, True)
        res: Dict[str, int] = {
            "loss": 0,
        }
        if self.validation_size:
            if self.validation_size > 0:
                res.update({f"val_{k}": v + len(res) for k, v in res})
        res: Dict[str, Array] = {k: self.history_[idx] for k, idx in res}
        return res

    @jax.jit
    def predict(
            self,
            X: ArrayLike,
    ) -> Array:
        r"""Apply the fitted model to the input data.

        Parameters
        ----------
        X
            The input data.

        Returns
        -------
        Array
            Predicted labels.
        """
        bde.utils.utils.check_predict_input(X)
        chex.assert_equal(self.is_fitted_, True)
        return self.model_.apply(self.params_, X)


class BDEEstimator(FullyConnectedEstimator):
    r"""SKlearn-compatible implementation of a BDE estimator.

    # TODO: Describe BDE estimator.

    Attributes
    ----------
    # TODO: List

    Methods
    -------
    fit(X, y=None)
        Fit the model to the training data.
    predict(X)
        Predict the output for the given input data using the trained model.
    """

    def __init__(
            self,
            model_class: Type[BasicModule] = FullyConnectedModule,
            model_kwargs: Optional[Dict[str, Any]] = None,
            n_chains: int = 1,
            n_init_runs: int = 1,
            chain_len: int = 1,
            warmup: int = 1,
            optimizer_class: Type[optax._src.base.GradientTransformation] = optax.adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            loss: loss.Loss = loss.LossMSE(),
            batch_size: int = 1,
            epochs: int = 1,
            metrics: Optional[list] = None,
            validation_size: Optional[Union[float, Tuple[ArrayLike, ArrayLike], datasets.BasicDataset]] = None,
            seed: int = cnfg.General.SEED,
            **kwargs,
    ):
        r"""Initialize the estimator architecture and training parameters.

        Parameters
        ----------
        model_class
            The neural network model class wrapped by the estimator.
        model_kwargs
            The kwargs used to init the wrapped model.
        n_chains
            Number chains used for sampling.
            This can't be greater than the number of computational devices.
        optimizer_class
            The optimizer class used by the estimator for training.
        optimizer_kwargs
            The kwargs used to init optimizer.
        loss
            The loss function used during training.
        batch_size
            The batch size for training, by default 1.
        epochs
            Number of epochs for training, by default 1.
        metrics
            A list of metrics to evaluate during training, by default None.
        validation_size
            The size of the validation set,
            or a tuple containing validation data. by default None.
        seed
            Random seed for initialization.
        **kwargs
            Additional keyword arguments.
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

        self.n_chains = n_chains
        self.n_init_runs = n_init_runs
        self.chain_len = chain_len
        self.warmup = warmup

        self.params_: List[Union[FrozenDict, Dict]] = [dict()]
        self.history_: Optional[Array] = None
        self.model_: Optional[BasicModule] = None
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize estimator into a JAX pytree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
             - The `children`, containing arrays & pytrees (2 elements).
             - The `aux_data`, containing static and hashable data (17 elements).
        """
        children = (
            self.model_,
            self.params_,
        )  # children must contain arrays & pytrees
        aux_data = (
            self.model_class,
            self.model_kwargs,
            self.n_chains,
            self.n_init_runs,
            self.chain_len,
            self.warmup,
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
    ) -> "BDEEstimator":
        r"""Specify how to build an estimator from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data (2 items).
        children
            Contain arrays & pytrees (17 items).

        Returns
        -------
        BDEEstimator
            Reconstructed estimator.
        """
        res = cls(
            *aux_data[:14]
        )
        res.model_ = children[0]
        res.params_ = children[1]
        res.is_fitted_ = aux_data[10]
        res.n_features_in_ = aux_data[11]
        res.history_ = aux_data[12]
        return res

    def _more_tags(self):
        r"""Define tags for SKlearn."""
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
                "check_estimators_dtypes": "This test freezes on Github CI/CD but passes locally."
                                           "This is despite implementing timeout for the test."
                                           "It will be temporarily canceled and examined later.",  # TODO: Examine.
            },
            # "array_api_support": True,
            "multioutput_only": True,
            "X_types": ["2darray", "2dlabels"]
        }

    @jax.jit
    def _prior_fitting(
            self,
            model_states,
            train,
            valid,
            metrics,
    ) -> Tuple[Any, Array]:
        r"""Perform non-Bayesian parallelized training to initialize parameters before sampling."""
        params, history = jax.pmap(
            fun=training.jitted_training,
            in_axes=(0, None, None, None, None, None),
            static_broadcasted_argnums=[2],
        )(
            model_states,
            jnp.arange(self.n_init_runs),
            self.loss,
            jnp.array(metrics),
            train,
            valid,
        )
        return params, history

    def fit(
            self,
            X: ArrayLike,
            y: Optional[ArrayLike] = None,
    ) -> "BDEEstimator":
        r"""Fit the function to the given data.

        Parameters
        ----------
        X
            The input data.
        y
            The labels.
            If y is None, X is assumed to include the labels as well.

        Returns
        -------
        BDEEstimator
            The fitted estimator.
        """
        split_key = jax.random.key(seed=self.seed)
        rng = jax.random.split(split_key, self.n_chains + 2)
        init_key_list, prep_key, split_key = rng[:self.n_chains], rng[-2], rng[-1]
        metrics, optimizer, train, valid = self._prep_fit_params(
            x=X,
            y=y,
            rng_key=prep_key,
        )
        model_states = jax.pmap(
            fun=self.init_inner_params,
            in_axes=(None, None, 0),
            static_broadcasted_argnums=[0, 1],
        )(X.shape[-1], optimizer, init_key_list)
        self.params_, self.history_ = self._prior_fitting(
            model_states,
            train,
            valid,
            metrics,
        )
        # TODO: Implement MCMC sampling
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[-1]
        return self

    @jax.jit
    def predict(
            self,
            X: ArrayLike,
    ) -> Array:
        r"""Apply the fitted model to the input data.

        Parameters
        ----------
        X
            The input data.

        Returns
        -------
        Array
            Predicted labels.
        """
        bde.utils.utils.check_predict_input(X)
        chex.assert_equal(self.is_fitted_, True)
        return jax.pmap(
            fun=self.model_.apply,
            in_axes=(0, None),
        )(
            self.params_,
            X,
        ).mean(axis=0)  # NOTE: Temporary implementation until proper BDE sampling and inference are implemented.


def init_dense_model(
        model: BasicModule,
        batch_size: int = 1,
        n_features: Optional[int] = None,
        seed: Union[PRNGKeyArray, int] = cnfg.General.SEED,
) -> Tuple[dict, Array]:
    r"""Fast initialization for a fully connected dense network.

    Parameters
    ----------
    model
        A model object.
    batch_size
        The batch size for training.
    n_features
        The size of the input layer.
        If it is set to `None`, it is inferred based on the provided model.
    seed
        A seed or a PRNGKey for initialization.

    Returns
    -------
    Tuple[dict, Array]
        A tuple with:
         - A parameters dict,
         - The input used for the initialization.
    """
    if not isinstance(seed, PRNGKeyArray):
        seed = jax.random.key(seed=seed)
    inp_rng, init_rng = jax.random.split(seed, 2)
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


@partial(jax.jit, static_argnums=[2, 3])
def init_dense_model_jitted(
        model: BasicModule,
        rng_key: PRNGKeyArray,
        batch_size: int = 1,
        n_features: int = 1,
) -> Tuple[dict, Array]:
    r"""Fast initialization for a fully connected dense network.

    A jitted version of `init_dense_model()`.

    Parameters
    ----------
    model
        A model object.
    rng_key
        A PRNGKey used for randomness in initialization.
    batch_size
        The batch size for training.
    n_features
        The size of the input layer.
        If it is set to `None`, it is inferred based on the provided model.

    Returns
    -------
    Tuple[dict, Array]
        A tuple with:
         - A parameters dict,
         - The input used for the initialization.
    """
    inp_rng, init_rng = jax.random.split(rng_key, 2)
    # TODO: Consider using chex.
    inp = jax.random.normal(inp_rng, (batch_size, n_features))
    params = model.init(init_rng, inp)
    return params, inp


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_models"
    pytest.main([str(tests_path)])
