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
- `init_dense_model_jitted`: ...

"""  # noqa: E501

import multiprocessing
import operator
import os
import pathlib
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import blackjax
import chex
import jax
import optax
import pytest
from blackjax.base import SamplingAlgorithm
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
from jax import Array
from jax import numpy as jnp
from jax._src.prng import PRNGKeyArray
from jax.scipy import stats
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from sklearn.base import BaseEstimator

import bde.utils
from bde.ml import (
    datasets,
    loss,
    training,
)
from bde.utils import configs as cnfg
from bde.utils.utils import (
    cond_with_const_f,
    eval_parallelization_params,
    post_pmap,
    pv_map,
)

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


@register_pytree_node_class
class BasicModule(nn.Module, ABC):
    r"""An abstract base class for easy inheritance and API implementation.

    Attributes
    ----------
    n_output_params : Union[int, list[int]]
        The number of output parameters or the shape of the output tensor(s).
        Similar to `n_input_params`, this can be an integer or a list.
    n_input_params : Optional[Union[int, list[int]]]
        The number of input parameters or the shape of the input tensor(s).
        This can be an integer for models with a single-input
        or a list of integers for multi-input models.

    Methods
    -------
    __call__(*args, **kwargs)
        Abstract method to be implemented by subclasses, defining the API of a
        forward pass of the module.
    tree_flatten()
        Serialize module into a JAX PyTree.
    tree_unflatten(aux_data, children)
        Build module from a serialized PyTree
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
        children: Tuple = tuple()  # children must contain arrays & pytrees
        aux_data: Tuple = (
            self.n_output_params,
            self.n_input_params,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    @abstractmethod
    def tree_unflatten(
        cls,
        aux_data: Optional[Tuple],
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
        If None, the number if determined based on the used params (usually determined
        by the data used for fitting).
    layer_sizes : Optional[Union[Iterable[int], int]], optional
        The number of neurons in each hidden layer.
        If an integer is provided, a single hidden layer with that many neurons is
        created.
        If an iterable of integers is provided, multiple hidden layers are created
        with the specified number of neurons.
        Default is None, which implies no hidden layers (only an input layer and an
        output layer).
    do_final_activation : bool, optional
        Whether to apply an activation function to the output layer.
        Default is True, meaning the final layer will have an activation function
        (softmax).

    Methods
    -------
    __call__(x)
        Define the forward pass of the fully connected network.
    tree_flatten()
        Serialize module into a JAX PyTree.
    tree_unflatten(aux_data, children)
        Build module from a serialized PyTree.
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
        children: Tuple = tuple()  # children must contain arrays & pytrees
        aux_data: Tuple = (
            self.n_output_params,
            self.n_input_params,
            self.layer_sizes,
            self.do_final_activation,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[Any, Any, Any, Any],  # type: ignore
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

        The forward pass processes the input data `x` through a series of fully
        connected layers, with the option to apply an activation function to the
        final layer.

        Parameters
        ----------
        x
            The input data, typically a batch of samples with shape
            `(batch_size, n_input_params)`.

        Returns
        -------
        Array
            The output of the network, with shape `(batch_size, n_output_params)`.
        """
        if self.layer_sizes is not None:
            layer_sizes = self.layer_sizes
            if isinstance(layer_sizes, int):
                layer_sizes = (layer_sizes,)
            for idx, layer_size in enumerate(layer_sizes):
                x = nn.Dense(
                    features=layer_size,
                    name=f"DenseLayer{idx}",
                )(x)
                x = nn.relu(x)
        x = nn.Dense(features=self.n_output_params, name="Output")(x)
        x = nn.softmax(x) if self.do_final_activation else x
        return x


@register_pytree_node_class
class FullyConnectedEstimator(BaseEstimator):
    r"""SKlearn-compatible estimator for training fully connected NN with Jax.

    The `FullyConnectedEstimator` class wraps a Flax-based neural network model into
    an SKlearn-style estimator, providing a compatible interface for fitting,
    predicting, and evaluating models.

    Attributes
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
        A class representing the loss function.
    batch_size
        Number of samples per batch (size of first dimension).
    epochs
        Number of epochs for training.
    metrics
        A list of metrics to evaluate during training, by default None.
    validation_size
        The size of the validation set,
        or a tuple containing validation data. by default None.
    seed
        Random seed for initialization.
    params_
        Parameters of fitted model.
    history_
        Loss and metric records during training.
    model_
        The initialized model class.
    is_fitted_
        A flag indicating weather the model has been fitted or not.
    n_features_in_
        Number of input features (the size of the last input dimension).

    Methods
    -------
    fit(X, y=None)
        Fit the model to the training data.
    predict(X)
        Predict the output for the given input data using the trained model.
    save(path)
        Save estimator to file.
    load(path)
        Load estimator from file.
    history_description()
        Make a readable version of the training history.
    _more_tags()
        Used by the SKlearn API to set model tags.
    tree_flatten()
        Serialize module into a JAX PyTree.
    tree_unflatten(aux_data, children)
        Build module from a serialized PyTree.
    __sklearn_is_fitted__()
        Check if the estimator is fitted.
    init_inner_params(n_features, optimizer, rng_key)
        Create trainable model state.
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
        validation_size: Optional[Union[float]] = None,
        # ADD above support to `Tuple[ArrayLike, ArrayLike], datasets.BasicDataset`
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
        self.history_: Array = None  # type: ignore
        self.model_: BasicModule = None  # type: ignore
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None

    def tree_flatten(self) -> Tuple[Sequence[Union[ArrayLike, Dict]], Any]:
        r"""Specify how to serialize estimator into a JAX pytree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
             - The `children`, containing arrays & pytrees.
             - The `aux_data`, containing static and hashable data.
        """
        atts = {
            "batch_size": self.batch_size,
            "validation_size": self.validation_size,
            "seed": self.seed,
            "is_fitted_": self.is_fitted_,
            "n_features_in_": self.n_features_in_,
        }
        children = (
            self.model_,
            self.params_,
            self.history_,
            self.loss,
            atts,
        )  # children must contain arrays & pytrees
        aux_data = (
            self.model_class,
            self.model_kwargs,
            self.optimizer_class,
            self.optimizer_kwargs,
            self.metrics,
            self.epochs,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[Any, ...],
        children: Tuple[Union[ArrayLike, Dict], ...],
    ) -> "FullyConnectedEstimator":
        r"""Specify how to build an estimator from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & pytrees.

        Returns
        -------
        FullyConnectedEstimator
            Reconstructed estimator.
        """
        atts = children[-1]
        res = cls(
            model_class=aux_data[0],
            model_kwargs=aux_data[1],
            optimizer_class=aux_data[2],
            optimizer_kwargs=aux_data[3],
            loss=children[3],  # type: ignore
            batch_size=atts["batch_size"],
            epochs=aux_data[5],
            metrics=aux_data[4],
            validation_size=atts["validation_size"],
            seed=atts["seed"],
        )
        res.model_ = children[0]  # type: ignore
        res.params_ = children[1]
        res.is_fitted_ = atts["is_fitted_"]
        res.n_features_in_ = atts["n_features_in_"]
        res.history_ = children[2]
        return res

    def save(self, path: Union[str, pathlib.Path]) -> None:
        r"""Save estimator to file."""
        # TODO: Use a more secure format
        children, aux_data = self.tree_flatten()
        with open(path, "wb") as file:
            pickle.dump((aux_data, children), file)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path],
    ) -> "FullyConnectedEstimator":
        r"""Load estimator from file."""
        # TODO: Use a more secure format
        with open(path, "rb") as file:
            aux_data, children = pickle.load(file)
        return cls.tree_unflatten(aux_data, children)

    def __sklearn_is_fitted__(self) -> bool:
        r"""Check if the estimator is fitted."""
        return self.is_fitted_

    def _more_tags(self):
        r"""Define tags for SKlearn."""
        return {
            "_xfail_checks": {
                # Note: By default SKlearn assumes models
                #  do not support complex valued data.
                #  If we decide we want to support it,
                #  the following line should be uncommented.
                # "check_complex_data": "Complex data is supported.",
                "check_dtype_object": (
                    "Numpy input not supported. `jax.numpy` is required."
                ),
                "check_fit1d": "1D data is not supported.",
                "check_no_attributes_set_in_init": (
                    "The model must set some internal attributes like params "
                    "in order to to properly turn it into a pytree."
                ),
                "check_n_features_in": (
                    "Needs to be set before fitting to "
                    "allow passing when flattening pytree."
                ),
            },
            # "array_api_support": True,
            "multioutput_only": True,
            "X_types": ["2darray", "2dlabels"],
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
        seed: int = cnfg.General.SEED,
    ) -> Tuple[
        List,
        optax._src.base.GradientTransformation,
        datasets.BasicDataset,
        datasets.BasicDataset,
    ]:
        r"""Handle model and parameter initialization before fitting."""
        if y is None:
            y = x
        bde.utils.utils.check_fit_input(x, y)
        metrics: List = [] if self.metrics is None else self.metrics
        if len(metrics) > 0:
            raise NotImplementedError(
                "Metrics are not yet supported."
            )  # TODO: Remove after implementation
        if self.validation_size is not None:
            raise NotImplementedError(
                "Validation is not yet supported."
            )  # TODO: Remove after implementation

        rng_key = jax.random.key(seed=seed)
        self.params_ = None
        model_kwargs: Dict = (
            {
                "n_output_params": 1,
            }
            if self.model_kwargs is None
            else self.model_kwargs
        )
        optimizer_kwargs: Dict = (
            {
                "learning_rate": 1e-3,
            }
            if self.optimizer_kwargs is None
            else self.optimizer_kwargs
        )
        self.model_ = self.model_class(**model_kwargs)
        optimizer = self.optimizer_class(**optimizer_kwargs)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        rng_key, train_key, valid_key = rng_key, rng_key, rng_key
        val_size = 0 if self.validation_size is None else self.validation_size
        train_key = int(jax.random.randint(train_key, (), 0, jnp.iinfo(jnp.int32).max))
        valid_key = int(jax.random.randint(valid_key, (), 0, jnp.iinfo(jnp.int32).max))

        if val_size > 0:
            ...  # TODO: Split the data between train and valid
        train = bde.ml.datasets.DatasetWrapper(
            x=x, y=y, batch_size=self.batch_size, seed=train_key
        )
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
        rng_key = jnp.array(rng_key).reshape()
        params, _ = init_dense_model_jitted(
            model=self.model_,
            rng_key=rng_key,
            batch_size=self.batch_size,
            n_features=n_features,
        )
        model_state = train_state.TrainState.create(
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
        metrics, optimizer, train, valid = self._prep_fit_params(
            x=X,
            y=y,
            seed=int(jax.random.randint(prep_key, (), 0, jnp.iinfo(jnp.int32).max)),
        )
        model_state = self.init_inner_params(
            n_features=X.shape[-1],
            optimizer=optimizer,
            rng_key=jnp.array([rng_init]),
        )
        if self.epochs > 0:
            model_state, self.history_ = training.jitted_training(
                model_state=model_state,
                epochs=jnp.arange(self.epochs),
                f_loss=self.loss,
                metrics=jnp.array(metrics),
                train=train,
                valid=valid,
            )
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
                res.update({f"val_{k}": v + len(res) for k, v in res.items()})
        res: Dict[str, Array] = {k: self.history_[idx] for k, idx in res.items()}
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
        bde.utils.utils.check_predict_input(X, self.is_fitted_)
        return self.model_.apply(self.params_, X)


@register_pytree_node_class
class BDEEstimator(FullyConnectedEstimator):
    r"""SKlearn-compatible implementation of a BDE estimator.

    The estimator attempts to sample the parameter distribution of probabilistic
    machine learning models to estimate the posterior predictive distribution,
    allowing to predict uncertainty and confidence values alongside predicted values.

    Attributes
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
        A class representing the loss function.
    batch_size
        Number of samples per batch (size of first dimension).
    epochs
        Number of epochs for the DE-Initialization stage (per chain).
    metrics
        A list of metrics to evaluate during training, by default None.
    validation_size
        The size of the validation set,
        or a tuple containing validation data. by default None.
    seed
        Random seed for initialization.
    n_chains
        Number of MCMC sampling chains.
    chain_len
        Number of sampling steps during the MCMC-Sampling stage (per chain).
    warmup
        Number of warmup (burn-in) steps before the MCMC-Sampling (per chain).
    n_samples
        Number of samples to take from each Gaussian-distribution during prediction.
    params_
        Parameters generated from DE init stage.
    samples_
        Sampled model parameters from mcmc stage.
    history_
        Loss and metric records during DE init stage.
    model_
        The initialized model class.
    is_fitted_
        A flag indicating weather the model has been fitted or not.
    n_features_in_
        Number of input features (the size of the last input dimension).

    Methods
    -------
    fit(X, y=None, n_devices=-1)
        Fit the model to the training data.
    predict(X)
        Predict the output for the given input data using the trained model.
    sample_from_samples(x, n_devices=1, batch_size=-1)
        Take samples from sampled Gaussian distributions based on model params.
    predict_with_credibility_eti(X, a=0.95)
        Make prediction with a credible interval.
    predict_as_de(X, n_devices=-1)
        Predict with model as a deep ensemble.
    tree_flatten()
        Serialize module into a JAX PyTree.
    tree_unflatten(aux_data, children)
        Build module from a serialized PyTree.
    log_prior(params):
        Calculate the log of the prior probability for a set of params.
    logdensity_for_batch(params, carry, batch)
        Evaluate log-density for a batch of data.
    burn_in_loop(rng, params, n_burns, warmup)
        Perform burn-in for sampler.
    mcmc_sampling(model_states, rng_key, train, n_devices, parallel_batch_size, mask)
        Perform MCMC-burn-in and sampling.
    """

    def __init__(
        self,
        model_class: Type[BasicModule] = FullyConnectedModule,
        model_kwargs: Optional[Dict[str, Any]] = None,
        n_chains: int = 1,
        chain_len: int = 1,
        warmup: int = 1,
        n_samples: int = 1,
        optimizer_class: Type[optax._src.base.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss: loss.NLLLoss = loss.GaussianNLLLoss(),
        batch_size: int = 1,
        epochs: int = 1,
        metrics: Optional[list] = None,
        validation_size: Optional[Union[float]] = None,
        # ADD above support to `Tuple[ArrayLike, ArrayLike], datasets.BasicDataset`
        seed: int = cnfg.General.SEED,
        # **kwargs,
    ):
        r"""Initialize the estimator architecture and training parameters.

        Parameters
        ----------
        model_class
            The neural network model class wrapped by the estimator.
        model_kwargs
            The kwargs used to init the wrapped model.
        n_chains
            Number of MCMC sampling chains.
        chain_len
            Number of sampling steps during the MCMC-Sampling stage (per chain).
        warmup
            Number of warmup (burn-in) steps before the MCMC-Sampling (per chain).
        n_samples
            Number of samples to take from each Gaussian-distribution during
            prediction.
        optimizer_class
            The optimizer class used by the estimator for training.
        optimizer_kwargs
            The kwargs used to init optimizer.
        loss
            A class representing the loss function.
        batch_size
            Number of samples per batch (size of first dimension).
        epochs
            Number of epochs for the DE-Initialization stage (per chain).
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
        # super().__init__(**kwargs)
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
        self.chain_len = chain_len
        self.warmup = warmup
        self.n_samples = n_samples

        self.params_: List[Union[FrozenDict, Dict]] = [dict()]
        self.samples_: List[Union[FrozenDict, Dict]] = [dict()]
        self.history_: Array = None  # type: ignore
        self.model_: BasicModule = None  # type: ignore
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None

    def tree_flatten(self) -> Tuple[Sequence[Union[ArrayLike, Dict]], Any]:
        r"""Specify how to serialize estimator into a JAX pytree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
             - The `children`, containing arrays & pytrees.
             - The `aux_data`, containing static and hashable data.
        """
        atts = {
            "n_chains": self.n_chains,
            "chain_len": self.chain_len,
            "warmup": self.warmup,
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            # "epochs": self.epochs,
            "validation_size": self.validation_size,
            "seed": self.seed,
            "is_fitted_": self.is_fitted_,
            "n_features_in_": self.n_features_in_,
        }
        children = (
            self.model_,
            self.params_,
            self.samples_,
            self.history_,
            self.loss,
            atts,
        )  # children must contain arrays & pytrees
        aux_data = (
            self.model_class,
            self.model_kwargs,
            self.optimizer_class,
            self.optimizer_kwargs,
            self.metrics,
            self.epochs,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[Any, ...],
        children: Tuple[Union[ArrayLike, Dict], ...],  # type: ignore
    ) -> "BDEEstimator":
        r"""Specify how to build an estimator from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & pytrees.

        Returns
        -------
        BDEEstimator
            Reconstructed estimator.
        """
        atts = children[-1]
        res = cls(
            model_class=aux_data[0],
            model_kwargs=aux_data[1],
            n_chains=atts["n_chains"],
            chain_len=atts["chain_len"],
            warmup=atts["warmup"],
            n_samples=atts["n_samples"],
            optimizer_class=aux_data[2],
            optimizer_kwargs=aux_data[3],
            loss=children[4],  # type: ignore
            batch_size=atts["batch_size"],
            epochs=aux_data[5],
            metrics=aux_data[4],
            validation_size=atts["validation_size"],
            seed=atts["seed"],
        )
        res.model_ = children[0]  # type: ignore
        res.params_ = children[1]  # type: ignore
        res.samples_ = children[2]  # type: ignore
        res.is_fitted_ = atts["is_fitted_"]
        res.n_features_in_ = atts["n_features_in_"]
        res.history_ = children[3]
        return res

    def _more_tags(self):
        r"""Define tags for SKlearn."""
        return {
            "_xfail_checks": {
                # Note: By default SKlearn assumes models
                #  do not support complex valued data.
                #  If we decide we want to support it,
                #  the following line should be uncommented.
                # "check_complex_data": "Complex data is supported.",
                "check_dtype_object": (
                    "Numpy input not supported. `jax.numpy` is required."
                ),
                "check_fit1d": "1D data is not supported.",
                "check_no_attributes_set_in_init": (
                    "The model must set some internal attributes like params "
                    "in order to to properly turn it into a pytree."
                ),
                "check_n_features_in": (
                    "Needs to be set before fitting to "
                    "allow passing when flattening pytree."
                ),
            },
            # "array_api_support": True,
            "multioutput_only": True,
            "X_types": ["2darray", "2dlabels"],
        }

    @jax.jit
    def log_prior(self, params) -> float:
        r"""Calculate the log of the prior probability for a set of params.

        Parameters
        ----------
        params
            A PyTree of model parameters.

        Returns
        -------
        float
            The log-prior probability of the parameters.
        """
        # TODO: Make customizable at init time

        @jax.jit
        def sum_log_pdf(x):
            return stats.norm.logpdf(x).sum()

        res = jax.tree.map(
            f=sum_log_pdf,
            tree=params,
        )
        res = jax.tree.reduce(
            function=operator.add,
            tree=res,
        )
        return res

    # @jax.jit
    def _prior_fitting(
        self,
        model_states,
        train,
        valid,
        metrics,
        mask: Array,
    ) -> Tuple[Any, Array]:
        r"""Init BDE using non-Bayesian parallelized training before sampling.

        Initialize BDE by performing regular neural network optimization
        prior to sampling.
        """

        @jax.jit  # noqa: D202
        def _parallel_training(
            mod_states,
            masks,
            epochs: Array,
            f_loss: loss.NLLLoss,
            metrics: Array,
            train: datasets.BasicDataset,
            valid: datasets.BasicDataset,
        ):
            r"""Perform parallel training."""

            @jax.jit
            def _scanned_training(_, xxx):
                r"""Wrap training loop for scanning."""
                mod_state, cond = xxx

                # TODO: The following commented out code is a more efficient
                #  implementation since it skips masked init steps,
                #  but it doesn't work due to a missmatch between the signatures of the
                #  true and false functions in `cond_with_const_f`.
                #  Once the source of the missmatch is found/ fixed, replace the
                #  evaluation of `res` with this one.

                # res = cond_with_const_f(
                #     cond.astype(bool),
                #     partial(
                #         training.jitted_training,
                #         f_loss=f_loss,
                #         epochs=epochs,
                #         metrics=metrics,
                #         train=train,
                #         valid=valid,
                #     ),
                #     mod_state,
                #     mod_state,
                # )
                res = training.jitted_training(
                    mod_state,
                    epochs=epochs,
                    f_loss=f_loss,
                    metrics=metrics,
                    train=train,
                    valid=valid,
                )
                return None, res

            return jax.lax.scan(
                f=_scanned_training,
                init=None,
                xs=[mod_states, masks],
            )

        _, (params, history) = jax.pmap(
            fun=_parallel_training,
            in_axes=(0, 0, None, None, None, None, None),
            # static_broadcasted_argnums=[],
        )(
            model_states,
            mask,
            jnp.arange(self.epochs),
            self.loss,
            jnp.array(metrics),
            train,
            valid,
        )
        return params, history

    @jax.jit
    def logdensity_for_batch(
        self,
        params,
        carry: float,
        batch: Tuple[ArrayLike, ArrayLike],
    ) -> Tuple[Array, None]:
        r"""Evaluate log-density for a batch of data.

        Parameters
        ----------
        params
            Parameters of the evaluated model.
        carry
            log-prior + logdensity of previous batches.
        batch
            Current batch to be evaluated.

        Returns
        -------
        Tuple:
         - Updated logdensity (carry + current batch value).
         - None (used for compatibility with `jax.lax.scan`)
        """
        x, y = batch
        y_pred = self.model_.apply(params, x)
        res_loss = -self.loss(y, y_pred)  # Of type negative log-likelihood loss
        return carry + jnp.sum(res_loss), None

    @staticmethod
    @partial(jax.jit, static_argnames=["warmup", "n_burns"])
    def burn_in_loop(
        rng: PRNGKeyArray,
        params: ArrayLike,
        n_burns: int,
        warmup,
    ):
        r"""Perform burn-in for sampler."""
        return warmup.run(jnp.array(rng).reshape(), params, n_burns)

    def mcmc_sampling(
        self,
        model_states: train_state.TrainState,
        rng_key: PRNGKeyArray,
        train: datasets.BasicDataset,
        n_devices: int,
        parallel_batch_size: int,
        mask: Array,
    ) -> List[Dict]:
        r"""Perform MCMC-burn-in and sampling.

        Parameters
        ----------
        model_states
            Initial model states for sampler.
        rng_key
            A key used to initialize the random processes.
        train
            Training dataset. The dataset must include only 1 batch (full-batch).
        n_devices
            Exact number of computational devices used for parallelization.
        parallel_batch_size
            The number of chains to compute on each device when running parallel
            computations.
        mask
            A 2D-array indicating which chains are used for padding during
            parallelization (shape = (`n_devices`, `batch_size`)).
            - Chains corresponding to the value `one` in the mask will be evaluated
              and sampled from.
            - Chains corresponding to the value `zero` in the mask will be ignored
              when possible and return pseudo-samples which can be discarded.

        Returns
        -------
        List[Dict]
            Samples from all mcmc-chains.
        """
        sample_keys = jax.random.split(rng_key, (parallel_batch_size * n_devices) + 1)
        split_key, sample_keys = sample_keys[0], sample_keys[1:].reshape(n_devices, -1)

        @jax.jit
        def logdensity(params):

            @jax.jit
            def _batch_iter_log_density(carry, batch):
                return self.logdensity_for_batch(
                    params=params,
                    carry=carry,
                    batch=batch,
                )

            res, _ = jax.lax.scan(
                f=_batch_iter_log_density,
                init=self.log_prior(params),
                xs=train.get_scannable(),  # Must be single-batched (see batch_mcmc_)
            )
            return res

        @partial(jax.jit, static_argnums=2)
        def burn_in_loop_wrapper(
            rng: PRNGKeyArray,
            params: ArrayLike,
            n_burns: int,
            pmask,
        ):
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                logdensity,
                progress_bar=False,
            )

            @jax.jit
            def sub_fun(_, xxx):
                key, prm, cond = xxx
                res = self.burn_in_loop(key, prm, n_burns, warmup)

                # TODO: The following commented out code is a more efficient
                #  implementation since it skips masked warmup steps,
                #  but it doesn't work due to a missmatch between the signatures of the
                #  true and false functions in `jax.lax.cond`.
                #  Once the source of the missmatch is found/ fixed, the line above can
                #  be replaced by this code for slightly more efficiency.

                # @partial(jax.jit, static_argnums=[3, 4])
                # def bruh(
                #     rng: PRNGKeyArray,
                #     params: ArrayLike,
                #     data: datasets.BasicDataset,
                #     n_burns: int,
                #     warmup,
                # ):
                #     (init_state_sampling, nuts_params), warmup_info = self.burn_in_loop(  # noqa: E501
                #         rng=rng,
                #         params=params,
                #         data=data,
                #         n_burns=1,
                #         warmup=warmup,
                #     )
                #
                #     @jax.jit
                #     def broadcaster(x):
                #         new_shape = (n_burns,) + x.shape
                #         return jnp.broadcast_to(x[None], new_shape)
                #
                #     init_state_sampling = jax.tree.map(
                #         broadcaster,
                #         init_state_sampling,
                #     ),
                #     warmup_info = jax.tree.map(broadcaster, warmup_info),
                #     return (init_state_sampling, nuts_params), warmup_info
                #
                # res = jax.lax.cond(
                #     cond.astype(bool),
                #     partial(self.burn_in_loop, warmup=warmup, n_burns=n_burns),
                #     partial(bruh, warmup=warmup, n_burns=n_burns),
                #     key,
                #     prm,
                #     data,
                # )
                return None, res

            return jax.lax.scan(
                f=sub_fun,
                init=None,
                xs=[rng, params, pmask],
            )

        _, ((init_state_sampling, nuts_params), warmup_info) = jax.pmap(
            fun=burn_in_loop_wrapper,
            in_axes=(0, 0, None, 0),
            static_broadcasted_argnums=(2,),
        )(
            sample_keys,
            model_states.params,
            self.warmup,  # num_steps
            mask,
        )

        @partial(jax.jit, static_argnames=["num_samples"])
        def inference_loop(
            rng_key,
            initial_state,
            nuts_params,
            num_samples,
        ) -> SamplingAlgorithm:
            kernel = blackjax.nuts(logdensity, **nuts_params).step

            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(
                f=one_step,
                init=initial_state,
                xs=keys,
            )

            return states

        sample_keys = jax.random.split(
            split_key,
            (parallel_batch_size * n_devices),
        ).reshape(n_devices, -1)

        @partial(jax.jit, static_argnums=3)
        def inference_loop_wrapper(
            rng_key,
            initial_state,
            nuts_params,
            num_samples,
            pmask,
        ):
            @jax.jit
            def broadcaster(x):
                new_shape = (self.chain_len,) + x.shape
                return jnp.broadcast_to(x[None], new_shape)

            @jax.jit
            def sub_fun(_, xxx):
                key, init_state, n_params, cond = xxx
                f_true = partial(inference_loop, num_samples=num_samples)
                res = cond_with_const_f(
                    cond.astype(bool),
                    f_true,
                    jax.tree.map(broadcaster, init_state),
                    key,
                    init_state,
                    n_params,
                )
                return None, res

            return jax.lax.scan(
                f=sub_fun,
                init=None,
                xs=[rng_key, initial_state, nuts_params, pmask],
            )

        _, pmap_states = jax.pmap(
            fun=inference_loop_wrapper,
            in_axes=(0, 0, 0, None, 0),
            static_broadcasted_argnums=(3,),
        )(sample_keys, init_state_sampling, nuts_params, self.chain_len, mask)
        return pmap_states.position

    def _prep_fit_params(
        self,
        x: ArrayLike,
        y: Optional[ArrayLike] = None,
        seed: int = cnfg.General.SEED,
    ) -> Tuple[
        List,
        optax._src.base.GradientTransformation,
        datasets.BasicDataset,
        datasets.BasicDataset,
    ]:
        r"""Handle model and parameter initialization before fitting."""
        metrics, optimizer, train, valid = super()._prep_fit_params(
            x=x,
            y=y,
            seed=seed,
        )
        if self.model_kwargs is None:
            model_kwargs: Dict = {
                "n_output_params": 2,
            }
            self.model_ = self.model_class(**model_kwargs)
        return metrics, optimizer, train, valid

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        n_devices: int = -1,
    ) -> "BDEEstimator":
        r"""Fit the function to the given data.

        Parameters
        ----------
        X
            The input data.
        y
            The labels.
            If y is None, X is assumed to include the labels as well.
        n_devices
            Number of devices to use for parallelization.
            `-1` means using all available devices.
            If a number greater than all available devices is given,
            the max number of devices is used.

        Returns
        -------
        BDEEstimator
            The fitted estimator.
        """
        self.samples_ = [dict()]  # Reset before fitting
        self.batch_mcmc_ = X.shape[0]
        n_devices, parallel_batch_size, pad_size, pmask = eval_parallelization_params(
            n_items=self.n_chains,
            n_devices=n_devices,
        )

        split_key = jax.random.key(seed=self.seed)
        rng = jax.random.split(split_key, (parallel_batch_size * n_devices) + 2)
        init_key_list, prep_key, split_key = rng[:-2], rng[-2], rng[-1]
        init_key_list = init_key_list.reshape(n_devices, -1)
        metrics, optimizer, train, valid = self._prep_fit_params(
            x=X,
            y=y,
            seed=int(jax.random.randint(prep_key, (), 0, jnp.iinfo(jnp.int32).max)),
        )
        model_states = pv_map(
            fun=self.init_inner_params,
            in_axes=(None, None, 0),
            static_broadcasted_argnums=[0, 1],
        )(X.shape[-1], optimizer, init_key_list)
        model_states, self.history_ = self._prior_fitting(
            model_states,
            train,
            valid,
            metrics,
            pmask,
        )
        self.params_ = model_states.params

        split_key, rng_key = jax.random.split(split_key)
        train.batch_size = self.batch_mcmc_
        self.samples_ = self.mcmc_sampling(
            model_states,
            rng_key,
            train,
            n_devices,
            parallel_batch_size,
            pmask,
        )
        self.samples_, self.params_ = post_pmap(
            pmask,
            self.samples_,
            self.params_,
        )
        self.samples_ = jax.tree.map(
            f=jnp.concatenate,
            tree=self.samples_,
        )
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[-1]
        return self

    @staticmethod
    # @jax.jit
    def _sample_from_samples_general(
        x: ArrayLike,
        samples,
        model: BasicModule,
        seed: int = cnfg.General.SEED,
        n_samples: int = 1,
        n_mcmc_samples: int = 1,
        n_devices: int = 1,
        batch_size: int = -1,
    ) -> Array:
        r"""Take samples from sampled Gaussian distributions.

        The mean and std predicted by the sampled models define Gaussian
        distributions. Take samples from these distributions.
        """
        # TODO: Parallelize
        split_key = jax.random.key(seed=seed)
        split_key = jax.random.split(split_key, n_mcmc_samples + 1)
        split_key, rng = split_key[0], split_key[1:]

        @jax.jit
        def define_dist(_, xx):
            params, key = xx
            pred = model.apply(params, x)
            m, s = jnp.split(pred, 2, axis=-1)
            samp_shape = (n_samples,) + ((1,) * (m.ndim - 1)) + m.shape[-1:]
            g_samples = jax.random.normal(key, samp_shape)
            g_samples = (g_samples * s) + m
            g_samples = jnp.moveaxis(g_samples, 0, -1)
            return None, g_samples

        _, res = jax.lax.scan(
            f=define_dist,
            init=None,
            xs=[samples, rng],
        )
        res = jnp.concatenate(res, axis=-1)
        return res

    # @jax.jit
    def sample_from_samples(
        self,
        x: ArrayLike,
        n_devices: int = 1,
        batch_size: int = -1,
    ) -> Array:
        r"""Take samples from sampled Gaussian distributions based on model params.

        The mean and std predicted by the sampled models define Gaussian
        distributions. Take samples from these distributions.

        Parameters
        ----------
        X
            The input data.
        n_devices
            ...
        batch_size
            ...

        Returns
        -------
        Array
            The last dim of the array is a list of samples taken from each predicted
            distribution in each batch.
            The shape is:
            `(b_1, b_2, ..., b_2, output_size / 2, self.n_samples * self.n_chains)`
        """
        return self._sample_from_samples_general(
            x=x,
            samples=self.samples_,
            model=self.model_,
            seed=self.seed,
            n_samples=self.n_samples,
            n_mcmc_samples=self.n_chains * self.chain_len,
            n_devices=n_devices,
            batch_size=batch_size,
        )

    # @jax.jit
    def predict_with_credibility_eti(
        self,
        X: ArrayLike,
        a: float = 0.95,
    ) -> Tuple[Array, Array, Array]:
        r"""Make prediction with a credible interval.

        Parameters
        ----------
        X
            The input data.
        a
            Size of credibility interval (in probability: 0 - 1).
        Returns
        -------
        3 arrays with:
        - Predicted values (median of samples).
        - Lower value of confidence interval per prediction.
        - Upper value of confidence interval per prediction.
        """
        bde.utils.utils.check_predict_input(X, self.is_fitted_)
        samples = self.sample_from_samples(
            x=X,
            # n_devices=1,
            # batch_size=-1,
        )
        pred = jnp.median(samples, axis=-1)
        tail_size = 100 * (1 - a) / 2
        i_low = jnp.percentile(samples, tail_size, axis=-1)
        i_high = jnp.percentile(samples, 100 - tail_size, axis=-1)
        return pred, i_low, i_high

    # @jax.jit
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
            Predicted values (mean of samples).
        """
        bde.utils.utils.check_predict_input(X, self.is_fitted_)
        samples = self.sample_from_samples(
            x=X,
            # n_devices=1,
            # batch_size=-1,
        )
        return jnp.mean(samples, axis=-1)

    @jax.jit
    def predict_as_de(
        self,
        X: ArrayLike,
        n_devices: int = -1,
    ) -> Array:
        r"""Predict with model as a deep ensemble.

        This method ignores the samples data and uses the initialization params only.

        Parameters
        ----------
        X
            The input data.
        n_devices
            Number of devices to use for parallelization.
            `-1` means using all available devices.
            If a number greater than all available devices is given,
            the max number of devices is used.

        Returns
        -------
        Array
            Predicted values (mean_of_predictions).
        """
        bde.utils.utils.check_predict_input(X, self.is_fitted_)
        # res = jax.pmap(
        #     fun=self.model_.apply,
        #     in_axes=(0, None),
        # )(
        #     self.params_,
        #     X,
        # )
        # TODO: Parallelize

        @jax.jit
        def sub_f(_, params):
            return 0.0, self.model_.apply(params, X)

        res = jax.lax.scan(
            f=sub_f,
            init=0.0,
            xs=self.params_,
        )[1]
        return res.mean(axis=0)


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
        if not isinstance(model.n_input_params, Sequence):
            n_features = model.n_input_params
        else:
            raise NotImplementedError
    if model.n_input_params is None:
        if n_features is None:
            raise ValueError(
                "`n_features` and `model.n_input_params` can't both be `None`."
            )
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
    tests_path = (
        pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_models"
    )
    pytest.main([str(tests_path)])
