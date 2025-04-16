r"""Defines APIs for evaluation classes.

This module contains implementations of loss functions and their wrappers
used in training Bayesian Neural Networks within the Bayesian Deep Ensembles (BDE)
framework.

Classes
-------
- `Evaluation`: Defines the base API used by all model evaluation classes.
- `Loss`: Defines the API used by loss-related classes.
- `Metric`: Defines the API used by metric classes.

"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)

import jax
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike


@register_pytree_node_class
class Evaluation(ABC):
    r"""An abstract base class defining an API for model evaluation related classes.

    Methods
    -------
    __call__(y_true, y_pred, **kwargs)
        Abstract method to be implemented by subclasses, defining the evaluation.
    tree_flatten()
        Used to turn the class into a jitible PyTree.
    tree_unflatten(aux_data, children)
        A class method used to recreate the class from a PyTree.
    apply_reduced(y_true, y_pred, **kwargs)
        The evaluation is performed separately for each item in the batch and the
        loss of all batches is reduced to a single value.
        The default implementation takes the arithmetic mean as the reduction, but
        classes implementing this API are free to reimplement this method.
    get_opt_factor()
        Returns the value of the `_opt_factor` parameter, indicating whether the
        evaluation needs to be minimized (1.0) or maximized (-1.0).

    Parameters
    ----------
    _do_minimize
        A parameter which needs to be set by classes implementing the API,
        indicating whether the evaluation needs to be minimized or maximized:
        True if the evaluation needs to be minimized.
        False if the evaluation needs to be Maximized.
    _opt_factor
        A parameter derived from `_do_minimize` which is used in jitted calculations.
        1.0 if the evaluation needs to be minimized.
        -1.0 if the evaluation needs to be Maximized.
    """

    _do_minimize: bool
    _opt_factor: float

    def __init__(self):
        r"""Create an instance of the evaluation class."""

        @jax.jit
        def f_true():
            return 1.0

        @jax.jit
        def f_false():
            return -1.0

        self._opt_factor = jax.lax.cond(
            self._do_minimize,
            f_true,
            f_false,
        )

    @jax.jit
    def get_opt_factor(self) -> float:
        r"""Get the optimization factor for the evaluation.

        Returns
        -------
        float
            1.0 if the optimization needs to be minimized.
            -1.0 if the optimization needs to be maximized.
        """
        return self._opt_factor

    @abstractmethod
    def call(
            self,
            y_true: ArrayLike,
            y_pred: ArrayLike,
            **kwargs,
    ) -> Array:
        r"""Perform evaluation.

        Returns an unreduced evaluation, i.e. evaluate each item separately in the
        batch.

        Parameters
        ----------
        y_true
            The ground truth labels.
        y_pred
            The predictions.

        Returns
        -------
        Array
            The unreduced loss value.
        """
        ...

    @jax.jit
    def _consolidate_batch(
        self,
        data: ArrayLike,
        **kwargs
    ) -> Array:
        r"""Consolidate the calculation to 1 item per batch.

        Parameters
        ----------
        data
            Raw data after evaluation.

        Returns
        -------
        Array
            The data where each item in the batch is reduced to one value.
        """
        return data.mean(axis=tuple(range(1, data.ndim)))

    @jax.jit
    def __call__(
            self,
            y_true: ArrayLike,
            y_pred: ArrayLike,
            **kwargs,
    ) -> Array:
        r"""Perform evaluation.

        Returns an unreduced evaluation, i.e. evaluate each item separately in the
        batch.

        Parameters
        ----------
        y_true
            The ground truth labels.
        y_pred
            The predictions.

        Returns
        -------
        Array
            The unreduced loss value.
        """
        res = self.call(y_true=y_true, y_pred=y_pred, **kwargs)
        return self._consolidate_batch(res)

    @jax.jit
    def apply_reduced(
            self,
            y_true: ArrayLike,
            y_pred: ArrayLike,
            **kwargs,
    ) -> ArrayLike:
        r"""Evaluate and reduce results.

        The evaluation is performed separately for each item in the batch, and it is
        reduced arithmetic mean to a single value.

        Parameters
        ----------
        y_true
            The ground truth labels.
        y_pred
            The predictions.
        **kwargs
            Other keywords that may be passed to the unreduced loss function.

        Returns
        -------
        Array
            The reduced loss value.
        """
        return self(y_true=y_true, y_pred=y_pred, **kwargs).mean()

    @abstractmethod
    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize module into a JAX PyTree.

        Returns
        -------
        A tuple with 2 elements:
         - The `children`, containing arrays & PyTrees
         - The `aux_data`, containing static and hashable data.
        """
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(
            cls,
            aux_data: Optional[Tuple],
            children: Tuple,
    ) -> "Evaluation":
        r"""Specify how to build a module from a JAX PyTree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & PyTrees.

        Returns
        -------
        Loss
            Reconstructed evaluation function.
        """
        ...


@register_pytree_node_class
class Loss(Evaluation, ABC):
    r"""An abstract base class defining an API for loss function classes.

    Methods
    -------
    __call__(y_true, y_pred, **kwargs)
        Abstract method to be implemented by subclasses, defining the loss evaluation.
    tree_flatten()
        Used to turn the class into a jitible PyTree.
    tree_unflatten(aux_data, children)
        A class method used to recreate the class from a PyTree.
    apply_reduced(y_true, y_pred, **kwargs)
        The loss is evaluated separately for each item in the batch and the loss of
        all batches is reduced to a single value.
        The default implementation takes the arithmetic mean as the reduction, but
        classes implementing this API are free to reimplement this method.
    """

    _do_minimize: bool = True


@register_pytree_node_class
class Metric(Evaluation, ABC):
    r"""An abstract base class defining an API for metric function classes.

    Methods
    -------
    __call__(y_true, y_pred, **kwargs)
        Abstract method to be implemented by subclasses, defining the loss evaluation.
    tree_flatten()
        Used to turn the class into a jitible PyTree.
    tree_unflatten(aux_data, children)
        A class method used to recreate the class from a PyTree.
    apply_reduced(y_true, y_pred, **kwargs)
        The loss is evaluated separately for each item in the batch and the loss of
        all batches is reduced to a single value.
        The default implementation takes the arithmetic mean as the reduction, but
        classes implementing this API are free to reimplement this method.
    """

    _do_minimize: bool = False


if __name__ == "__main__":
    ...
