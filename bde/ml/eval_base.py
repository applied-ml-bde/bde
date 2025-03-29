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
    """

    @abstractmethod
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
        ...

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

    @abstractmethod
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> Array:
        r"""Evaluate the loss.

        Returns an unreduced evaluation of the loss, i.e. the loss is calculated
        separately for each item in the batch.

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
    def apply_reduced(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        r"""Evaluate and reduces the loss.

        The loss is evaluated separately for each item in the batch and the loss of
        all batches is reduced by arithmetic mean to a single value.

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
    ) -> "Loss":
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
            Reconstructed loss function.
        """
        ...


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

    @abstractmethod
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> Array:
        r"""Evaluate the loss.

        Returns an unreduced evaluation of the loss, i.e. the loss is calculated
        separately for each item in the batch.

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
    def apply_reduced(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        r"""Evaluate and reduces the loss.

        The loss is evaluated separately for each item in the batch and the loss of
        all batches is reduced by arithmetic mean to a single value.

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
    ) -> "Metric":
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
            Reconstructed loss function.
        """
        ...


if __name__ == "__main__":
    ...
