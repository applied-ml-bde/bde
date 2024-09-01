r"""Dataset related functionality.

This module provides functionality for handling data and datasets for model training.

Classes
-------
- `BasicDataset`: An abstract base class defining an API for dataset classes.
- `DatasetWrapper`: Wraps dataset functionality around 2 arrays.

"""

from abc import ABC, abstractmethod
import chex
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.tree_util import register_pytree_node_class
import pathlib
import pytest
from typing import Any, Union, Optional, Tuple, List, Dict, Sequence

import bde.ml
from bde.utils import configs as cnfg


class BasicDataset(ABC):
    r"""An abstract base class defining an API for dataset classes.

    Methods
    -------
    set_state(**kwargs)
        Updates attributes of state.
    tree_flatten()
        Used to turn the class into a jitible pytree.
    tree_unflatten(aux_data, children)
        A class method used to recreate the class from a pytree.
    shuffle()
        Randomly rearrange the items in the dataset while preserving the connection between related items
        such as data and labels.
    __len__()
        Returns the number of batches in the dataset.
    __getitem__(ids)
        Return a batch from the dataset.
    __iter__()
        Iterate through the dataset.
    get_scannable()
        Return the dataset in a scannable form, corresponding to its shuffled state.
    """

    _batch_size: int
    seed: int

    def set_state(
            self,
            **kwargs,
    ) -> "BasicDataset":
        r"""Return a new instance with updated attributes.

        Used for easily updating the state in jitted functions.

        Parameters
        ----------
        **kwargs
            The attributes to be updated.

        Returns
        -------
            A copy of the object with updated attributes.
        """
        new_state = self.__dict__.copy()
        new_state.update(**kwargs)

        new_instance = self.__class__.__new__(self.__class__)
        new_instance.__dict__.update(**new_state)
        return new_instance

    @abstractmethod
    def tree_flatten(
            self,
    ) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize the dataset into a JAX pytree.

        :return: A tuple with 2 elements:
         - The `children`, containing arrays & pytrees
         - The `aux_data`, containing static and hashable data.
        """
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(
            cls,
            aux_data: Any,
            children: Tuple[ArrayLike, ...],
    ) -> "BasicDataset":
        r"""Specify how to construct a dataset from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & pytrees.

        Returns
        -------
        BasicDataset
            Reconstructs object from PyTree.
        """
        ...

    @abstractmethod
    def shuffle(
            self,
    ) -> "BasicDataset":
        r"""Randomly reorganize the dataset.

        Perform a random shuffle on the dataset items based on the dataset's seed.

        Returns
        -------
        BasicDataset
            Shuffled variation of the dataset.
        """
        ...

    @abstractmethod
    def __len__(
            self,
    ) -> int:
        r"""Return the number of batches in the dataset.

        Returns
        -------
        int
            Number of batches in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(
            self,
            idx: int,
    ) -> Tuple[Array, Array]:
        r"""Retrieve a batch from the dataset.

        Parameters
        ----------
        idx
            Index of the batch to retrieve.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            A batch of data: `(x, y)`.
        """
        ...

    @abstractmethod
    def __iter__(
            self,
    ):
        r"""Iterate through the dataset."""
        ...

    @abstractmethod
    def get_scannable(
            self,
    ) -> Tuple[Array, Array]:
        r"""Return the dataset in a scannable form, corresponding to its shuffled state.

        Returns
        -------
            A tuple representing the x and y in their shuffled form.
        """
        ...

    @property
    @jax.jit
    def batch_size(
            self,
    ) -> int:
        r"""The number of items in each batch (leading axis)."""
        return self._batch_size

    @batch_size.setter
    @abstractmethod
    def batch_size(self, batch_size: int) -> None:
        r"""Change the batch size.

        Provide logic for updating the batch size while keeping related values consistent (like size).

        Parameters
        ----------
        batch_size
            The new batch size.
        """
        ...


@register_pytree_node_class
class DatasetWrapper(BasicDataset):
    r"""Create a dataset object around a data and labels pair.

    # TODO: Complete
    Methods
    -------
    tree_flatten()
        Used to turn the class into a jitible pytree.
    tree_unflatten(aux_data, children)
        A class method used to recreate the class from a pytree.
    shuffle()
        Randomly rearrange the items in the dataset while preserving the connection between related items
        such as data and labels.
    __len__()
        Returns the number of batches in the dataset.
    __getitem__(ids)
        Return a batch from the dataset.
    __iter__()
        Return a generator which retrieves all batches from the dataset.
    get_scannable()
        Returns a flattened and shuffled form of the dataset which can be used with `jax.lax.scan()`.
    batch_size()
        A property for setting the batch size while adjusting other corresponding parameters.
    """
    
    def __init__(
            self,
            x: ArrayLike,
            y: ArrayLike,
            batch_size: int = 1,
            seed: int = cnfg.General.SEED,
    ):
        r"""Initiate the class.

        # TODO: Complete
        """
        chex.assert_equal(x.shape[0], y.shape[0])
        self.x = x
        self.y = y
        self._batch_size = batch_size
        self.seed = seed

        self.n_items_ = x.shape[0]
        self.size_ = self.n_items_ // self._batch_size
        self.items_lim_ = self.size_ * self._batch_size

        self.split_key = jax.random.key(seed=self.seed)
        self.rng_key = self.split_key
        self.was_shuffled_ = jnp.array(False)
        self.assignment = jnp.arange(self.items_lim_).reshape(self.size_, self._batch_size)

    def tree_flatten(
            self,
    ) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize the dataset into a JAX pytree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
             - The `children`, containing arrays & pytrees (2 elements).
             - The `aux_data`, containing static and hashable data (5 elements).
        """
        children = (
            self.x,
            self.y,
            self.rng_key,
            self.split_key,
            self.was_shuffled_,
        )  # children must contain arrays & pytrees
        aux_data = (
            self._batch_size,
            self.seed,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
            cls,
            aux_data: Tuple[Any, ...],
            children: Tuple[ArrayLike, ArrayLike],
    ) -> "DatasetWrapper":
        r"""Specify how to construct a dataset from a JAX pytree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data (5 elements).
        children
            Contain arrays & pytrees (2 elements).

        Returns
        -------
        DatasetWrapper
            Reconstructs object from PyTree.
        """
        res = cls(*children[:2], *aux_data[:2])
        res.rng_key, res.split_key = children[2:4]
        res.was_shuffled_ = children[4]
        res.assignment = jax.lax.cond(
            res.was_shuffled_,
            lambda: jax.random.permutation(
                res.rng_key,
                res.n_items_,
            )[:(res.size_ * aux_data[0])].reshape(res.size_, aux_data[0]),
            lambda: res.assignment,
        )
        return res

    @jax.jit
    def shuffle(
            self,
    ) -> "DatasetWrapper":
        r"""Randomly reorganize the dataset.

        Perform a random shuffle on the dataset items based on the dataset's seed.

        Returns
        -------
        BasicDataset
            Shuffled variation of the dataset.
        """
        rng_key, split_key = jax.random.split(self.split_key)
        assignment = jax.random.permutation(
            rng_key,
            self.n_items_,
        )[:self.items_lim_].reshape(self.size_, self._batch_size)

        return self.set_state(
            assignment=assignment,
            rng_key=rng_key,
            split_key=split_key,
            was_shuffled_=jnp.array(True),
        )

    @jax.jit
    def __len__(
            self,
    ) -> int:
        r"""Return the number of batches in the dataset.

        Returns
        -------
        int
            Number of batches in the dataset.
        """
        return self.size_

    @jax.jit
    def __getitem__(
            self,
            idx: int,
    ) -> Tuple[Array, Array]:
        r"""Retrieve a batch from the dataset.

        Parameters
        ----------
        idx
            Index of the batch to retrieve.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            A tuple consisting of training data and corresponding labels
        """
        idx2 = self.assignment[idx]
        return self.x[idx2], self.y[idx2]

    def __iter__(
            self,
    ):
        r"""Iterate through the dataset."""
        return (self[idx] for idx in range(self.size_))

    @jax.jit
    def get_scannable(
            self,
    ) -> Tuple[Array, Array]:
        r"""Return the dataset in a scannable form, corresponding to its shuffled state.

        Returns
        -------
            A tuple representing the x and y in their shuffled form.
        """
        return self.x[self.assignment], self.y[self.assignment]

    @BasicDataset.batch_size.setter
    @jax.jit
    def batch_size(self, batch_size: int) -> None:
        r"""Change the batch size.

        Provide logic for updating the batch size while keeping related values consistent (like size).

        Parameters
        ----------
        batch_size
            The new batch size.

        Parameters
        ----------
        batch_size
            The new batch size.
        """
        self._batch_size = batch_size
        self.size_ = self.n_items_ // self.batch_size
        self.items_lim_ = len(self) * self.batch_size
        self.assignment = self.assignment.reshape(len(self), self.batch_size)


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_datasets"
    pytest.main([str(tests_path)])
