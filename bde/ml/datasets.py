r"""Dataset related functionality.

This module provides functionality for handling data and datasets for model training.

Classes
-------
- `BasicDataset`: An abstract base class defining an API for dataset classes.
- `DatasetWrapper`: Wraps dataset functionality around 2 arrays: features and labels.

"""

import pathlib
from abc import ABC, abstractmethod
from typing import (
    Any,
    Sequence,
    Tuple,
)

import chex
import jax
import pytest
from jax import Array
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from bde.utils import configs as cnfg


class BasicDataset(ABC):
    r"""An abstract base class defining an API for dataset classes.

    Attributes
    ----------
    batch_size()
        The number of items in each batch (leading axis).
        Updating the `batch_size` updates related attributes.
    _batch_size
        Any class implementing this API must handle the logic of `batch_size` via
        this attribute.
    seed()
        The seed used to generate randomness.
        Updating the `seed` resets the randomness accordingly.
    _seed
        Any class implementing this API must handle the logic of `seed` via this
        attribute.

    Methods
    -------
    set_state(**kwargs)
        Updates attributes of state.
    tree_flatten()
        Used to turn the class into a jitible PyTree.
    tree_unflatten(aux_data, children)
        A class method used to recreate the class from a PyTree.
    shuffle()
        Randomly rearrange the items in the dataset while preserving the connection
        between related items such as data and labels.
    gen_empty()
        Create an empty version of the current dataset.
    __len__()
        Returns the number of batches in the dataset.
    __getitem__(ids)
        Return a batch from the dataset.
    __iter__()
        Iterate through the dataset.
    get_scannable()
        Returns a flattened and shuffled form of the dataset which can be used with
        `jax.lax.scan()`.
    """

    _batch_size: int
    _seed: int

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
    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize the dataset into a JAX PyTree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
             - The `children`, containing arrays & PyTrees
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
        r"""Specify how to construct a dataset from a JAX PyTree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data.
        children
            Contain arrays & PyTrees.

        Returns
        -------
        BasicDataset
            Reconstructs object from PyTree.
        """
        ...

    @abstractmethod
    def shuffle(self) -> "BasicDataset":
        r"""Randomly reorganize the dataset.

        Perform a random shuffle on the dataset items based on the dataset's seed.

        Returns
        -------
        BasicDataset
            Shuffled variation of the dataset.
        """
        ...

    @abstractmethod
    def gen_empty(self) -> "BasicDataset":
        r"""Create an empty version of the current dataset."""
        ...

    @abstractmethod
    def __len__(self) -> int:
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
    def __iter__(self):
        r"""Iterate through the dataset."""
        ...

    @abstractmethod
    def get_scannable(self) -> Tuple[Array, Array]:
        r"""Return the dataset in a scannable form.

        Returns the dataset in a scannable form, corresponding to its shuffled state.

        Returns
        -------
            A tuple representing the x and y in their shuffled form.
        """
        ...

    @property
    @jax.jit
    def batch_size(self) -> int:
        r"""The number of items in each batch (leading axis)."""
        return self._batch_size

    @batch_size.setter
    @abstractmethod
    def batch_size(
        self,
        batch_size: int,
    ) -> None:
        r"""Change the batch size.

        Provide logic for updating the batch size while keeping related values
        consistent (like size).

        Parameters
        ----------
        batch_size
            The new batch size.
        """
        ...

    @property
    @jax.jit
    def seed(self) -> int:
        r"""The seed used to generate randomness."""
        return self._seed

    @seed.setter
    @abstractmethod
    def seed(
        self,
        seed: int,
    ) -> None:
        r"""Change the seed.

        Provide logic for updating the seed and restarts the randomness using the new
        seed.

        Parameters
        ----------
        seed
            The new seed.
        """
        ...


@register_pytree_node_class
class DatasetWrapper(BasicDataset):
    r"""Create a dataset object around a data and labels pair.

    A dataset wrapping a features array and a corresponding labels array.

    Attributes
    ----------
    batch_size()
        The number of items in each batch (leading axis).
        Updating the `batch_size` updates related attributes.
    seed()
        The seed used to generate randomness.
        Updating the `seed` resets the randomness accordingly.
    x
        The features stored in the dataset.
    y
        The labels stored in the dataset.
    n_items_
        Total number of items in the dataset.
    size_
        Number of full batches in the dataset.
    items_lim_
        Number of items that can be put into full batches (`size_` * `batch_size`).
    rng_key
        The randomness key used to determine shuffling.
    split_key
        The randomness key used to update the `rng_key`.
    was_shuffled_
        A flag indicating whether the dataset was shuffled or not.
        If not, the items will be ordered the same as they were at init time.
    assignment
        An ordering array corresponding to the shuffled state.

    Methods
    -------
    set_state(**kwargs)
        Updates attributes of state.
    tree_flatten()
        Used to turn the class into a jitible PyTree.
    tree_unflatten(aux_data, children)
        A class method used to recreate the class from a PyTree.
    shuffle()
        Randomly rearrange the items in the dataset while preserving the connection
        between related items such as data and labels.
    gen_empty()
        Create an empty version of the current dataset.
    __len__()
        Returns the number of batches in the dataset.
    __getitem__(ids)
        Return a batch from the dataset.
    __iter__()
        Return a generator which retrieves all batches from the dataset.
    get_scannable()
        Returns a flattened and shuffled form of the dataset which can be used with
        `jax.lax.scan()`.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        batch_size: int = 1,
        seed: int = cnfg.General.SEED,
    ):
        r"""Initiate the class.

        Parameters
        ----------
        x
            The features stored in the dataset.
        y
            The labels stored in the dataset.
        batch_size
            The number of items in each batch (leading axis).
        seed
            The seed used to generate randomness.
        """
        chex.assert_equal(x.shape[0], y.shape[0])
        self.x = x.astype(jnp.float32)  # TODO: Needs better handling
        self.y = y.astype(jnp.float32)
        self._batch_size = batch_size
        self._seed = seed

        self.n_items_ = x.shape[0]
        self.size_ = self.n_items_ // self._batch_size
        self.items_lim_ = self.size_ * self._batch_size

        self.split_key = jax.random.key(seed=self._seed)
        self.rng_key = self.split_key
        self.was_shuffled_ = jnp.array(False)
        self.assignment = jnp.arange(self.items_lim_, dtype=int)
        self.assignment = self.assignment.reshape(self.size_, self._batch_size)

    def tree_flatten(self) -> Tuple[Sequence[ArrayLike], Any]:
        r"""Specify how to serialize the dataset into a JAX PyTree.

        Returns
        -------
        Tuple[Sequence[ArrayLike], Any]
            A tuple with 2 elements:
             - The `children`, containing arrays & PyTrees (2 elements).
             - The `aux_data`, containing static and hashable data (5 elements).
        """
        children = (
            self.x,
            self.y,
            self.rng_key,
            self.split_key,
            self.was_shuffled_,
        )  # children must contain arrays & PyTrees
        aux_data = (
            self._batch_size,
            self._seed,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Tuple[int, int],
        children: Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    ) -> "DatasetWrapper":
        r"""Specify how to construct a dataset from a JAX PyTree.

        Parameters
        ----------
        aux_data
            Contains static, hashable data (5 elements).
        children
            Contain arrays & PyTrees (2 elements).

        Returns
        -------
        DatasetWrapper
            Reconstructs object from PyTree.
        """
        res = cls(*children[:2], *aux_data[:2])
        res.rng_key, res.split_key = children[2:4]
        res.was_shuffled_ = children[4]

        @jax.jit
        def f_true():
            return jax.random.permutation(
                res.rng_key,
                res.n_items_,
            )[: (res.size_ * aux_data[0])].reshape(
                res.size_,
                aux_data[0],
            )

        @jax.jit
        def f_false():
            return res.assignment

        res.assignment = jax.lax.cond(
            res.was_shuffled_,
            f_true,
            f_false,
        )
        return res

    @jax.jit
    def shuffle(self) -> "BasicDataset":
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
        )[: self.items_lim_].reshape(
            self.size_,
            self._batch_size,
        )

        return self.set_state(
            assignment=assignment,
            rng_key=rng_key,
            split_key=split_key,
            was_shuffled_=jnp.array(True),
        )

    def gen_empty(self) -> "DatasetWrapper":
        r"""Create an empty version of the current dataset."""
        x_shape, y_shape = (0,) + self.x.shape[1:], (0,) + self.y.shape[1:]
        res = DatasetWrapper(
            x=jnp.empty(x_shape, dtype=self.x.dtype),
            y=jnp.empty(y_shape, dtype=self.y.dtype),
            batch_size=self._batch_size,
            seed=self._seed,
        )
        return res

    @jax.jit
    def __len__(self) -> int:
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

    def __iter__(self):
        r"""Iterate through the dataset."""
        return (self[idx] for idx in range(self.size_))

    @jax.jit
    def get_scannable(self) -> Tuple[Array, Array]:
        r"""Return the dataset in a scannable form, corresponding to its shuffled state.

        Returns
        -------
            A tuple representing the x and y in their shuffled form.
        """
        return self.x[self.assignment], self.y[self.assignment]

    @property
    @jax.jit
    def batch_size(self) -> int:
        r"""The number of items in each batch (leading axis)."""
        # NOTE: Not required due to inheritance. Implemented here to satisfy mypy.
        return self._batch_size

    @batch_size.setter
    def batch_size(
        self,
        batch_size: int,
    ) -> None:
        r"""Change the batch size.

        Provide logic for updating the batch size while keeping related values
        consistent (like size).

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
        self.assignment = jax.random.permutation(
            self.rng_key,
            self.n_items_,
        )[: self.items_lim_].reshape(
            self.size_,
            self._batch_size,
        )

    @property
    @jax.jit
    def seed(self) -> int:
        r"""The number of items in each batch (leading axis)."""
        # NOTE: Not required due to inheritance. Implemented here to satisfy mypy.
        return self._seed

    @seed.setter
    def seed(
        self,
        seed: int,
    ) -> None:
        r"""Change the seed.

        Provide logic for updating the seed and restarts the randomness using the new
        seed.

        Parameters
        ----------
        seed
            The new seed.
        """
        self._seed = seed
        self.split_key = jax.random.key(seed=self._seed)
        if self.was_shuffled_:
            shuffled = self.shuffle()
            self.split_key = shuffled.split_key
            self.rng_key = shuffled.rng_key
            self.assignment = shuffled.assignment
            return
        self.rng_key = self.split_key
        self.assignment = jnp.arange(self.items_lim_).reshape(
            self.size_,
            self._batch_size,
        )


if __name__ == "__main__":
    tests_path = (
        pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_datasets"
    )
    pytest.main([str(tests_path)])
