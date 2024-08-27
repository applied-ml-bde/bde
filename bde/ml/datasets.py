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
from typing import Any, Union, Optional, Tuple, List, Dict

import bde.ml
from bde.utils import configs as cnfg


class BasicDataset(ABC):
    r"""An abstract base class defining an API for dataset classes.

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
    """

    batch_size: int
    seed: int

    @abstractmethod
    def tree_flatten(
            self,
    ) -> Tuple[Optional[Any], Optional[Any]]:
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
            aux_data,
            children,
    ) -> "BasicDataset":
        r"""Specify how to construct a dataset from a JAX pytree.

        :param aux_data: Contains static, hashable data.
        :param children: Contain arrays & pytrees.
        :return:
        """
        ...

    @abstractmethod
    def shuffle(
            self,
    ) -> None:
        r"""Randomly reorganize the dataset.

        Perform a random shuffle on the dataset items based on the dataset's seed.
        """
        ...

    @abstractmethod
    def __len__(
            self,
    ) -> int:
        r"""Return the number of batches in the dataset."""
        ...

    @abstractmethod
    def __getitem__(
            self,
            idx: int,
    ) -> Tuple[ArrayLike, ArrayLike]:
        r"""Retrieve a batch from the dataset.

        :param idx: Index of the batch to retrieve.
        :return: A batch of data.
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
        self.batch_size = batch_size
        self.seed = seed

        self.n_items = x.shape[0]
        self.size = self.n_items // self.batch_size
        self.items_lim = self.size * self.batch_size

        self.split_key = jax.random.key(seed=self.seed)
        self.rng_key = self.split_key
        self.was_shuffled_ = False
        self.assignment = jnp.arange(self.items_lim)
        self.assignment = self.assignment.reshape(self.size, self.batch_size)

    def set_rng_state(
            self,
            rng_key,
            split_key,
    ) -> None:
        r"""Manually provide the random-key for next randomness generation.

        Used mainly for pytree reconstruction.
        """
        self.rng_key, self.split_key = rng_key, split_key

    def update_rng_state(self):
        r"""Prepare random-key for next randomness generation."""
        self.set_rng_state(*jax.random.split(self.split_key))
        self.was_shuffled_ = True

    def _shuffle_without_updates(
            self,
    ) -> None:
        r"""Shuffle the data without updating the random state."""
        self.assignment = jax.lax.cond(
            self.was_shuffled_,
            lambda: jax.random.permutation(self.rng_key, self.n_items)[:self.items_lim].reshape(self.size, self.batch_size),
            lambda: self.assignment,
        )

    def shuffle(
            self,
    ) -> None:
        r"""Randomly reorganize the dataset.

        Perform a random shuffle on the dataset items based on the dataset's seed.
        """
        self.update_rng_state()
        self._shuffle_without_updates()

    @jax.jit
    def __len__(
            self,
    ) -> int:
        r"""Return the number of batches in the dataset."""
        return self.size

    def tree_flatten(
            self,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        r"""Specify how to serialize the dataset into a JAX pytree.

        :return: A tuple with 2 elements:
         - The `children`, containing arrays & pytrees
         - The `aux_data`, containing static and hashable data.
        """
        children = (
            self.x,
            self.y,
        )  # children must contain arrays & pytrees
        aux_data = (
            self.batch_size,
            self.seed,
            self.rng_key,
            self.split_key,
            self.was_shuffled_,
        )  # aux_data must contain static, hashable data.
        return children, aux_data

    @classmethod
    def tree_unflatten(
            cls,
            aux_data,
            children,
    ) -> "DatasetWrapper":
        r"""Specify how to construct a dataset from a JAX pytree.

        :param aux_data: Contains static, hashable data.
        :param children: Contain arrays & pytrees.
        :return:
        """
        res = cls(*children, *aux_data[:2])
        res.set_rng_state(*aux_data[2:4])
        res.was_shuffled_ = aux_data[4]
        res._shuffle_without_updates()
        return res

    @jax.jit
    def __getitem__(
            self,
            idx: int,
    ) -> Tuple[ArrayLike, ArrayLike]:
        r"""Retrieve a batch from the dataset.

        :param idx: Index of the batch to retrieve.
        :return: A tuple consisting of training data and corresponding labels.
        """
        idx2 = self.assignment[idx]
        return self.x[idx2], self.y[idx2]


if __name__ == "__main__":
    tests_path = pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_datasets"
    pytest.main([str(tests_path)])
