import pytest

import jax
import jax.numpy as jnp

from bde.ml.datasets import DatasetWrapper
from bde.utils import configs as cnfg


@pytest.fixture
def make_range_dataset():
    def func(
            n_items,
            seed,
    ):
        data = jnp.arange(n_items, dtype=int).reshape(-1, 1)
        return DatasetWrapper(x=data, y=data, batch_size=n_items, seed=seed), data
    return func


@pytest.fixture
def recreate_with_pytree():
    def func(
            ds,
    ):
        return DatasetWrapper.tree_unflatten(*ds.tree_flatten()[::-1])
    return func


if __name__ == '__main__':
    pytest.main()
