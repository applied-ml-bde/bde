import jax
from jax import numpy as jnp
import pathlib
import pytest

from bde.ml.datasets import DatasetWrapper
from bde.utils import configs as cnfg

SEED = cnfg.General.SEED
TESTED_SEEDS = [0, 24, 42]


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


class TestPyTreePacking:
    @staticmethod
    def test_shuffle_randomness_is_conserved(
            make_range_dataset,
            recreate_with_pytree,
    ):
        n_items = 128
        ds, _ = make_range_dataset(n_items=n_items, seed=SEED)
        ds.shuffle()
        ds2 = recreate_with_pytree(ds)
        assert jnp.all(ds.assignment == ds2.assignment)

    @staticmethod
    def test_pytree_recreation_with_no_shuffling_is_ordered(
            make_range_dataset,
            recreate_with_pytree,
    ):
        n_items = 128
        ds, data = make_range_dataset(n_items=n_items, seed=SEED)
        ds2 = recreate_with_pytree(ds)
        assert jnp.all(ds2.assignment.reshape(-1) == data.reshape(-1))


class TestShuffling:
    @staticmethod
    def test_xy_match_after_n_shuffles(
            make_range_dataset,
    ):
        n_shuffles, n_items = 5, 128
        ds, _ = make_range_dataset(n_items=n_items, seed=SEED)
        for n in range(1, n_shuffles + 1):
            ds.shuffle()
            assert jnp.all(ds[0][0] == ds[0][1]), f"check `x == y` after {n} shuffles"

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_shuffle_rearranges_ds(
            do_use_jit,
            make_range_dataset,
    ):
        with jax.disable_jit(disable=not do_use_jit):
            n_shuffles, n_items = 5, 128
            ds, data = make_range_dataset(n_items=n_items, seed=SEED)
            for n in range(1, n_shuffles + 1):
                prev = ds[0][0]
                ds.shuffle()
                assert not jnp.all(ds[0][0] == prev), f"Shuffle #{n}."

    @staticmethod
    def test_same_seed_gets_same_shuffles():
        ...

    @staticmethod
    def test_raw_data_is_not_shuffled(
            make_range_dataset,
    ):
        n_items = 128
        ds, data = make_range_dataset(n_items=n_items, seed=SEED)
        assert jnp.all(ds[0][0] == data)


class TestBatchRetrival:
    @staticmethod
    def test_():
        ...


if __name__ == '__main__':
    pytest.main()
