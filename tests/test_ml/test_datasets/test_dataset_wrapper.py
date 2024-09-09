from contextlib import nullcontext as does_not_raise
import jax
from jax import numpy as jnp
import pathlib
import pytest

from bde.utils import configs as cnfg

SEED = cnfg.General.SEED
TESTED_SEEDS = [0, 24, 42]
DATA_IDX = {
    "x": 0,
    "y": 1,
}


class TestPyTreePacking:
    @staticmethod
    @pytest.fixture(
        scope="class",
        params=[True, False],
        ids=["with_shuffle", "no_shuffle"],
    )
    def default_ds(
            request,
            make_range_dataset,
    ):
        n_items = 128
        ds, data = make_range_dataset(n_items=n_items, seed=SEED)
        if request.param:
            ds = ds.shuffle()
        yield ds

    @staticmethod
    def test_pytree_recreation_with_no_shuffling_is_ordered(
            make_range_dataset,
            recreate_with_pytree,
    ):
        n_items = 128
        ds, data = make_range_dataset(n_items=n_items, seed=SEED)
        ds2 = recreate_with_pytree(ds)
        assert jnp.all(ds2.assignment.reshape(-1) == data.reshape(-1))

    @staticmethod
    @pytest.mark.parametrize("att", [
        "_batch_size",
        "batch_size",
        "_seed",
        "seed",
        "n_items_",
        "size_",
        "items_lim_",
        "was_shuffled_",
    ])
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_aux_elements_are_recreated_properly(
            att,
            do_use_jit,
            recreate_with_pytree,
            default_ds,
    ):
        with jax.disable_jit(disable=not do_use_jit):
            ds_original = default_ds
            ds_recreated = recreate_with_pytree(ds_original)
            assert getattr(ds_original, att) == getattr(ds_recreated, att)

    @staticmethod
    @pytest.mark.parametrize("att", [
        "x",
        "y",
        "split_key",
        "rng_key",
        "assignment",
    ])
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_array_elements_are_recreated_properly(
            att,
            do_use_jit,
            recreate_with_pytree,
            default_ds,
    ):
        n_items = 128
        with jax.disable_jit(disable=not do_use_jit):
            ds_original = default_ds
            ds_recreated = recreate_with_pytree(ds_original)
            assert jnp.all(getattr(ds_original, att) == getattr(ds_recreated, att))


class TestShufflingAndRandomness:
    @staticmethod
    def test_xy_match_after_n_shuffles(
            make_range_dataset,
    ):
        n_shuffles, n_items = 3, 128
        ds, _ = make_range_dataset(n_items=n_items, seed=SEED)
        for n in range(1, n_shuffles + 1):
            ds = ds.shuffle()
            assert jnp.all(ds[0][0] == ds[0][1]), f"check `x == y` after {n} shuffles"

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_shuffle_rearranges_ds(
            do_use_jit,
            make_range_dataset,
    ):
        n_shuffles, n_items = 3, 128
        ds, data = make_range_dataset(n_items=n_items, seed=SEED)
        for n in range(1, n_shuffles + 1):
            before_reshuffle = ds[0][0]
            with jax.disable_jit(disable=not do_use_jit):
                ds = ds.shuffle()
                after_reshuffle = ds[0][0]
                assert not jnp.all(after_reshuffle == before_reshuffle), f"Shuffle #{n}/{n_shuffles}."

    @staticmethod
    @pytest.mark.parametrize("reseed_with", ["equals", "plus_equals"])
    def test_reseed_shuffled_dataset_is_rearranged(
            reseed_with,
            make_range_dataset,
    ):
        n_reseed, n_items = 2, 128
        ds, _ = make_range_dataset(n_items=n_items, seed=SEED)
        ds = ds.shuffle()
        for n in range(1, n_reseed + 1):
            before_reseed = ds[0][0]
            if reseed_with == "equals":
                ds.seed = SEED + n
            elif reseed_with == "plus_equals":
                ds.seed += 1
            else:
                raise ValueError(f"The test was executed with an unsupported value of `reseed_with`: {reseed_with}")
            after_reseed = ds[0][0]
            assert not jnp.all(after_reseed == before_reseed), f"Reseed #{n}/{n_reseed}."

    @staticmethod
    def test_reseed_to_self_when_shuffled_resets_randomness_to_1st_shuffle(
            make_range_dataset,
    ):
        n_shuffles, n_items = 2, 128
        ds, data = make_range_dataset(n_items=n_items, seed=SEED)
        ds = ds.shuffle()
        first_shuffle_before_reseed = ds[0][0]
        for n in range(1, n_shuffles + 1):
            ds.seed = SEED
            after_reseed = ds[0][0]
            assert jnp.all(after_reseed == first_shuffle_before_reseed), f"Shuffle #{n}/{n_shuffles}."
            ds = ds.shuffle()

    @staticmethod
    def test_same_seed_gets_same_shuffles(make_range_dataset):
        n_items = 128
        ds1, _ = make_range_dataset(n_items=n_items, seed=SEED)
        ds2, _ = make_range_dataset(n_items=n_items, seed=SEED)
        ds1, ds2 = ds1.shuffle(), ds2.shuffle()
        assert jnp.all(ds1.assignment == ds2.assignment)

    @staticmethod
    @pytest.mark.parametrize("do_reseed", [False, True])
    def test_raw_data_is_not_shuffled(
            do_reseed,
            make_range_dataset,
    ):
        n_items = 128
        ds, data = make_range_dataset(n_items=n_items, seed=SEED)
        if do_reseed:
            ds.seed = SEED + 1
        assert jnp.all(ds[0][0] == data)


class TestBatching:
    @staticmethod
    @pytest.fixture(
        scope="class",
        params=[True, False],
        ids=["batch_size_changed_after_init", "batch_size_set_on_init_only"],
    )
    def default_ds_n_bs(
            request,
            make_range_dataset,
    ):
        n_items, batch_size = 24, 4
        ds, _ = make_range_dataset(n_items=n_items, batch_size=batch_size)
        if request.param:
            batch_size = 5
            ds.batch_size = batch_size
        yield ds, n_items, batch_size

    @staticmethod
    @pytest.mark.parametrize("checking", [
        "len",
        "size_",
        "batch_size",
        "_batch_size",
        "n_items_",
        "items_lim_",
    ])
    def test_batch_size_calculations(
            checking,
            default_ds_n_bs,
    ):
        ds, n_items, batch_size = default_ds_n_bs
        expected_size = n_items // batch_size
        n_available = batch_size * expected_size

        if checking == "len":
            assert len(ds) == expected_size
            return
        if checking == "size_":
            assert len(ds) == expected_size
            return
        if checking == "batch_size":
            assert ds.batch_size == batch_size
            return
        if checking == "_batch_size":
            assert ds._batch_size == batch_size
            return
        if checking == "n_items_":
            assert ds.n_items_ == n_items
            return
        if checking == "items_lim_":
            assert ds.items_lim_ == n_available
            return


class TestGenEmpty:
    @staticmethod
    @pytest.fixture(scope="class")
    def default_empty_ds(
            request,
            make_range_dataset,
    ):
        n_items, batch_size = 24, 4
        ds, _ = make_range_dataset(n_items=n_items, batch_size=batch_size)
        ds = ds.gen_empty()
        yield ds

    @staticmethod
    @pytest.mark.parametrize("checking", [
        "size_",
        "n_items_",
        "items_lim_",
    ])
    def test_size_related_attributes(
            checking,
            default_empty_ds,
    ):
        assert getattr(default_empty_ds, checking) == 0

    @staticmethod
    def test_len_0(default_empty_ds):
        assert len(default_empty_ds) == 0

    @staticmethod
    @pytest.mark.parametrize("checking", [
        "x",
        "y",
        "assignment",
    ])
    def test_xy_are_empty(
            checking,
            default_empty_ds,
    ):
        assert getattr(default_empty_ds, checking).shape[0] == 0


# class TestIteration:
#     @staticmethod
#     def test_():
#         ...


class TestScannableDatasetForm:
    @staticmethod
    @pytest.mark.xfail(
        condition=False,
        reason="Check that no errors are raised when scanning over with Jax",
        raises=BaseException,
        run=True,
    )
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_can_be_scanned_with_jax(
            do_use_jit,
            make_range_dataset,
    ):
        n_items, batch_size = 128, 4
        ds, _ = make_range_dataset(n_items=n_items, batch_size=batch_size)
        with jax.disable_jit(disable=not do_use_jit):
            jax.lax.scan(
                f=lambda cc, sx: (0, 0),
                init=0,
                xs=ds.get_scannable(),
            )

    @staticmethod
    def test_scanning_gets_right_number_of_iterations(make_range_dataset):
        n_items, batch_size = 128, 4
        ds, _ = make_range_dataset(n_items=n_items, batch_size=batch_size)
        _, scanned = jax.lax.scan(
            f=lambda cc, sx: (cc + 1, 0),
            init=0,
            xs=ds.get_scannable(),
        )
        assert scanned.shape[0] == len(ds)

    @staticmethod
    def test_xy_correspondence(make_range_dataset):
        n_items, batch_size = 128, 4
        ds, _ = make_range_dataset(n_items=n_items, batch_size=batch_size)
        xx, yy = ds.get_scannable()
        assert jnp.allclose(xx, yy)

    @staticmethod
    @pytest.mark.parametrize("data_name", ["x", "y"])
    @pytest.mark.parametrize("tested", [
        "iter_same_as_scan",
        "scan_preserves_data_shape",
    ])
    def test_scanned_data(
            data_name,
            tested,
            make_range_dataset,
    ):
        n_items, batch_size, idx = 128, 4, DATA_IDX[data_name]
        ds, _ = make_range_dataset(n_items=n_items, batch_size=batch_size)
        _, scanned = jax.lax.scan(
            f=lambda cc, sx: (cc + 1, sx[idx]),
            init=0,
            xs=ds.get_scannable(),
        )
        if tested == "iter_same_as_scan":
            iterred = jnp.array([x[idx] for x in ds])
            assert jnp.allclose(scanned, iterred)
            return
        if tested == "scan_preserves_data_shape":
            assert scanned.shape[1:] == ds[0][idx].shape
            return
        raise ValueError(f"An unsupported test name was given: `{tested}`")


if __name__ == '__main__':
    pytest.main()
