import jax.tree
import pytest
from jax import numpy as jnp

from bde.ml.models import BDEEstimator
from bde.utils.utils import get_n_devices


class TestInit:
    ...  # fmt: skip


class TestFit:
    ...  # fmt: skip


class TestPredict:
    ...  # fmt: skip


class TestParallelization:
    @staticmethod
    @pytest.fixture(
        scope="class",
        params=[-1, 1, None, 2],
        ids=["-1_devices", "1_device", "all_devices", "2_devices"],
    )
    def trained_model(request):
        n_devices = int(get_n_devices())
        # TODO:
        #  - If we have only 1 device, the `n=2` test should be skipped.
        #  - Make sure that testing all devices isn't too expensive.
        #    Testing just `n=2` might be enough.
        n_devices_pass = (
            n_devices if request.param is None else min(int(request.param), n_devices)
        )
        n_devices = n_devices_pass if n_devices_pass > 0 else n_devices
        n_chains = n_devices + 1
        if n_devices == 1:
            warmup, chain_len = 3, 5
        else:
            warmup = 5 if n_devices == 2 else 2
            chain_len = 6 if n_devices == 2 else (5 if n_devices == 3 else 3)
        epochs = warmup

        model_original = BDEEstimator(
            chain_len=chain_len,
            n_chains=n_chains,
            warmup=warmup,
            epochs=epochs,
        )
        x = jnp.arange(20, dtype=float).reshape(-1, 1)
        model_original = model_original.fit(x, x, n_devices=n_devices)
        size_params = n_devices, n_chains, chain_len, warmup, epochs
        yield model_original, size_params

    @staticmethod
    def test_samples_shape(trained_model):
        model, (n_devices, n_chains, chain_len, warmup, epochs) = trained_model
        exp_size = chain_len * n_chains
        res = [x.shape[0] == exp_size for x in jax.tree.flatten(model.samples_)[0]]
        assert jnp.all(jnp.array(res))

    @staticmethod
    def test_params_shape(trained_model):
        model, (n_devices, n_chains, chain_len, warmup, epochs) = trained_model
        exp_size = n_chains
        res = [x.shape[0] == exp_size for x in jax.tree.flatten(model.params_)[0]]
        assert jnp.all(jnp.array(res))


if __name__ == '__main__':
    pytest.main()
