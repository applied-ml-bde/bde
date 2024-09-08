import pytest
import jax
from jax import numpy as jnp

import bde.ml
from bde.ml.models import BDEEstimator
from bde.utils import configs as cnfg


class TestInit:
    ...


class TestPyTree:
    @staticmethod
    @pytest.mark.parametrize("do_fit", [True, False])
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("att", [
        "model_class",
        "model_kwargs",
        "optimizer_class",
        "optimizer_kwargs",
        "loss",
        "batch_size",
        "epochs",
        "metrics",
        "validation_size",
        "seed",
        "n_chains",
        "n_init_runs",
        "chain_len",
        "warmup",

        "params_",
        "history_",
        "model_",
        "is_fitted_",
        "n_features_in_",

    ])
    @pytest.mark.skip(reason="In development.")
    def test_reconstruct_before_fit(
            do_fit,
            do_use_jit,
            att,
            recreate_with_pytree,
    ):
        with jax.disable_jit(disable=not do_use_jit):
            model_original = BDEEstimator()
            if do_fit:
                x = jnp.arange(20).reshape(-1, 1)
                model_original = model_original.fit(x)
            model_recreated = recreate_with_pytree(model_original)
            assert getattr(model_original, att) == getattr(model_recreated, att)


class TestTraining:
    ...


class TestPredict:
    ...


if __name__ == '__main__':
    pytest.main()
