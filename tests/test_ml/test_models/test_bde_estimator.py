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
    @pytest.fixture(
        scope="class",
        params=[True, False],
        ids=["After_fitting", "Before_fitting"],
    )
    def default_model(request):
        model_original = BDEEstimator()
        if request.param:
            x = jnp.arange(20).reshape(-1, 1)
            model_original = model_original.fit(x)
        yield model_original

    @staticmethod
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
        "model_",
        "is_fitted_",
        "n_features_in_",

    ])
    def test_reconstruct_on_comparable_objects(
            do_use_jit,
            att,
            recreate_with_pytree,
            default_model,
    ):
        with jax.disable_jit(disable=not do_use_jit):
            model_original = default_model
            model_recreated = recreate_with_pytree(model_original)
            assert getattr(model_original, att) == getattr(model_recreated, att)

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("att", [
        "history_",

    ])
    def test_reconstruct_on_arrays(
            do_use_jit,
            att,
            recreate_with_pytree,
            default_model,
    ):
        with jax.disable_jit(disable=not do_use_jit):
            model_original = default_model
            model_recreated = recreate_with_pytree(model_original)
            if model_original.is_fitted_:
                assert jnp.allclose(getattr(model_original, att), getattr(model_recreated, att))
            assert getattr(model_original, att) == getattr(model_recreated, att)


class TestFit:
    ...


class TestPredict:
    ...


if __name__ == '__main__':
    pytest.main()
