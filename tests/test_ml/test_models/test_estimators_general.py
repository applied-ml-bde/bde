import pathlib
import tempfile

import jax
import pytest
from jax import numpy as jnp
from sklearn.utils.estimator_checks import parametrize_with_checks

from bde.ml.models import BDEEstimator, FullyConnectedEstimator


@parametrize_with_checks(
    [
        FullyConnectedEstimator(),
        BDEEstimator(),
    ]
)
@pytest.mark.parametrize("do_use_jit", [False])
@pytest.mark.timeout(300)
# @pytest.mark.skip(
#     reason=(
#         "This causes the tests to freeze on github (not locally). "
#         "Temp skip until resolved."
#     ),
# )
def test_sklearn_estimator(do_use_jit, estimator, check):
    # NOTE: These tests fail in jitted mode.
    #  Make sure that this is due to the test design, and not our code.
    with jax.disable_jit(disable=not do_use_jit):
        check(estimator)


class TestReconstruct:
    @staticmethod
    @pytest.fixture(
        scope="class",
        params=[FullyConnectedEstimator, BDEEstimator],
        ids=["FCE", "BDE"],
    )
    def my_models(request):
        yield request.param

    @staticmethod
    @pytest.fixture(
        scope="class",
        params=[True, False],
        ids=["After_fitting", "Before_fitting"],
    )
    def default_model(request, my_models):
        model_original = my_models()
        if request.param:
            x = jnp.arange(20, dtype=float).reshape(-1, 1)
            model_original = model_original.fit(x, x)
        with tempfile.TemporaryDirectory() as tmpdir_name:
            path = pathlib.Path(tmpdir_name) / "model_file.pkl"
            model_original.save(path)
            yield model_original, path

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize(
        "tested_reconstruction",
        [
            "PyTree",
            "SaveLoad",
        ],
    )
    @pytest.mark.parametrize(
        "att",
        [
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
            "samples_",
            "model_",
            "is_fitted_",
            "n_features_in_",
            "history_",
        ],
    )
    def test_reconstruct_on_comparable_objects(
        do_use_jit,
        tested_reconstruction,
        att,
        recreate_with_pytree,
        default_model,
    ):
        with jax.disable_jit(disable=not do_use_jit):
            model_original, path = default_model
            if att not in model_original.__dict__:
                # TODO: Make it so `att` is loaded directly from the `__dict__` of
                #  `model_original`.
                pytest.skip(f"No attribute `{att}` in `{model_original}`")

            if tested_reconstruction == "PyTree":
                model_recreated = recreate_with_pytree(model_original)
            elif tested_reconstruction == "SaveLoad":
                model_recreated = model_original.__class__.load(path)
            else:
                raise ValueError(
                    f"Unrecognized `tested_reconstruction`: {tested_reconstruction}",
                )

            if isinstance(getattr(model_original, att), jnp.ndarray):
                assert jnp.allclose(
                    getattr(model_original, att), getattr(model_recreated, att)
                )
                return
            if att in ["params_", "samples_"]:
                tree_of_original = jax.tree.leaves(getattr(model_original, att))
                tree_of_recreated = jax.tree.leaves(getattr(model_recreated, att))
                leaf_comparisons = [
                    jnp.allclose(org, rec)
                    for org, rec in zip(tree_of_original, tree_of_recreated)
                ]
                assert jnp.all(jnp.array(leaf_comparisons))
                return
            assert getattr(model_original, att) == getattr(model_recreated, att)


if __name__ == '__main__':
    pytest.main()
