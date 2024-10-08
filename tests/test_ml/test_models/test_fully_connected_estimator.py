import jax
import optax
import pytest
from jax import numpy as jnp

import bde
from bde.utils import configs as cnfg


class TestInit:
    ...  # fmt: skip


class TestFit:
    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_training_improvement_when_fitting_identity(do_use_jit):
        xx = jnp.arange(1_000, dtype=float).reshape(-1, 1)
        net_kwargs = {
            "n_output_params": 1,
            "n_input_params": 1,
            "layer_sizes": None,
            "do_final_activation": False,
        }
        est_base = bde.ml.models.FullyConnectedEstimator(
            model_class=bde.ml.models.FullyConnectedModule,
            model_kwargs=net_kwargs,
            optimizer_class=optax.adam,
            optimizer_kwargs={"learning_rate": 0},
            loss=bde.ml.loss.LossMSE(),
            batch_size=100,
            epochs=0,
            seed=cnfg.General.SEED,
        )
        est = bde.ml.models.FullyConnectedEstimator(
            model_class=bde.ml.models.FullyConnectedModule,
            model_kwargs=net_kwargs,
            optimizer_class=optax.adam,
            optimizer_kwargs={"learning_rate": 1e-3},
            loss=bde.ml.loss.LossMSE(),
            batch_size=100,
            epochs=1,
            seed=cnfg.General.SEED,
        )
        with jax.disable_jit(disable=not do_use_jit):
            est_base.fit(xx)
            est.fit(xx)
            assert jnp.mean((xx - est.predict(xx)) ** 2) < jnp.mean(
                (xx - est_base.predict(xx)) ** 2
            )

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_training_when_fitting_identity(do_use_jit):
        xx = jnp.linspace(-32, 32, 128).reshape(-1, 1)
        net_kwargs = {
            "n_output_params": 1,
            "n_input_params": 1,
            "layer_sizes": None,
            "do_final_activation": False,
        }
        est = bde.ml.models.FullyConnectedEstimator(
            model_class=bde.ml.models.FullyConnectedModule,
            model_kwargs=net_kwargs,
            optimizer_class=optax.adam,
            optimizer_kwargs={"learning_rate": jnp.sqrt(1e-1) ** 2},
            loss=bde.ml.loss.LossMSE(),
            batch_size=128,
            epochs=24,
            seed=cnfg.General.SEED,
        )
        with jax.disable_jit(disable=not do_use_jit):
            est.fit(xx)
        assert jnp.allclose(
            est.params_["params"]["Output"]["bias"], 0.0, atol=jnp.sqrt(1e-1)
        ) and jnp.allclose(
            est.params_["params"]["Output"]["kernel"], 1.0, atol=jnp.sqrt(1e-1)
        )


class TestPredict:
    ...  # fmt: skip


if __name__ == '__main__':
    pytest.main()
