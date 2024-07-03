import pytest
from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections.abc import Iterable
import flax
from flax import linen as nn
from flax.struct import dataclass, field
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import optax
import pathlib

import bde
from bde.utils import configs as cnfg

class TestPredict:
    ...


class TestFit:
    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_training_improvement_when_fitting_identity(do_use_jit):
        xx = jnp.arange(1_000, dtype=float).reshape(-1, 1)
        net = bde.ml.models.FullyConnectedModule(
            n_input_params=1,
            n_output_params=1,
            layer_sizes=None,
            do_final_activation=False,
        )
        est_base = bde.ml.models.FullyConnectedEstimator(
            model=net,
            optimizer=optax.adam(learning_rate=0),
            loss=bde.ml.loss.LogLikelihoodLoss(),
            batch_size=100,
            epochs=0,
            seed=cnfg.General.SEED,
        )
        est = bde.ml.models.FullyConnectedEstimator(
            model=net,
            optimizer=optax.adam(learning_rate=1e-3),
            loss=bde.ml.loss.LogLikelihoodLoss(),
            batch_size=100,
            epochs=1,
            seed=cnfg.General.SEED,
        )
        with jax.disable_jit(disable=not do_use_jit):
            est_base.fit(xx)
            est.fit(xx)
        assert jnp.mean((xx - est.predict(xx)) ** 2) < jnp.mean((xx - est_base.predict(xx)) ** 2)

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_training_when_fitting_identity(do_use_jit):
        xx = jnp.linspace(-32, 32, 128).reshape(-1, 1)
        net = bde.ml.models.FullyConnectedModule(
            n_input_params=1,
            n_output_params=1,
            layer_sizes=None,
            do_final_activation=False,
        )
        est = bde.ml.models.FullyConnectedEstimator(
            model=net,
            optimizer=optax.adam(learning_rate=jnp.sqrt(1e-1) ** 2),
            loss=bde.ml.loss.LogLikelihoodLoss(),
            # loss=optax.losses.l2_loss,
            batch_size=128,
            epochs=24,
            seed=cnfg.General.SEED,
        )
        with jax.disable_jit(disable=not do_use_jit):
            est.fit(xx)
        assert (jnp.allclose(est.params_["params"]["Output"]["bias"], 0.0, atol=jnp.sqrt(1e-1)) and
                jnp.allclose(est.params_["params"]["Output"]["kernel"], 1.0, atol=jnp.sqrt(1e-1))
                )


class TestInit:
    ...


if __name__ == '__main__':
    pytest.main()
