import pytest
from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections.abc import Iterable

import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import pathlib

from bde.ml.loss import LossMSE
from bde.utils import configs as cnfg

SEED = cnfg.General.SEED
POSSIBLE_REDUCTIONS = [True, False]


class TestLossMSECall:
    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("reduction", POSSIBLE_REDUCTIONS)
    def test_loss_when_predicting_zeros(do_use_jit, reduction):
        n_batch, n_features = 10, 1
        key = jax.random.key(seed=SEED)
        key, subkey = jax.random.split(key)
        y_true = jax.random.normal(subkey, (n_batch, n_features))
        expected_loss = (y_true ** 2).mean() if reduction else y_true.reshape((-1)) ** 2
        with jax.disable_jit(disable=not do_use_jit):
            f_loss = LossMSE().apply_reduced if reduction else LossMSE()
            assert jnp.allclose(
                f_loss(y_true, jnp.zeros_like(y_true)),
                expected_loss,
            )

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("reduction", POSSIBLE_REDUCTIONS)
    def test_correct_pred_gives_zero_loss(do_use_jit, reduction):
        n_batch, n_features = 10, 1
        key = jax.random.key(seed=SEED)
        key, subkey = jax.random.split(key)
        y_true = jax.random.normal(subkey, (n_batch, n_features))
        expected_loss = jnp.zeros((1,)) if reduction else jnp.zeros((n_batch,))
        with jax.disable_jit(disable=not do_use_jit):
            f_loss = LossMSE().apply_reduced if reduction else LossMSE()
            assert jnp.allclose(
                f_loss(y_true, y_true),
                expected_loss,
            )


if __name__ == '__main__':
    pytest.main()
