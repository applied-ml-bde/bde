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
possible_reductions = [True, False]


class TestLossMSECall:
    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("reduction", possible_reductions)
    def test_no_std_like_mse(do_use_jit, reduction):
        n_batch, n_features = 10, 1
        key = jax.random.key(seed=SEED)
        key, subkey = jax.random.split(key)
        y_true = jax.random.normal(subkey, (n_batch, n_features))
        with jax.disable_jit(disable=not do_use_jit):
            if reduction:
                assert jnp.allclose(
                    LossMSE(do_reduce=reduction)(y_true, jnp.zeros_like(y_true)),
                    (y_true ** 2).mean(),
                )
            else:
                assert jnp.allclose(
                    LossMSE(do_reduce=reduction)(y_true, jnp.zeros_like(y_true)),
                    y_true.reshape((-1)) ** 2,
                )


if __name__ == '__main__':
    pytest.main()
