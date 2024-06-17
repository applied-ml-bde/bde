import pytest
from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections.abc import Iterable

import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import pathlib

from bde.ml.loss import LogLikelihoodLoss
from bde.utils import configs as cnfg

SEED = cnfg.General.SEED


class TestHSplitPred:
    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("n_dim", [1, 2, 4])
    def test_shapes_match(do_use_jit, n_dim):
        y_true = jnp.array([1]).reshape(tuple([1 for _ in range(n_dim - 1)] + [-1]))
        y_pred = jnp.array([1, 2, 3]).reshape(tuple([1 for _ in range(n_dim - 1)] + [-1]))
        with jax.disable_jit(disable=not do_use_jit):
            mean, sigma = LogLikelihoodLoss()._split_pred(y_true, y_pred)
        assert mean.shape == sigma.shape and mean.shape == y_true.shape

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_shapes_small_std(do_use_jit):
        b_batch, n_features, n_std_features = 2, 3, 1
        y_true = jnp.arange(b_batch * n_features).reshape((b_batch, -1))
        y_pred = jnp.arange(b_batch * (n_features + n_std_features)).reshape((b_batch, -1))
        with jax.disable_jit(disable=not do_use_jit):
            mean, sigma = LogLikelihoodLoss()._split_pred(y_true, y_pred)
        assert mean.shape == sigma.shape

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_shapes_std_equals_mean(do_use_jit):
        b_batch, n_features, n_std_features = 2, 3, 3
        y_true = jnp.arange(b_batch * n_features).reshape((b_batch, -1))
        y_pred = jnp.arange(b_batch * (n_features + n_std_features)).reshape((b_batch, -1))
        with jax.disable_jit(disable=not do_use_jit):
            mean, sigma = LogLikelihoodLoss()._split_pred(y_true, y_pred)
        assert mean.shape == sigma.shape

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_shapes_std_too_large(do_use_jit):
        b_batch, n_features, n_std_features = 2, 3, 6
        y_true = jnp.arange(b_batch * n_features).reshape((b_batch, -1))
        y_pred = jnp.arange(b_batch * (n_features + n_std_features)).reshape((b_batch, -1))
        with jax.disable_jit(disable=not do_use_jit):
            mean, sigma = LogLikelihoodLoss()._split_pred(y_true, y_pred)
        assert mean.shape == sigma.shape

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("n_std_features", [0, 1])
    def test_std_padded_values(do_use_jit, n_std_features):
        b_batch, n_features = 2, 3
        y_true = jnp.arange(b_batch * n_features).reshape((b_batch, -1))
        y_pred = jnp.arange(b_batch * (n_features + n_std_features)).reshape((b_batch, -1))
        with jax.disable_jit(disable=not do_use_jit):
            mean, sigma = LogLikelihoodLoss()._split_pred(y_true, y_pred)
        assert jnp.allclose(sigma[..., n_std_features:], jnp.ones_like(sigma[..., n_std_features:]))

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_std_matches_pred(do_use_jit):
        b_batch, n_features, n_std_features = 2, 3, 1
        y_true = jnp.arange(b_batch * n_features).reshape((b_batch, -1))
        y_pred = jnp.arange(b_batch * (n_features + n_std_features)).reshape((b_batch, -1))
        with jax.disable_jit(disable=not do_use_jit):
            mean, sigma = LogLikelihoodLoss()._split_pred(y_true, y_pred)
        assert jnp.allclose(sigma[..., :n_std_features], y_pred[..., n_features:])

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_mean_matches_pred(do_use_jit):
        b_batch, n_features, n_std_features = 2, 3, 1
        y_true = jnp.arange(b_batch * n_features).reshape((b_batch, -1))
        y_pred = jnp.arange(b_batch * (n_features + n_std_features)).reshape((b_batch, -1))
        with jax.disable_jit(disable=not do_use_jit):
            mean, sigma = LogLikelihoodLoss()._split_pred(y_true, y_pred)
        assert jnp.allclose(mean, y_pred[..., :n_features])


class TestLogLikelihoodLossCall:
    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_no_std_mean_eq_mean_is_0(do_use_jit):
        n_batch, n_features = 2, 3
        key = jax.random.key(seed=SEED)
        key, subkey = jax.random.split(key)
        y_true = jax.random.normal(subkey, (n_batch, n_features))
        with jax.disable_jit(disable=not do_use_jit):
            assert jnp.allclose(
                LogLikelihoodLoss(mean_weight=2)(y_true, y_true),
                jnp.zeros((n_batch, )),
            )

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_no_std_like_mse(do_use_jit):
        n_batch, n_features = 10, 1
        key = jax.random.key(seed=SEED)
        key, subkey = jax.random.split(key)
        y_true = jax.random.normal(subkey, (n_batch, n_features))
        with jax.disable_jit(disable=not do_use_jit):
            assert jnp.allclose(
                LogLikelihoodLoss(mean_weight=2)(y_true, jnp.zeros_like(y_true)),
                y_true.reshape((-1)) ** 2,
            )

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_no_std_and_mean_weight_is_0(do_use_jit):
        n_batch, n_features = 10, 3
        key = jax.random.key(seed=SEED)
        key, subkey = jax.random.split(key)
        y_true = jax.random.normal(subkey, (n_batch, n_features))
        key, subkey = jax.random.split(key)
        y_pred = jax.random.normal(subkey, (n_batch, n_features))
        with jax.disable_jit(disable=not do_use_jit):
            assert jnp.allclose(
                LogLikelihoodLoss(mean_weight=0)(y_true, y_pred),
                jnp.zeros((n_batch,)),
            )

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_log_std_when_mean_weight_is_0(do_use_jit):
        n_batch, n_features = 10, 1
        key = jax.random.key(seed=SEED)
        key, subkey = jax.random.split(key)
        y_true = jax.random.normal(subkey, (n_batch, n_features))

        expected_values = jnp.arange(- 5, n_batch - 5)
        y_pred = jnp.stack([expected_values, jnp.e ** expected_values], axis=1)
        with jax.disable_jit(disable=not do_use_jit):
            assert jnp.allclose(
                LogLikelihoodLoss(mean_weight=0)(y_true, y_pred),
                expected_values,
                atol=1e-4,
            )


if __name__ == '__main__':
    pytest.main()
