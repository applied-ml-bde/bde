import pytest
from abc import ABC, abstractmethod
from typing import Any, Union, Optional
from collections.abc import Iterable, Sized
import flax
from flax import linen as nn
from flax.struct import dataclass, field
import jax
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import pathlib

import bde.ml.models
from bde.utils import configs as cnfg


@pytest.mark.parametrize("n_mids", [None, 1, (1, 1)])
def test_num_of_layers(n_mids):
    net = bde.ml.models.FullyConnectedModule(
        n_output_params=2,
        n_input_params=2,
        layer_sizes=n_mids,
        do_final_activation=False,
    )
    params, _ = bde.ml.models.init_dense_model(
        model=net,
        batch_size=1,
        n_features=None,
        seed=cnfg.General.SEED,
    )
    params = params['params']
    assert len(params) == 1 + (0 if n_mids is None else len(n_mids) if isinstance(n_mids, Sized) else 1)


@pytest.mark.parametrize("n_input", [1, 3])
def test_input_size(n_input):
    net = bde.ml.models.FullyConnectedModule(
        n_output_params=2,
        n_input_params=n_input,
        layer_sizes=None,
        do_final_activation=False,
    )
    assert net.n_input_params == n_input


@pytest.mark.parametrize("n_output", [7, 3])
def test_output_size(n_output):
    net = bde.ml.models.FullyConnectedModule(
        n_output_params=n_output,
        n_input_params=2,
        layer_sizes=None,
        do_final_activation=False,
    )
    params, _ = bde.ml.models.init_dense_model(
        model=net,
        batch_size=1,
        n_features=None,
        seed=cnfg.General.SEED,
    )
    params = params['params']
    k = [kk for kk in params.keys() if "output" in kk.lower()]
    k = k[-1]
    assert params[k]["bias"].size == n_output


@pytest.mark.parametrize("layer_size", [5, 3])
def test_layers_size_with_int_layers(layer_size):
    net = bde.ml.models.FullyConnectedModule(
        n_output_params=2,
        n_input_params=2,
        layer_sizes=layer_size,
        do_final_activation=False,
    )
    params, _ = bde.ml.models.init_dense_model(
        model=net,
        batch_size=1,
        n_features=None,
        seed=cnfg.General.SEED,
    )
    params = params['params']
    k = [kk for kk in params.keys() if "dense" in kk.lower()]
    k = k[-1]
    assert params[k]["bias"].size == layer_size


def test_layers_sizes_with_a_list_of_layer_sizes():
    layer_sizes = [5, 3]
    net = bde.ml.models.FullyConnectedModule(
        n_output_params=2,
        n_input_params=2,
        layer_sizes=layer_sizes,
        do_final_activation=False,
    )
    params, _ = bde.ml.models.init_dense_model(
        model=net,
        batch_size=1,
        n_features=None,
        seed=cnfg.General.SEED,
    )
    params = params['params']
    k = [kk for kk in params.keys() if "dense" in kk.lower()]
    k0 = [kk for kk in k if "0" in kk][0]
    k1 = [kk for kk in k if "1" in kk][0]
    assert params[k0]["bias"].size == layer_sizes[0] and params[k1]["bias"].size == layer_sizes[1]


@pytest.mark.parametrize("n_batch", [1, 4])
@pytest.mark.parametrize("n_output", [7, 3])
def test_output_shape(n_batch, n_output):
    net = bde.ml.models.FullyConnectedModule(
        n_output_params=n_output,
        n_input_params=5,
        layer_sizes=None,
        do_final_activation=False,
    )
    params, inp = bde.ml.models.init_dense_model(
        model=net,
        batch_size=n_batch,
        n_features=None,
        seed=cnfg.General.SEED,
    )
    res = net.apply(params, inp)
    assert res.shape == (n_batch, n_output)


@pytest.mark.parametrize("n_batch", [1, 4])
@pytest.mark.parametrize("n_size", [1, 3])
@pytest.mark.parametrize("n_layers", [0, 1, 2])
def test_application_as_unit_network(n_batch, n_size, n_layers):
    net = bde.ml.models.FullyConnectedModule(
        n_output_params=n_size,
        n_input_params=n_size,
        layer_sizes=None if n_layers < 1 else ([n_size] * n_layers),
        do_final_activation=False,
    )
    params, inp = bde.ml.models.init_dense_model(
        model=net,
        batch_size=n_batch,
        n_features=None,
        seed=cnfg.General.SEED,
    )
    for k, v in params["params"].items():
        params["params"][k]["kernel"] = jnp.eye(n_size)
        params["params"][k]["bias"] = jnp.zeros_like(params["params"][k]["bias"])
    inp = jnp.abs(inp)  # Intermediate layers use ReLU.
    assert jnp.allclose(inp, net.apply(params, inp))


if __name__ == '__main__':
    pytest.main()
