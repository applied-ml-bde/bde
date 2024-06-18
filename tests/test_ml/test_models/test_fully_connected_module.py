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
        n_input_params=2,
        n_output_params=2,
        layer_sizes=n_mids,
        do_final_activation=False,
    )
    params, _ = bde.ml.models.init_dense_model(
        model=net,
        batch_size=1,
        seed=cnfg.General.SEED,
    )
    params = params['params']
    assert len(params) == 1 + (0 if n_mids is None else len(n_mids) if isinstance(n_mids, Sized) else 1)


@pytest.mark.parametrize("n_input", [1, 3])
def test_input_size(n_input):
    net = bde.ml.models.FullyConnectedModule(
        n_input_params=n_input,
        n_output_params=2,
        layer_sizes=None,
        do_final_activation=False,
    )
    assert net.n_input_params == n_input


@pytest.mark.parametrize("n_output", [1, 3])
def test_output_size(n_output):
    net = bde.ml.models.FullyConnectedModule(
        n_input_params=2,
        n_output_params=n_output,
        layer_sizes=None,
        do_final_activation=False,
    )
    params, _ = bde.ml.models.init_dense_model(
        model=net,
        batch_size=1,
        seed=cnfg.General.SEED,
    )
    params = params['params']
    k = [kk for kk in params.keys() if "output" in kk.lower()]
    k = k[-1]
    assert params[k]["bias"].size == n_output


if __name__ == '__main__':
    pytest.main()
