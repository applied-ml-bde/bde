import pytest

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

import bde.ml
from bde.ml.datasets import DatasetWrapper
from bde.ml.models import FullyConnectedModule
from bde.utils import configs as cnfg


@pytest.fixture(scope="session")
def make_range_dataset():
    def func(
            n_items,
            seed=cnfg.General.SEED,
            batch_size=None,
    ):
        data = jnp.arange(n_items, dtype=int).reshape(-1, 1)
        batch_size = n_items if batch_size is None else batch_size
        return DatasetWrapper(x=data, y=data, batch_size=batch_size, seed=seed), data
    return func


@pytest.fixture(scope="session")
def generate_model_state():
    def func(
            n_input_params=1,
            n_output_params=1,
            batch_size=1,
            layer_sizes=None,
            seed=cnfg.General.SEED,
            **kwargs,
    ):
        model = FullyConnectedModule(
            n_input_params=n_input_params,
            n_output_params=n_output_params,
            layer_sizes=layer_sizes,
        )
        params, _ = bde.ml.models.init_dense_model(
            model=model,
            batch_size=batch_size,
            n_features=n_input_params,
            seed=seed,
        )
        model_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(learning_rate=1e-3),
        )
        return model_state
    return func


@pytest.fixture(scope="session")
def gen_training_items(make_range_dataset):
    def func(
            n_items,
            batch_size=None,
            seed=cnfg.General.SEED,
            do_validation: bool = False,
            do_metrics: bool = False,
    ):
        f_loss = bde.ml.loss.LossMSE()
        train = make_range_dataset(n_items, batch_size=batch_size, seed=seed)[0]
        valid = train if do_validation else make_range_dataset(0, batch_size=batch_size, seed=seed)[0]
        metrics = None if do_metrics else jnp.array([])  # ADD: Once metrics are implemented
        return f_loss, train, valid, metrics
    return func


@pytest.fixture(scope="session")
def recreate_with_pytree():
    def func(
            src,
    ):
        return src.__class__.tree_unflatten(*src.tree_flatten()[::-1])
    return func


if __name__ == '__main__':
    pytest.main()
