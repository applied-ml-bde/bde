import pytest
import flax
from flax import linen as nn
from flax.struct import dataclass, field
from flax.training import train_state
import jax
from jax import numpy as jnp
import optax
import pathlib

import bde.ml
from bde.utils import configs as cnfg


@pytest.mark.parametrize("do_use_jit", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_same_as_training_step(do_use_jit, batch_size):
    n_features = 1
    net = bde.ml.models.FullyConnectedModule(
        n_input_params=n_features,
        n_output_params=n_features,
        layer_sizes=None,
        do_final_activation=False,
    )
    params_base, _ = bde.ml.models.init_dense_model(
        model=net,
        batch_size=batch_size,
        seed=cnfg.General.SEED,
    )
    optimizer = optax.adam(learning_rate=1e-1)
    f_loss = bde.ml.loss.LogLikelihoodLoss()
    rng_key = jax.random.key(seed=cnfg.General.SEED)
    x = jax.random.normal(rng_key, (batch_size, n_features), jnp.float32)

    model_state_raw = train_state.TrainState.create(
        apply_fn=net.apply,
        params=params_base,
        tx=optimizer,
    )
    model_state_tested = train_state.TrainState.create(
        apply_fn=net.apply,
        params=params_base,
        tx=optimizer,
    )

    model_state_raw, _ = bde.ml.training.train_step(
        model_state_raw,
        batch=(x, x),
        f_loss=f_loss,
    )
    dataset = bde.ml.datasets.DatasetWrapper(x=x, y=x, batch_size=batch_size)
    with jax.disable_jit(disable=not do_use_jit):
        model_state_tested, _ = bde.ml.training.jitted_training_over_batch(
            model_state_loss=(model_state_tested, 0.0),
            f_loss=f_loss,
            batches=dataset,
            num_batch=0,
        )
        raw_and_tested_are_the_same = [
            jnp.allclose(x, y) for x, y in zip(
                jax.tree_util.tree_flatten(model_state_tested.params)[0],
                jax.tree_util.tree_flatten(model_state_raw.params)[0],
            )
        ]
        raw_and_tested_are_the_same = jnp.all(jnp.array(raw_and_tested_are_the_same))
        changed_during_training = [
            not jnp.allclose(x, y) for x, y in zip(
                jax.tree_util.tree_flatten(model_state_tested.params)[0],
                jax.tree_util.tree_flatten(params_base)[0],
            )
        ]
        changed_during_training = jnp.all(jnp.array(changed_during_training))
    assert raw_and_tested_are_the_same and changed_during_training


if __name__ == '__main__':
    pytest.main()
