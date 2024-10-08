import jax
import pytest
from jax import numpy as jnp

import bde.ml


class TestHistory:
    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    def test_history_is_2d(
        do_use_jit,
        gen_training_items,
        generate_model_state,
    ):
        batch_size = 4
        f_loss, train, valid, metrics = gen_training_items(
            n_items=24,
            batch_size=batch_size,
            do_validation=False,
            do_metrics=False,
        )
        with jax.disable_jit(disable=not do_use_jit):
            _, history = bde.ml.training.jitted_training(
                model_state=generate_model_state(batch_size=batch_size),
                train=train,
                valid=valid,
                f_loss=f_loss,
                metrics=metrics,
                epochs=jnp.arange(1),
            )
        assert history.ndim == 2

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize(
        "do_validation", [False]
    )  # TODO: Implement `True` once feature is implemented
    @pytest.mark.parametrize(
        "do_metrics", [False]
    )  # TODO: Implement `True` once feature is implemented
    @pytest.mark.parametrize("num_train_batches", [1, 4])
    def test_1st_dim_size(
        do_use_jit,
        do_validation,
        do_metrics,
        num_train_batches,
        gen_training_items,
        generate_model_state,
    ):
        train_size = 24
        batch_size = train_size // num_train_batches
        f_loss, train, valid, metrics = gen_training_items(
            n_items=train_size,
            batch_size=batch_size,
            do_validation=do_validation,
            do_metrics=do_metrics,
        )
        with jax.disable_jit(disable=not do_use_jit):
            _, history = bde.ml.training.jitted_training(
                model_state=generate_model_state(batch_size=batch_size),
                train=train,
                valid=valid,
                f_loss=f_loss,
                metrics=metrics,
                epochs=jnp.arange(1),
            )
        assert history.shape[0] == (1 + metrics.shape[0]) * (2 if do_validation else 1)

    @staticmethod
    @pytest.mark.parametrize("do_use_jit", [True, False])
    @pytest.mark.parametrize("num_epochs", [1, 4])
    def test_2nd_dim_is_num_of_epochs(
        do_use_jit,
        num_epochs,
        gen_training_items,
        generate_model_state,
    ):
        train_size = 24
        batch_size = train_size
        f_loss, train, valid, metrics = gen_training_items(
            n_items=train_size,
            batch_size=batch_size,
            do_validation=False,
            do_metrics=False,
        )
        with jax.disable_jit(disable=not do_use_jit):
            _, history = bde.ml.training.jitted_training(
                model_state=generate_model_state(batch_size=batch_size),
                train=train,
                valid=valid,
                f_loss=f_loss,
                metrics=metrics,
                epochs=jnp.arange(num_epochs),
            )
        assert history.shape[1] == num_epochs


@pytest.mark.parametrize("do_use_jit", [True, False])
@pytest.mark.parametrize("do_validation", [False, True])
@pytest.mark.parametrize("do_metrics", [False])  # TODO: Implement
def test_1_epoch_with_batch_that_covers_whole_dataset_is_the_same_as_training_step(
    do_use_jit,
    do_validation,
    do_metrics,
    gen_training_items,
    generate_model_state,
):
    n_features, n_items = 1, 128
    batch_size = n_items
    f_loss, train, valid, metrics = gen_training_items(
        n_items=n_items,
        batch_size=batch_size,
        do_validation=do_validation,
        do_metrics=do_metrics,
    )
    model_state_raw = generate_model_state(
        n_input_params=n_features,
        n_output_params=n_features,
        batch_size=batch_size,
    )
    model_state_tested = generate_model_state(
        n_input_params=n_features,
        n_output_params=n_features,
        batch_size=batch_size,
    )
    with jax.disable_jit(disable=not do_use_jit):
        model_state_raw, _ = bde.ml.training.train_step(
            model_state_raw,
            batch=train[0],
            f_loss=f_loss,
        )
        tested_params, _ = bde.ml.training.jitted_training(
            model_state=model_state_tested,
            f_loss=f_loss,
            train=train,
            valid=valid,
            metrics=metrics,
            epochs=jnp.arange(1),
        )
        assert model_state_raw.params == tested_params.params


if __name__ == '__main__':
    pytest.main()
