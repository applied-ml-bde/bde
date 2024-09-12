r"""Training Utilities for Bayesian Neural Networks.

This module provides functionality for training Bayesian Neural Networks within the
Bayesian Deep Ensembles (BDE) framework.

Functions
---------
- `train_step`: Executes a single optimization step for the neural network.
- `jitted_training`: Fits a model over data for a parameters-set.
- `jitted_training_epoch`: Performs 1 training epoch for model training
    (parameter optimization + metrics evaluation + validation).
"""

import pathlib
from typing import Tuple

import jax

# import bde
import pytest
from flax.training.train_state import TrainState
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

import bde.ml
from bde.ml.datasets import BasicDataset
from bde.ml.loss import Loss


@jax.jit
def train_step(
    state: TrainState,
    batch: tuple[ArrayLike, ArrayLike],
    f_loss: Loss,
) -> Tuple[TrainState, float]:
    r"""Perform an optimization step for the network.

    This function updates the model parameters by performing a single
    optimization step using the provided loss function.

    Parameters
    ----------
    state
        The training-state of the network.
    batch
        Input data-points for the training set, containing 2 items:
            - A set of training data-points.
            - The corresponding labels.
    f_loss
        The loss function used while training. Should have the following signature:
            (y_true, y_pred)

    Returns
    -------
    Tuple[TrainState, float]
        Updated state of the network and the loss.
    """
    grad_fn = jax.value_and_grad(
        bde.ml.loss.flax_training_loss_wrapper_regression(f_loss=f_loss),
        argnums=1,  # Parameters are second argument of the function
        has_aux=False,  # Function has additional outputs, here accuracy
    )
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


# @jax.jit
# def jitted_training_over_multiple_models(
#         model: nn.Module,
#         params,
#         optimizer,
#         epochs: int,
#         f_loss: Loss,
#         metrics: List,
#         train: BasicDataset,
#         valid: BasicDataset,
#         history: Array,
# ):
#     r"""Train several model on different parameter sets in parallel.
#
#     Parameters
#     ----------
#     TODO: Complete
#
#     Return
#     ------
#     TODO: Complete
#     """
#     return jax.pmap(
#         jitted_training,
#         axis_name='params',
#     )(
#         model=model,
#         params=params,
#         optimizer=optimizer,
#         epochs=epochs,
#         f_loss=f_loss,
#         metrics=metrics,
#         train=train,
#         valid=valid,
#         history=history,
#     )


@jax.jit
def jitted_training(
    # model_class: Type[nn.Module],
    # model_kwargs: Dict[str, Any],
    model_state: TrainState,
    # params,
    # optimizer_class: Type[optax._src.base.GradientTransformation],
    # optimizer_kwargs: Dict[str, Any],
    epochs: Array,
    f_loss: Loss,
    metrics: Array,
    train: BasicDataset,
    valid: BasicDataset,
) -> Tuple[TrainState, Array]:
    r"""Train a model on a single parameters set.

    A jitted training loop for a model using a single parameter set.

    Parameters
    ----------
    model_state
        A class containing the model architecture + training parameters 6 optimizer.
    epochs
        An array with the indices of the training epochs.
    f_loss
        A class implementing the optimized loss function.
    metrics
        An array of metric classes.
    train
        The training dataset.
    valid
        The validation dataset.

    Returns
    -------
    Tuple[TrainState, Array]
        Updated training state and an array describing the metrics over the
        training epochs.
    """
    # model_state = train_state.TrainState.create(
    #     # apply_fn=model.apply,
    #     apply_fn=model_class(**model_kwargs).apply,
    #     params=params,
    #     tx=optimizer_class(**optimizer_kwargs),
    # )
    (model_state, train, valid), history = jax.lax.scan(
        f=lambda ms_train_val, sx: jitted_training_epoch(
            model_state=ms_train_val[0],
            train=ms_train_val[1],
            valid=ms_train_val[2],
            f_loss=f_loss,
            metrics=metrics,
        ),
        init=(model_state, train, valid),
        xs=epochs,
    )
    return model_state, history.T


@jax.jit
def jitted_training_epoch(
    model_state: TrainState,
    train: BasicDataset,
    valid: BasicDataset,
    f_loss: Loss,
    metrics: Array,
) -> Tuple[Tuple[TrainState, BasicDataset, BasicDataset], Array]:
    r"""Train a model for 1 epoch.

    A jitted training loop for a model over a single epoch.
    Performs training, metrics evaluation and validation.

    Parameters
    ----------
    model_state
        A class containing the model architecture + training parameters 6 optimizer.
    train
        The training dataset.
    valid
        The validation dataset.
    f_loss
        A class implementing the optimized loss function.
    metrics
        An array of metric classes.

    Returns
    -------
    Tuple[Tuple[TrainState, BasicDataset, BasicDataset], Array]
        2 items are returned:
        - The first item is a triplet containing:
            - Updated model state.
            - Updated training dataset (updates shuffling).
            - Updated validation dataset (updates shuffling).
        - The 2nd item is a 1D-array describing the evaluation of all metrics over this epoch.
    """  # noqa: E501
    history = jnp.array(
        [], dtype=jnp.float32
    )  # An empty 1D-array to store the history for current epoch.

    train = train.shuffle()
    model_state, loss = jax.lax.scan(
        f=lambda cc, sx: train_step(
            cc,
            batch=sx,
            f_loss=f_loss,
        ),
        init=model_state,
        xs=train.get_scannable(),
    )
    history = jnp.hstack([history, jnp.array(loss.mean())])

    # n_metrics = len(metrics)
    # history = jax.lax.fori_loop(
    #     lower=0,
    #     upper=n_metrics,
    #     body_fun=lambda i, val: jitted_evaluation_for_a_metric(
    #         model_state=model_state,
    #         history=val,
    #         metrics=metrics,
    #         idx_metric=i,
    #         idx_history=i + 1,
    #         idx_epoch=num_epoch,
    #         batches=train,
    #     ),
    #     init_val=history,
    # )

    valid = valid.shuffle()
    # ADD:
    #  - Validation step support.
    #  - Early stopping support.
    #  - General callbacks support (stretch goal).
    return (model_state, train, valid), history


@jax.jit
def jitted_evaluation_for_a_metric(
    model_state: TrainState,
    batches: BasicDataset,
    metrics,
    history: Array,
    idx_metric: int,
    idx_history: int,
    idx_epoch: int,
):
    r"""Evaluate a training epoch for 1 metric.

    Parameters
    ----------
    TODO: Complete

    Returns
    -------
    TODO: Complete
    """
    metric = metrics[idx_metric]
    m_val, n_batches = 0.0, len(batches)
    m_val = jax.lax.fori_loop(
        lower=0,
        upper=n_batches,
        body_fun=lambda i, val: jitted_evaluation_over_batch(
            model_state=model_state,
            batches=batches,
            f_eval=metric,
            num_batch=i,
            m_val=val,
        ),
        init_val=m_val,
    )
    history = history.at[idx_history, idx_epoch].set(m_val / n_batches)
    return history


@jax.jit
def jitted_evaluation_over_batch(
    model_state: TrainState,
    batches: BasicDataset,
    f_eval,
    num_batch: int,
    m_val: float,
) -> float:
    r"""Perform intermediate evaluation over a metric for 1 batch of data.

    Parameters
    ----------
    TODO: Complete

    Returns
    -------
    TODO: Complete
    """
    x, y_true = batches[num_batch]
    y_pred = model_state.apply_fn(model_state.params, x)
    return m_val + f_eval(y_true, y_pred)


if __name__ == "__main__":
    tests_path = (
        pathlib.Path(__file__).parent.parent / "test" / "test_ml" / "test_training"
    )
    pytest.main([str(tests_path)])
