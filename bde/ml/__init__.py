r"""Machine Learning Module for Bayesian Deep Ensembles (BDE).

The `bde.ml` module provides the core machine learning components required for
building and training Bayesian Neural Networks within the Bayesian Deep Ensembles (BDE) framework.

This module includes submodules for defining loss functions, neural network models,
and training procedures, enabling flexible and robust implementation of BDE models.

Submodules
----------
- `datasets`: Handles data and dataset management.
- `loss`: Contains loss functions implementations and loss function related utilities.
- `models`: Defines the neural network architectures supported by the BDE framework.
- `training`: Implements the training algorithms and routines used for model optimization.

Example Usage
-------------
# TODO: Provide examples

    >>> # TODO: Provide an example
    >>>
    >>>
    >>>
"""  # noqa: E501

from bde.ml import (
    datasets,
    loss,
    models,
    training,
)

__all__ = [
    "loss",
    "models",
    "training",
    "datasets",
]
