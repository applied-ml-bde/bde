r"""General utility methods.

This module contains utility methods used by other modules.

Functions
---------
- `apply_to_multilayer_data`: A method for applying a transformation over a multilayered collection of data.

"""

import pytest
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing import Any, Union, Callable, Dict, Tuple, List, Optional


def apply_to_multilayer_data(
        data: Union[Dict, List, Any],
        f: Callable[[Any], Any],
        dtypes: Optional[Union[type, Tuple[type, ...]]] = None,
) -> Union[Dict, List, Any]:
    r"""Apply a transformation over a multilayer collection of data.

    If a dict or a list is given, the function will be applied on the data recursively.
    For other data types, the transformation is applied if data corresponds to `dtypes`.

    :param data: The input data to be transformed.
    If data is a dict or a list, the transformation will be applied to all its members recursively
    until other datatypes are encountered.
    :param f: The transformation to be applied.
    :param dtypes: The datatypes to apply the transformation to.
    If None is given, the transformation is applied on all data types.
    If a tuple of datatypes is given, all datatypes in the tuple will be considered.
    :return:
    """
    if isinstance(data, List):
        for idx, d in enumerate(data):
            data[idx] = apply_to_multilayer_data(d, f, dtypes)
        return data
    if isinstance(data, Dict):
        for k, v in data.items():
            data[k] = apply_to_multilayer_data(v, f, dtypes)
        return data

    if dtypes is None:
        return f(data)
    if isinstance(data, dtypes):
        return f(data)
    return data


def check_fit_input(
        x: ArrayLike,
        y: ArrayLike,
) -> None:
    r"""Runs test on fitting inputs and raises corresponding errors.

    The tests are largely determined by the SKlearn API.

    :param x: The data used for fitting.
    :param y: The labels used for fitting.
    :raises ValueError: If:
     - `x` or `y` are complex.
     - `x` is empty.
     - `x` or `y` contain Nans or infs.
    """
    # TODO: Use jax.jit friendly method instead
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
        raise ValueError("Complex data not supported.")
    if x.size == 0:
        raise ValueError(
            f"0 feature(s) (shape={x.shape}) while a minimum of 1 is required."
            f"Got Got Cannot fit empty data.",
        )
    if not (jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y))):
        raise ValueError("Nans/ inf not supported.")


def check_predict_input(
        x: ArrayLike,
) -> None:
    r"""Runs test on prediction inputs and raises corresponding errors.

    The tests are largely determined by the SKlearn API.

    :param x: The data used for the predictions.
    :raises ValueError: If the input has Nans/ infs:
    """
    # TODO: Use jax.jit friendly method instead
    if not jnp.all(jnp.isfinite(x)):
        raise ValueError("Nans/ inf not supported.")
    if x.ndim == 1:
        raise ValueError("Input array must be at least 2D. Reshape your data.")


if __name__ == '__main__':
    pytest.main()
