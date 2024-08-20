r"""General utility methods.

This module contains utility methods used by other modules.

Classes
-------
- `JaxErrors`: Handles errors for jitted functions.

Functions
---------
- `apply_to_multilayer_data`: A method for applying a transformation over a multilayered collection of data.
- `check_fit_input`: Validates the input of fit functions in a jit compatible way
according to the SKlearn specifications for estimators.
- `check_predict_input`: Validates the input of predict functions in a jit compatible way
according to the SKlearn specifications for estimators.

"""
import jax
import pytest
import chex
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.tree_util import register_pytree_node_class
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


# @jax.jit
def check_fit_input(
        x: ArrayLike,
        y: ArrayLike,
) -> None:
    r"""Validates the input of fit functions according to the SKlearn specifications for estimators.

    The validation is implemented in a jit-compatible way and in case of failure a corresponding error is raised.

    :param x: The data used for fitting.
    :param y: The labels used for fitting.
    :raises ValueError: If:
     - `x` or `y` are complex.
     - `x` is empty.
     - `x` or `y` contain Nans or infs.
    """
    # TODO: The current implementation works fine since `fit` is not jitted (try-except works).
    #  Change to the same format as in `check_predict_input` for jit-proofing.
    try:
        chex.assert_equal(
            jnp.iscomplexobj(x),
            False,
            custom_message="Complex data not supported.",
        )
        chex.assert_equal(
            jnp.iscomplexobj(y),
            False,
            custom_message="Complex data not supported.",
        )
    except AssertionError as Err:
        raise ValueError("Complex data not supported.")

    try:
        chex.assert_equal(
            x.size > 0,
            True,
            custom_message=f"0 feature(s) (shape={x.shape}) while a minimum of 1 is required."
                           f"Got Got Cannot fit empty data.",
        )
    except AssertionError as Err:
        raise ValueError(
            f"0 feature(s) (shape={x.shape}) while a minimum of 1 is required."
            f"Got Got Cannot fit empty data.",
        )

    try:
        chex.assert_equal(
            jnp.all(jnp.isfinite(x)),
            True,
            custom_message="While fitting, encountered in y: Nans/ inf not supported.",
        )
        chex.assert_equal(
            jnp.all(jnp.isfinite(y)),
            True,
            custom_message="While fitting, encountered in y: Nans/ inf not supported.",
        )
    except AssertionError as Err:
        raise ValueError("Nans/ inf not supported.")


@jax.jit
def check_predict_input(
        x: ArrayLike,
) -> None:
    r"""Validates the input of predict functions according to the SKlearn specifications for estimators.

    The validation is implemented in a jit-compatible way and in case of failure a corresponding error is raised.

    :param x: The data used for the predictions.
    :raises ValueError: If:
     - the input has Nans/ infs.
     - the input is empty.
    """
    # truth = jnp.array(True, dtype=bool)

    jax.lax.cond(
        jnp.all(jnp.isfinite(x)),
        JaxErrors.true_fn,
        lambda: jax.debug.callback(JaxErrors.false_fn_predict_on_non_finite),
    )
    # try:
    #     chex.assert_trees_all_equal(
    #         jnp.all(jnp.isfinite(x)),
    #         truth,
    #         custom_message="While predicting. Nans/ inf not supported.",
    #     )
    #     chex.block_until_chexify_assertions_complete()
    # except AssertionError as Err:
    #     raise ValueError("Nans/ inf not supported.")

    jax.lax.cond(
        x.ndim > 1,
        JaxErrors.true_fn,
        lambda: jax.debug.callback(JaxErrors.false_fn_predict_on_too_low_dim),
    )

    # try:
    #     chex.assert_equal(
    #         x.ndim > 1,
    #         True,
    #         custom_message="Input array must be at least 2D. Reshape your data.",
    #     )
    #     chex.block_until_chexify_assertions_complete()
    # except AssertionError as Err:
    #     raise ValueError("Input array must be at least 2D. Reshape your data.")


@register_pytree_node_class
class JaxErrors:
    r"""This class includes static methods for raising exceptions in jitted methods.

    It is recommended to use these methods with `jax.debug.callback`.
    """

    @staticmethod
    def tree_flatten(
    ) -> Tuple[Optional[Any], Optional[Any]]:
        r"""Specifies how to serialize model into a JAX pytree.

        :return: A tuple with 2 elements:
         - The `children`, containing arrays & pytrees
         - The `aux_data`, containing static and hashable data.
        """
        children = None
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(
            cls,
            aux_data,
            children,
    ) -> "JaxErrors":
        r"""Specifies how to build a model from a JAX pytree.

        :param aux_data: Contains static, hashable data.
        :param children: Contain arrays & pytrees.
        :return:
        """
        return cls()

    @staticmethod
    @jax.jit
    def value_error():
        r"""A method for passing a `ValueError`."""
        return ValueError

    @staticmethod
    @jax.jit
    def type_error():
        r"""A method for passing a `TypeError`."""
        return TypeError

    @staticmethod
    @jax.jit
    def true_fn(*args):
        r"""A method representing no error."""
        ...

    @staticmethod
    @jax.jit
    def false_fn(msg=None):
        r"""A method for a general error."""
        raise ValueError(msg)

    @staticmethod
    @jax.jit
    def false_fn_predict_on_non_finite(*args):
        r"""An error method for `predict` with Nans/ infs in the input."""
        raise ValueError("While predicting. Nans/ inf not supported.")

    @staticmethod
    @jax.jit
    def false_fn_predict_on_too_low_dim(*args):
        r"""An error method for `predict` with a low dimensional input."""
        raise ValueError("Input array must be at least 2D. Reshape your data.")


if __name__ == '__main__':
    pytest.main()
