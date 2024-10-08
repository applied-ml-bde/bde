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

"""  # noqa: E501

from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.typing import ArrayLike


def apply_to_multilayer_data(
    data: Union[Dict, List, Any],
    f: Callable[[Any], Any],
    dtypes: Optional[Union[type, Tuple[type, ...]]] = None,
) -> Union[Dict, List, Any]:
    r"""Apply a transformation over a multilayer collection of data.

    If a dict or a list is given, the function will be applied on the data recursively.
    For other data types, the transformation is applied if data corresponds
    to `dtypes`.

    Parameters
    ----------
    data
        The input data to be transformed.
        If data is a dict or a list, the transformation will be applied to all its
        members recursively until other datatypes are encountered.
    f
        The transformation to be applied.
    dtypes
        The datatypes to apply the transformation to.
         - If None is given, the transformation is applied on all data types.
         - If a tuple of datatypes is given,
           all datatypes in the tuple will be considered.

    Returns
    -------
    Union[Dict, List, Any]
        Transformed `data` with the same structure as the input.
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


@jax.jit  # TODO: Test
def get_n_devices() -> int:
    r"""Get the max number of computational devices available to jax."""
    return len(jax.devices())


@partial(jax.jit, static_argnums=1)
def pad_first_axis(
    pytree,
    pad_width,
):
    r"""Pad the leading axis of an array by a specified amount."""

    @jax.jit
    def f_pad(x):
        pad_size = [(0, pad_width)] + [(0, 0)] * (x.ndim - 1)
        return jnp.pad(x, pad_size)

    return jax.tree.map(f_pad, pytree)


@partial(jax.jit, static_argnums=[1, 2, 3])
def prep_for_pmap(
    pytree,
    pad_size,
    batch_size,
    n_devices,
):
    r"""Prepare pytree data for parallelization.

    Pads data and reshapes to prepare for parallelization.
    """
    pytree = pad_first_axis(pytree, pad_size)

    @jax.jit
    def reshape_fn(x):
        new_shape = (n_devices, batch_size) + x.shape[1:]
        return x.reshape(new_shape)

    pytree = jax.tree.map(reshape_fn, pytree)
    return pytree


# @partial(jax.jit, static_argnums=0)
def post_pmap(
    mask,
    *pytree,
):
    r"""Undo changes made to PyTree for parallelization."""
    pytree, mask = jax.tree.map(
        f=jnp.concatenate,
        tree=[pytree, mask],
    )

    # @jax.jit
    def filter_fn(leaf):
        return leaf[mask.astype(bool)]

    pytree = jax.tree.map(filter_fn, pytree)
    return pytree


@partial(jax.jit, static_argnums=[1])
def cond_with_const_f(
    cond: bool,
    f_true: Callable,
    false_like: ArrayLike,
    *args,
):
    r"""Jit compatible conditioning.

    A wrapper for `jax.lax.cond`.
    Execute a function only if the condition is true.
    If false, a const zero-valued tree is returned.
    """

    @jax.jit
    def f_false(*_):
        return jax.tree.map(jnp.zeros_like, false_like)

    return jax.lax.cond(cond, f_true, f_false, *args)


def pv_map(
    fun,
    in_axes,
    static_broadcasted_argnums,
):
    r"""Parallelizes data which is larger than the number of available devices.

    The data must be prepared in such a way that
    the leading axis is split among the available devices
    and the 2nd axis is vectorized.
    """
    fun = jax.vmap(
        fun=fun,
        in_axes=in_axes,
    )
    fun = jax.pmap(
        fun=fun,
        in_axes=in_axes,
        static_broadcasted_argnums=static_broadcasted_argnums,
    )
    return fun


@jax.jit
def eval_parallelization_params_jitted(
    n_items: int,
    n_devices: int,
) -> Tuple[Array, Array, Array]:
    r"""Calculate size parameters for parallelization over arrays.

    Calculate the jittable parameters for parallelization.

    Parameters
    ----------
    n_items
        Number of items needed to be evaluated in parallel.
    n_devices
        Number of device available for parallel evaluation.
        If `-1`, all possible devices will be used.
        If the value is greater than the number of available devices, the number of
        available devices will be used.

    Returns
    -------
    Tuple[int, int, int]
        A tuple with:
        - The number of devices (adjusted based on availability and selection).
        - Number of items evaluated on each device.
        - The number of items needed to be padded.
    """

    @jax.jit
    def f_true():
        return n_devices

    n_devices = jax.lax.cond(
        n_devices > 0,
        f_true,
        get_n_devices,
    )
    n_devices = jnp.min(jnp.array([n_devices, n_items])).astype(int)
    pad_size = (n_devices - (n_items % n_devices)) % n_devices
    parallel_batch_size = (n_items + pad_size) // n_devices
    return n_devices, parallel_batch_size, pad_size


def eval_parallelization_params(
    n_items: int,
    n_devices: int,
) -> Tuple[int, int, int, Array]:
    r"""Calculate size parameters for parallelization over arrays.

    Parameters
    ----------
    n_items
        Number of items needed to be evaluated in parallel.
    n_devices
        Number of device available for parallel evaluation.
        If `-1`, all possible devices will be used.
        If the value is greater than the number of available devices, the number of
        available devices will be used.

    Returns
    -------
    Tuple[int, int, int, Array]
        A tuple with:
        - The number of devices (adjusted based on availability and selection).
        - Number of items evaluated on each device.
        - The number of items needed to be padded.
        - A 2D mask indicating which items are used for padding.
    """
    n_devices, parallel_batch_size, pad_size = eval_parallelization_params_jitted(
        n_items,
        n_devices,
    )
    pmask = jnp.ones([n_items])
    pmask = jnp.pad(pmask, [(0, pad_size)])
    pmask = pmask.reshape(n_devices, -1)
    return n_devices, parallel_batch_size, pad_size, pmask


# @jax.jit
def check_fit_input(
    x: ArrayLike,
    y: ArrayLike,
) -> None:
    r"""Validate the input of `fit` functions according to the SKlearn specifications for estimators.

    The validation is implemented in a jit-compatible way and in case of failure a corresponding error is raised.

    Parameters
    ----------
    x
        The data used for fitting.
    y
        The labels used for fitting.

    Raises
    ------
    ValueError
        If:
         - `x` or `y` are complex.
         - `x` is empty.
         - `x` or `y` contain Nans or infs.
    """  # noqa: E501
    jax.lax.cond(
        jnp.iscomplexobj(x) or jnp.iscomplexobj(y),
        lambda: jax.debug.callback(JaxErrors.raise_error_for_fit_on_complex_data_error),
        JaxErrors.raise_no_error,
    )
    jax.lax.cond(
        x.size > 0,
        JaxErrors.raise_no_error,
        lambda xx: jax.debug.callback(JaxErrors.raise_error_for_fit_on_empty_data, xx),
        x,
    )
    jax.lax.cond(
        jnp.all(jnp.isfinite(x)) and jnp.all(jnp.isfinite(y)),
        JaxErrors.raise_no_error,
        lambda: jax.debug.callback(JaxErrors.raise_error_for_fit_on_nans_or_infs),
    )


@jax.jit
def check_predict_input(
    x: ArrayLike,
    is_fitted: bool = False,
) -> None:
    r"""Validate the input of `predict` functions according to the SKlearn specifications for estimators.

    The validation is implemented in a jit-compatible way and
    in case of failure a corresponding error is raised.

    Parameters
    ----------
    x
        The data used for the predictions.
    is_fitted
        The `is_fitted_` flag of the estimator.

    Raises
    ------
    ValueError
        If:
         - the input has Nans/ infs.
         - the input is empty.

    AssertionError
        If:
         - The estimator is not fitted.
    """  # noqa: E501
    jax.lax.cond(
        jnp.all(jnp.isfinite(x)),
        JaxErrors.raise_no_error,
        lambda: jax.debug.callback(JaxErrors.raise_error_for_predict_on_non_finite),
    )
    jax.lax.cond(
        x.ndim > 1,
        JaxErrors.raise_no_error,
        lambda: jax.debug.callback(JaxErrors.raise_error_for_predict_on_too_low_dim),
    )
    jax.lax.cond(
        is_fitted,
        JaxErrors.raise_no_error,
        lambda: jax.debug.callback(JaxErrors.raise_error_non_fitted),
    )


def check_fit_input_chex(
    x: ArrayLike,
    y: ArrayLike,
) -> None:
    r"""Validate the input of `fit` functions according to the SKlearn specifications for estimators.

    The validation is implemented in a jit-compatible way and
    in case of failure a corresponding error is raised.

    Parameters
    ----------
    x
        The data used for fitting.
    y
        The labels used for fitting.

    Raises
    ------
    ValueError:
        If:
         - `x` or `y` are complex.
         - `x` is empty.
         - `x` or `y` contain Nans or infs.
    """  # noqa: E501
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
    except AssertionError:
        raise ValueError("Complex data not supported.")

    try:
        chex.assert_equal(
            x.size > 0,
            True,
            custom_message=(
                f"0 feature(s) (shape={x.shape}) while a minimum of 1 is required. "
                f"Got Cannot fit empty data."
            ),
        )
    except AssertionError:
        raise ValueError(
            f"0 feature(s) (shape={x.shape}) while a minimum of 1 is required."
            f"Got Cannot fit empty data.",
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
    except AssertionError:
        raise ValueError("Nans/ inf not supported.")


@jax.jit
def check_predict_input_chex(
    x: ArrayLike,
) -> None:
    r"""Validate the input of `predict` functions according to the SKlearn specifications for estimators.

    The validation is implemented in a jit-compatible way and
    in case of failure a corresponding error is raised.

    Parameters
    ----------
    x
        The data used for the predictions.

    Raises
    ------
    ValueError
        If:
         - the input has Nans/ infs.
         - the input is empty.
    """  # noqa: E501
    truth = jnp.array(True, dtype=bool)
    try:
        chex.assert_trees_all_equal(
            jnp.all(jnp.isfinite(x)),
            truth,
            custom_message="While predicting. Nans/ inf not supported.",
        )
        chex.block_until_chexify_assertions_complete()
    except AssertionError:
        raise ValueError("Nans/ inf not supported.")

    try:
        chex.assert_equal(
            x.ndim > 1,
            True,
            custom_message="Input array must be at least 2D. Reshape your data.",
        )
        chex.block_until_chexify_assertions_complete()
    except AssertionError:
        raise ValueError("Input array must be at least 2D. Reshape your data.")


# @register_pytree_node_class
class JaxErrors:
    r"""This class includes static methods for raising exceptions in jitted methods.

    It is recommended to use these methods with `jax.debug.callback`.
    """

    # @staticmethod
    # def tree_flatten(
    # ) -> Tuple[ArrayLike, ArrayLike]:
    #     r"""Specify how to serialize class into a JAX pytree.
    #
    #     :return: A tuple with 2 elements:
    #      - The `children`, containing arrays & pytrees
    #      - The `aux_data`, containing static and hashable data.
    #     """
    #     children = ()
    #     aux_data = None
    #     return children, aux_data
    #
    # @classmethod
    # def tree_unflatten(
    #         cls,
    #         aux_data: Tuple[Any, ...],
    #         children: Tuple[ArrayLike, ArrayLike],
    # ) -> "JaxErrors":
    #     r"""Specify how to reconstruct class from a JAX pytree.
    #
    #     :param aux_data: Contains static, hashable data.
    #     :param children: Contain arrays & pytrees.
    #     :return: Reconstructed class.
    #     """
    #     return cls()
    #
    # @staticmethod
    # @jax.jit
    # def value_error():
    #     r"""Pass `ValueError` via functional interface."""
    #     return ValueError
    #
    # @staticmethod
    # @jax.jit
    # def type_error():
    #     r"""Pass `TypeError` via functional interface."""
    #     return TypeError

    @staticmethod
    @jax.jit
    def raise_no_error(*args):
        r"""Handle case with no errors."""
        ...

    @staticmethod
    @jax.jit
    def raise_value_error_with_custom_message(msg=None):
        r"""Handle general errors."""
        raise ValueError(msg)

    @staticmethod
    @jax.jit
    def raise_error_for_predict_on_non_finite(*args):
        r"""Handle error in `predict`-methods for inputs with Nans/ infs."""
        raise ValueError("While predicting. Nans/ inf not supported.")

    @staticmethod
    @jax.jit
    def raise_error_for_predict_on_too_low_dim(*args):
        r"""Handle error in `predict`-methods with a low dimensional input."""
        raise ValueError("Input array must be at least 2D. Reshape your data.")

    @staticmethod
    @jax.jit
    def raise_error_for_fit_on_complex_data_error(*args):
        r"""Handle error in `fit`-methods when complex data is encountered."""
        raise ValueError("While fitting. Complex data not supported.")

    @staticmethod
    @jax.jit
    def raise_error_for_fit_on_empty_data(x, *args):
        r"""Handle error in `fit`-methods when an empty array is given."""
        raise ValueError(
            f"0 feature(s) (shape={x.shape}) while a minimum of 1 is required."
            f"Got Cannot fit empty data.",
        )

    @staticmethod
    @jax.jit
    def raise_error_for_fit_on_nans_or_infs(*args):
        r"""Handle error in `fit`-methods Nans/ infs are encountered."""
        raise ValueError("While fitting. Nans/ inf not supported.")

    @staticmethod
    @jax.jit
    def raise_error_non_fitted(*args):
        r"""Handle errors raised by an un-fitted estimator."""
        raise AssertionError("Estimator has not been fitted.")


if __name__ == "__main__":
    pytest.main()
