r"""General utility methods.

This module contains utility methods used by other modules.

Functions
---------
- `apply_to_multilayer_data`: A method for applying a transformation over a multilayered collection of data.

"""

import pytest
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


if __name__ == '__main__':
    pytest.main()
