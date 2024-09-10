import jax
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import bde


@parametrize_with_checks(
    [
        bde.ml.models.FullyConnectedEstimator(),
        bde.ml.models.BDEEstimator(),
    ]
)
@pytest.mark.parametrize("do_use_jit", [False])
@pytest.mark.timeout(180)
# @pytest.mark.skip(
#     reason=(
#         "This causes the tests to freeze on github (not locally). "
#         "Temp skip until resolved."
#     ),
# )
def test_sklearn_estimator(do_use_jit, estimator, check):
    # NOTE: These tests fail in jitted mode.
    #  Make sure that this is due to the test design, and not our code.
    with jax.disable_jit(disable=not do_use_jit):
        check(estimator)


if __name__ == '__main__':
    pytest.main()
