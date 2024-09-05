import pytest
import jax

from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks
import bde
from bde.utils import configs as cnfg


@parametrize_with_checks([
    bde.ml.models.FullyConnectedEstimator(),
    bde.ml.models.BDEEstimator(),
])
@pytest.mark.parametrize("do_use_jit", [False])
def test_sklearn_estimator(do_use_jit, estimator, check):
    # NOTE: These tests fail in jitted mode.
    #  Make sure that these is due to the test design, and not our code.
    with jax.disable_jit(disable=not do_use_jit):
        check(estimator)


if __name__ == '__main__':
    pytest.main()
