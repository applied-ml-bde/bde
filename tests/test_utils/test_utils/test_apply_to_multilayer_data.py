import pytest
import pathlib

import bde.utils.utils
import numpy as np
from bde.utils import configs as cnfg

SEED = cnfg.General.SEED
DATA_SAMPLES = [None, 1, 1.0, 1.0 + 1.0j, np.ones((1,)), (1,), "1.0"]
D_SAMPLES_ZEROS = [0 for _ in DATA_SAMPLES]


@pytest.fixture
def nullify():
    return lambda x: 0


class TestOnRawInputs:
    @staticmethod
    @pytest.mark.parametrize("data", DATA_SAMPLES)
    def test_dtypes_is_none(data, nullify):
        assert bde.utils.utils.apply_to_multilayer_data(data=data, f=nullify, dtypes=None) == 0

    @staticmethod
    @pytest.mark.parametrize("data", DATA_SAMPLES)
    def test_dtypes_tuple(data, nullify):
        dtypes = [type(d) for d in DATA_SAMPLES]
        dtypes = tuple(dtypes)
        assert bde.utils.utils.apply_to_multilayer_data(data=data, f=nullify, dtypes=dtypes) == 0

    @staticmethod
    @pytest.mark.parametrize("data", DATA_SAMPLES)
    def test_dtype(data, nullify):
        assert bde.utils.utils.apply_to_multilayer_data(data=data, f=nullify, dtypes=type(data)) == 0

    @staticmethod
    @pytest.mark.parametrize("data", DATA_SAMPLES)
    def test_dtype_incorrect(data, nullify):
        bad_dtype = int if type(data) is not int else float
        assert bde.utils.utils.apply_to_multilayer_data(data=data, f=nullify, dtypes=bad_dtype) == data

    @staticmethod
    @pytest.mark.parametrize("data", DATA_SAMPLES)
    def test_dtype_not_in_dtypes_tuple(data, nullify):
        dtypes = [type(d) for d in DATA_SAMPLES if d != data]
        dtypes = tuple(dtypes)
        assert bde.utils.utils.apply_to_multilayer_data(data=data, f=nullify, dtypes=dtypes) == data


class TestSingleLayer:
    @staticmethod
    def test_on_flat_list_dtypes_none(nullify):
        assert bde.utils.utils.apply_to_multilayer_data(
            data=DATA_SAMPLES,
            f=nullify,
            dtypes=None,
        ) == [0 for _ in DATA_SAMPLES]

    @staticmethod
    @pytest.mark.parametrize("data", DATA_SAMPLES)
    def test_on_flat_list_dtypes_single_dtype(data, nullify):
        assert bde.utils.utils.apply_to_multilayer_data(
            data=DATA_SAMPLES,
            f=nullify,
            dtypes=type(data),
        ) == [0 if isinstance(x, type(data)) else x for x in DATA_SAMPLES]


class TestDoubleLayers:
    @staticmethod
    def test_list_of_items_dtypes_none(nullify):
        data = DATA_SAMPLES + [
            {"1": 1, "2.0": 2.0, "c": "c", "4": 4.0 + 0.0j},
            {"test": -1e0, "": None},
            DATA_SAMPLES,
        ]
        exp_output = D_SAMPLES_ZEROS + [
            {"1": 0, "2.0": 0, "c": 0, "4": 0},
            {"test": 0, "": 0},
            D_SAMPLES_ZEROS,
        ]
        assert bde.utils.utils.apply_to_multilayer_data(
            data=data,
            f=nullify,
            dtypes=None,
        ) == exp_output

    @staticmethod
    def test_dict_of_items_dtypes_none(nullify):
        data = {
            "samples": DATA_SAMPLES,
            "sub": {"1": 1, "2.0": 2.0, "c": "c", "4": 4.0 + 0.0j},
            "test": -1e0,
            "": None,
        }
        exp_output = {
            "samples": D_SAMPLES_ZEROS,
            "sub": {"1": 0, "2.0": 0, "c": 0, "4": 0},
            "test": 0,
            "": 0,
        }
        assert bde.utils.utils.apply_to_multilayer_data(
            data=data,
            f=nullify,
            dtypes=None,
        ) == exp_output


if __name__ == '__main__':
    pytest.main()
