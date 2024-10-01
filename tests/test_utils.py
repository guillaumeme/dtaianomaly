
import numpy as np
from typing import Any
from dtaianomaly.utils import is_valid_list, is_valid_array_like


class TestIsValidList:

    def test_empty_list(self):
        assert is_valid_list([], Any)

    def test_np_array(self):
        assert not is_valid_list(np.array([1, 2, 3, 4, 5, 6]), Any)

    def test_valid(self):
        assert is_valid_list([1, 2, 3, 4, 5, 6], int)

    def test_invalid(self):
        assert not is_valid_list([1, 2, 3, '4', 5, 6], int)


class TestIsValidArrayLike:

    def test_empty_list(self):
        assert is_valid_array_like([])

    def test_empty_np_array(self):
        assert is_valid_array_like(np.array([]))

    def test_valid_list_int(self):
        assert is_valid_array_like([1, 2, 3, 4, 5])

    def test_valid_np_array_int(self):
        assert is_valid_array_like(np.array([1, 2, 3, 4, 5]))

    def test_valid_list_float(self):
        assert is_valid_array_like([1.9, 2.8, 3.7, 4.6, 5.5])

    def test_valid_np_array_float(self):
        assert is_valid_array_like(np.array([1.9, 2.8, 3.7, 4.6, 5.5]))

    def test_valid_list_bool(self):
        assert is_valid_array_like([True, True, False, True, False])

    def test_valid_np_array_bool(self):
        assert is_valid_array_like(np.array([True, True, False, True, False]))

    def test_valid_list_mixed_type(self):
        assert is_valid_array_like([1.9, 2, True, 4, 5.5])

    def test_valid_np_array_mixed_type(self):
        assert is_valid_array_like(np.array([1.9, 2, True, 4, 5.5]))

    def test_invalid_list(self):
        assert not is_valid_array_like([1, 2, 3, 4, '5'])

    def test_invalid_np_array(self):
        assert not is_valid_array_like(np.array([1, 2, 3, 4, '5']))

    def test_invalid_int(self):
        assert not is_valid_array_like(1)

    def test_invalid_float(self):
        assert not is_valid_array_like(1.9)

    def test_invalid_str(self):
        assert not is_valid_array_like('1')

    def test_invalid_bool(self):
        assert not is_valid_array_like(True)





