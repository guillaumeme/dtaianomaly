
import pytest
from dtaianomaly.anomaly_detection import MatrixProfileDetector


class TestMatrixProfileDetector:

    def test_initialize_non_int_window_size(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=True)
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size='a string')
        MatrixProfileDetector(5)  # Doesn't raise an error

    def test_initialize_too_small_window_size(self):
        with pytest.raises(ValueError):
            MatrixProfileDetector(window_size=0)
        MatrixProfileDetector(5)  # Doesn't raise an error

    def test_initialize_non_bool_normalize(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, normalize=0)
        MatrixProfileDetector(5, normalize=False)  # Doesn't raise an error

    def test_initialize_non_numeric_p(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, p=False)
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, p='a string')
        MatrixProfileDetector(5, p=2.5)  # Doesn't raise an error with float
        MatrixProfileDetector(5, p=3)  # Doesn't raise an error with int

    def test_initialize_too_small_p(self):
        with pytest.raises(ValueError):
            MatrixProfileDetector(window_size=15, p=0.5)
        MatrixProfileDetector(5, p=2.5)  # Doesn't raise an error with float
        MatrixProfileDetector(5, p=3)  # Doesn't raise an error with int

    def test_initialize_non_integer_k(self):
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, k=1.5)
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, k=True)
        with pytest.raises(TypeError):
            MatrixProfileDetector(window_size=15, k='a string')
        MatrixProfileDetector(5, k=2)  # Doesn't raise an error

    def test_initialize_too_small_k(self):
        with pytest.raises(ValueError):
            MatrixProfileDetector(window_size=15, k=0)
        MatrixProfileDetector(5, k=2)  # Doesn't raise an error

    def test_str(self):
        assert str(MatrixProfileDetector(5)) == "MatrixProfileDetector(window_size=5)"
        assert str(MatrixProfileDetector(15, normalize=False, p=1.5)) == "MatrixProfileDetector(window_size=15,normalize=False,p=1.5)"
        assert str(MatrixProfileDetector(15, p=1.5, normalize=False)) == "MatrixProfileDetector(window_size=15,normalize=False,p=1.5)"
        assert str(MatrixProfileDetector(25, k=2)) == "MatrixProfileDetector(window_size=25,k=2)"
