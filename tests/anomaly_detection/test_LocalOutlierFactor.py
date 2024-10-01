
from dtaianomaly.anomaly_detection import LocalOutlierFactor


class TestIsolationForest:

    def test_initialize(self):
        detector = LocalOutlierFactor(15, n_neighbors=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.detector.n_neighbors == 42
        assert not detector.is_fitted

    def test_str(self):
        detector = LocalOutlierFactor(15, 3)
        assert str(detector) == "LocalOutlierFactor_15_3"
