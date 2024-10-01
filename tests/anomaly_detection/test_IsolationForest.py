
from dtaianomaly.anomaly_detection import IsolationForest


class TestIsolationForest:

    def test_initialize(self):
        detector = IsolationForest(15, n_estimators=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.detector.n_estimators == 42
        assert not detector.is_fitted

    def test_str(self):
        detector = IsolationForest(15, 3)
        assert str(detector) == "IsolationForest_15_3"
