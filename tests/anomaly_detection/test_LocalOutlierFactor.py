
from dtaianomaly.anomaly_detection import LocalOutlierFactor


class TestIsolationForest:

    def test_initialize(self):
        detector = LocalOutlierFactor(15, n_neighbors=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.kwargs['n_neighbors'] == 42

    def test_str(self):
        assert str(LocalOutlierFactor(5)) == "LocalOutlierFactor(window_size=5)"
        assert str(LocalOutlierFactor(15, 3)) == "LocalOutlierFactor(window_size=15,stride=3)"
        assert str(LocalOutlierFactor(25, n_neighbors=42)) == "LocalOutlierFactor(window_size=25,n_neighbors=42)"
