
from dtaianomaly.anomaly_detection import IsolationForest


class TestIsolationForest:

    def test_initialize(self):
        detector = IsolationForest(15, n_estimators=42)
        assert detector.window_size == 15
        assert detector.stride == 1
        assert detector.kwargs['n_estimators'] == 42

    def test_str(self):
        assert str(IsolationForest(5)) == "IsolationForest(window_size=5)"
        assert str(IsolationForest(15, 3)) == "IsolationForest(window_size=15,stride=3)"
        assert str(IsolationForest(25, n_estimators=42)) == "IsolationForest(window_size=25,n_estimators=42)"
        assert str(IsolationForest(25, max_samples=50)) == "IsolationForest(window_size=25,max_samples=50)"
        assert str(IsolationForest(25, max_samples='auto')) == "IsolationForest(window_size=25,max_samples='auto')"
