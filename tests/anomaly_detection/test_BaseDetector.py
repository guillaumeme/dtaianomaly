
import os
import pytest
import numpy as np
from typing import Optional

from dtaianomaly.anomaly_detection import BaseDetector, load_detector
from dtaianomaly import utils


class RandomDetector(BaseDetector):

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'RandomDetector':
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.random.uniform(size=X.shape[0])


class AlwaysNormalDetector(BaseDetector):

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AlwaysNormalDetector':
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(shape=X.shape[0])


class TestBaseDetector:

    def test_proba(self):
        data = np.random.standard_normal((50,))
        detector = RandomDetector()
        probas = detector.predict_proba(data)
        assert 0. <= np.min(probas)
        assert np.max(probas) <= 1.

    def test_proba_always_normal(self):
        data = np.random.standard_normal((50,))
        detector = AlwaysNormalDetector()
        probas = detector.predict_proba(data)
        assert 0. == np.min(probas)
        assert np.max(probas) == 0.0

    def test_proba_invalid(self):
        invalid_data = None
        detector = RandomDetector()
        assert not utils.is_valid_array_like(invalid_data)
        with pytest.raises(ValueError):
            detector.predict_proba(invalid_data)

    def test_save_invalid_path(self, tmp_path):
        detector = RandomDetector()
        detector.save(tmp_path / 'some' / 'invalid' / 'directory' / 'testing')
        assert os.path.exists(tmp_path / 'some' / 'invalid' / 'directory')

    def test_save_and_load(self, tmp_path):

        # Save the detector
        detector = RandomDetector()
        detector.save(tmp_path / 'testing')

        # Load the detector
        loaded_detector = load_detector(tmp_path / 'testing.dtai')

        # Check if the original detector and the loaded detector have the same properties
        assert detector.__dict__ == loaded_detector.__dict__

        # The loaded detector can make a prediction
        data = np.random.standard_normal((50,))
        _ = loaded_detector.predict_proba(data)

    def test_str(self):
        assert str(RandomDetector()) == 'RandomDetector()'
        assert str(AlwaysNormalDetector()) == 'AlwaysNormalDetector()'
