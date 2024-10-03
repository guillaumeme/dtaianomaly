import pytest

from dtaianomaly.preprocessing import Identity, ZNormalizer, ChainedPreprocessor
from dtaianomaly.anomaly_detection import IsolationForest

from dtaianomaly.pipeline import Pipeline


class TestPipeline:

    def test_initialization(self):
        Pipeline(ZNormalizer(), IsolationForest(15))

    def test_list_of_preprocessors(self):
        Pipeline([ZNormalizer(), Identity()], IsolationForest(15))

    def test_no_preprocessors(self):
        with pytest.raises(ValueError):
            Pipeline([], IsolationForest(15))

    def test_invalid_preprocessor(self):
        with pytest.raises(TypeError):
            Pipeline('bonkers', IsolationForest(15))

    def test_invalid_list(self):
        with pytest.raises(TypeError):
            Pipeline([ZNormalizer(), 'bonkers'], IsolationForest(15))

    def test_invalid_detector(self):
        with pytest.raises(TypeError):
            Pipeline(ZNormalizer(), 'IsolationForest')

    def test_str(self):
        assert str(Pipeline(ZNormalizer(), IsolationForest(15, 3))) == 'ZNormalizer()->IsolationForest(window_size=15,stride=3)'
        assert (str(Pipeline(ChainedPreprocessor(Identity(), ZNormalizer()), IsolationForest(15, 3)))
                == 'Identity()->ZNormalizer()->IsolationForest(window_size=15,stride=3)')


