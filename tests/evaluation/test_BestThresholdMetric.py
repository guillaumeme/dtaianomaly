
import pytest
import numpy as np

from dtaianomaly.evaluation.BestThresholdMetric import BestThresholdMetric
from dtaianomaly.evaluation.simple_proba_metrics import AreaUnderROC
from dtaianomaly.evaluation.simple_binary_metrics import Precision, FBeta, Recall


class TestBestThresholdMetric:

    def test_string_metric(self):
        with pytest.raises(TypeError):
            BestThresholdMetric('Precision()')

    def test_proba_metric(self):
        with pytest.raises(TypeError):
            BestThresholdMetric(AreaUnderROC())

    def test_string_max_nb_thresholds(self):
        with pytest.raises(TypeError):
            BestThresholdMetric(Precision(), "10")

    def test_bool_max_nb_thresholds(self):
        with pytest.raises(TypeError):
            BestThresholdMetric(Precision(), True)

    def test_zero_max_nb_thresholds(self):
        with pytest.raises(ValueError):
            BestThresholdMetric(Precision(), 0)

    def test_negative_max_nb_thresholds(self):
        with pytest.raises(ValueError):
            BestThresholdMetric(Precision(), -2)
        BestThresholdMetric(Precision(), -1)  # is ok

    def test_precision(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.8, 0.2, 0.7, 0.7, 0.6])
        # Sorted scores: [0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(Precision())
        assert metric.compute(y_true, y_pred) == pytest.approx(1.0)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_recall(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.8, 0.2, 0.7, 0.7, 0.6])
        # Sorted scores: [0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(Recall())
        assert metric.compute(y_true, y_pred) == pytest.approx(1.0)
        assert metric.threshold_ == pytest.approx(0.0)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.8, 0.2, 0.7, 0.7, 0.6])
        # Sorted scores: [0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(FBeta())
        assert metric.compute(y_true, y_pred) == pytest.approx(1.0)
        assert metric.threshold_ == pytest.approx(0.65)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta_2(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(FBeta())
        assert metric.compute(y_true, y_pred) == pytest.approx(0.8333333, abs=1e-5)
        assert metric.threshold_ == pytest.approx(0.4)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta_2_subset_thresholds(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=4)
        assert metric.compute(y_true, y_pred) == pytest.approx(0.75)
        assert metric.threshold_ == pytest.approx(0.65)
        assert metric.thresholds_.shape == (4,)
        assert metric.thresholds_.shape == metric.scores_.shape

    def test_fbeta_2_given_thresholds(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = np.array([0.2, 0.3, 0.65, 0.85, 0.9])
        metric = BestThresholdMetric(FBeta())
        assert metric.compute(y_true, y_pred, thresholds=thresholds) == pytest.approx(0.75)
        assert metric.threshold_ == pytest.approx(0.65)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert np.array_equal(metric.thresholds_, thresholds)

    def test_fbeta_2_given_thresholds_and_subset(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.3, 0.3, 0.8, 0.9, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5, 0.6])
        # Sorted scores: [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = np.array([0.2, 0.3, 0.65, 0.85, 0.9])
        metric = BestThresholdMetric(FBeta(), max_nb_thresholds=2)
        assert metric.compute(y_true, y_pred, thresholds=thresholds) == pytest.approx(2/3)
        assert metric.threshold_ == pytest.approx(0.3)
        assert metric.thresholds_.shape == (2,)
        assert metric.thresholds_.shape == metric.scores_.shape
        assert np.array_equal(metric.thresholds_, np.array([0.3, 0.85]))

    def test_str(self):
        assert str(BestThresholdMetric(Precision())) == "BestThresholdMetric(metric=Precision())"
        assert str(BestThresholdMetric(FBeta(beta=2.0))) == "BestThresholdMetric(metric=FBeta(beta=2.0))"
