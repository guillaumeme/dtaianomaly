
import pytest
import numpy as np

from dtaianomaly.evaluation.classification_metrics import Precision, Recall, Fbeta
from dtaianomaly.evaluation.thresholding import FixedValueThresholding
from tests.evaluation.TestEvaluationUtil import TestEvaluationUtil


class TestClassificationMetrics:

    ground_truth: np.array = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    predicted: np.array = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    nb_reps: int = 100
    length_synthetic: int = 1000

    @staticmethod
    def precision(ground_truth, predicted):
        return Precision(FixedValueThresholding(0.5)).compute(ground_truth, predicted)

    @staticmethod
    def recall(ground_truth, predicted):
        return Recall(FixedValueThresholding(0.5)).compute(ground_truth, predicted)

    @staticmethod
    def fbeta(ground_truth, predicted, beta=1.0):
        return Fbeta(FixedValueThresholding(0.5), beta).compute(ground_truth, predicted)

    def test_precision(self):
        assert np.count_nonzero(self.predicted) == 8
        assert self.precision(self.ground_truth, self.predicted) == pytest.approx(5 / 8)

    def test_recall(self):
        assert np.count_nonzero(self.ground_truth) == 8
        assert self.recall(self.ground_truth, self.predicted) == pytest.approx(5 / 8)

    def test_f1(self):
        assert self.fbeta(self.ground_truth, self.predicted, beta=1.0) == pytest.approx(5 / 8)

    def test_f05(self):
        assert self.fbeta(self.ground_truth, self.predicted, beta=0.5) == pytest.approx(5 / 8)

    def test_f2(self):
        assert self.fbeta(self.ground_truth, self.predicted, beta=2.0) == pytest.approx(5 / 8)

    def test_util(self):
        assert TestEvaluationUtil.true_positive(self.ground_truth, self.predicted) == 5
        assert TestEvaluationUtil.false_positive(self.ground_truth, self.predicted) == 3
        assert TestEvaluationUtil.false_negative(self.ground_truth, self.predicted) == 3

    def test_precision_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fp = TestEvaluationUtil.false_positive(ground_truth, predicted)
            assert self.precision(ground_truth, predicted) == pytest.approx(tp / (tp + fp))

    def test_recall_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fn = TestEvaluationUtil.false_negative(ground_truth, predicted)
            assert self.recall(ground_truth, predicted) == pytest.approx(tp / (tp + fn))

    def test_f1_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fp = TestEvaluationUtil.false_positive(ground_truth, predicted)
            fn = TestEvaluationUtil.false_negative(ground_truth, predicted)
            assert self.fbeta(ground_truth, predicted, beta=1.0) == pytest.approx(2*tp / (2*tp + fp + fn))

    def test_f05_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fp = TestEvaluationUtil.false_positive(ground_truth, predicted)
            fn = TestEvaluationUtil.false_negative(ground_truth, predicted)
            assert self.fbeta(ground_truth, predicted, beta=0.5) == pytest.approx(1.25*tp / (1.25*tp + fp + 0.25*fn))

    def test_f2_large_scale(self):
        for _ in range(self.nb_reps):
            ground_truth = np.random.choice([0, 1], size=self.length_synthetic)
            predicted = np.random.choice([0, 1], size=self.length_synthetic)
            tp = TestEvaluationUtil.true_positive(ground_truth, predicted)
            fp = TestEvaluationUtil.false_positive(ground_truth, predicted)
            fn = TestEvaluationUtil.false_negative(ground_truth, predicted)
            assert self.fbeta(ground_truth, predicted, beta=2.0) == pytest.approx(5*tp / (5*tp + fp + 4*fn))
