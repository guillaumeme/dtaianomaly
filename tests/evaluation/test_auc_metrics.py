
import pytest
import numpy as np

from dtaianomaly.evaluation.auc_metrics import RocAUC, PrAUC
from dtaianomaly.evaluation.classification_metrics import Precision, Recall
from dtaianomaly.evaluation.thresholding import FixedValueThresholding
from tests.evaluation.TestEvaluationUtil import TestEvaluationUtil


class TestAUCMetrics:

    ground_truth: np.array = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1])
    predicted: np.array = np.array([0.2, 0.7, 0.6, 0.3, 0.8, 0.4, 0.9, 0.1, 0.8])
    nb_reps: int = 100
    length_synthetic: int = 1000

    @staticmethod
    def fixed_value_threshold(ground_truth, scores, threshold=None):
        return FixedValueThresholding(threshold=threshold).apply(scores, ground_truth)

    def test_util(self):
        predicted_02 = self.fixed_value_threshold(self.ground_truth, self.predicted, 0.2)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_02) == 5
        assert TestEvaluationUtil.false_positive(self.ground_truth, predicted_02) == 3
        assert TestEvaluationUtil.false_negative(self.ground_truth, predicted_02) == 0
        assert TestEvaluationUtil.true_negative(self.ground_truth, predicted_02) == 1

        predicted_07 = self.fixed_value_threshold(self.ground_truth, self.predicted, 0.7)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_07) == 4
        assert TestEvaluationUtil.false_positive(self.ground_truth, predicted_07) == 0
        assert TestEvaluationUtil.false_negative(self.ground_truth, predicted_07) == 1
        assert TestEvaluationUtil.true_negative(self.ground_truth, predicted_07) == 4

    def tpr(self, threshold: float) -> float:
        predicted = self.fixed_value_threshold(self.ground_truth, self.predicted, threshold)
        tp = TestEvaluationUtil.true_positive(self.ground_truth, predicted)
        fn = TestEvaluationUtil.false_negative(self.ground_truth, predicted)
        return tp / (fn + tp)

    def fpr(self, threshold: float) -> float:
        predicted = self.fixed_value_threshold(self.ground_truth, self.predicted, threshold)
        fp = TestEvaluationUtil.false_positive(self.ground_truth, predicted)
        tn = TestEvaluationUtil.true_negative(self.ground_truth, predicted)
        return fp / (fp + tn)

    def test_roc_auc(self):
        tpr_00 = self.tpr(0.0)
        fpr_00 = self.fpr(0.0)
        assert tpr_00 == pytest.approx(5/5)
        assert fpr_00 == pytest.approx(4/4)

        tpr_01 = self.tpr(0.1)
        fpr_01 = self.fpr(0.1)
        assert tpr_01 == pytest.approx(5/5)
        assert fpr_01 == pytest.approx(4/4)

        tpr_02 = self.tpr(0.2)
        fpr_02 = self.fpr(0.2)
        assert tpr_02 == pytest.approx(5/5)
        assert fpr_02 == pytest.approx(3/4)

        tpr_03 = self.tpr(0.3)
        fpr_03 = self.fpr(0.3)
        assert tpr_03 == pytest.approx(5/5)
        assert fpr_03 == pytest.approx(2/4)

        tpr_04 = self.tpr(0.4)
        fpr_04 = self.fpr(0.4)
        assert tpr_04 == pytest.approx(5/5)
        assert fpr_04 == pytest.approx(1/4)

        tpr_05 = self.tpr(0.5)
        fpr_05 = self.fpr(0.5)
        assert tpr_05 == pytest.approx(5/5)
        assert fpr_05 == pytest.approx(0/4)

        tpr_06 = self.tpr(0.6)
        fpr_06 = self.fpr(0.6)
        assert tpr_06 == pytest.approx(5/5)
        assert fpr_06 == pytest.approx(0/4)

        tpr_07 = self.tpr(0.7)
        fpr_07 = self.fpr(0.7)
        assert tpr_07 == pytest.approx(4/5)
        assert fpr_07 == pytest.approx(0/4)

        tpr_08 = self.tpr(0.8)
        fpr_08 = self.fpr(0.8)
        assert tpr_08 == pytest.approx(3/5)
        assert fpr_08 == pytest.approx(0/4)

        tpr_09 = self.tpr(0.9)
        fpr_09 = self.fpr(0.9)
        assert tpr_09 == pytest.approx(1/5)
        assert fpr_09 == pytest.approx(0/4)

        tpr_1 = self.tpr(1.0)
        fpr_1 = self.fpr(1.0)
        assert tpr_1 == pytest.approx(0/5)
        assert fpr_1 == pytest.approx(0/4)

        assert RocAUC().compute(self.ground_truth, self.predicted) == 1.0

    def test_pr_auc(self):
        assert Precision(FixedValueThresholding(0.0)).compute(self.ground_truth, self.predicted) == pytest.approx(5/9)
        assert Recall(FixedValueThresholding(0.0)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)

        assert Precision(FixedValueThresholding(0.1)).compute(self.ground_truth, self.predicted) == pytest.approx(5/9)
        assert Recall(FixedValueThresholding(0.1)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)

        assert Precision(FixedValueThresholding(0.2)).compute(self.ground_truth, self.predicted) == pytest.approx(5/8)
        assert Recall(FixedValueThresholding(0.2)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)

        assert Precision(FixedValueThresholding(0.3)).compute(self.ground_truth, self.predicted) == pytest.approx(5/7)
        assert Recall(FixedValueThresholding(0.3)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)

        assert Precision(FixedValueThresholding(0.4)).compute(self.ground_truth, self.predicted) == pytest.approx(5/6)
        assert Recall(FixedValueThresholding(0.4)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)

        assert TestEvaluationUtil.true_positive(self.ground_truth, FixedValueThresholding(0.5).apply(self.predicted, self.ground_truth)) == 5
        assert Precision(FixedValueThresholding(0.5)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)
        assert Recall(FixedValueThresholding(0.5)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)

        assert TestEvaluationUtil.true_positive(self.ground_truth, FixedValueThresholding(0.6).apply(self.predicted, self.ground_truth)) == 5
        assert Precision(FixedValueThresholding(0.6)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)
        assert Recall(FixedValueThresholding(0.6)).compute(self.ground_truth, self.predicted) == pytest.approx(5/5)

        assert TestEvaluationUtil.true_positive(self.ground_truth, FixedValueThresholding(0.7).apply(self.predicted, self.ground_truth)) == 4
        assert Precision(FixedValueThresholding(0.7)).compute(self.ground_truth, self.predicted) == pytest.approx(4/4)
        assert Recall(FixedValueThresholding(0.7)).compute(self.ground_truth, self.predicted) == pytest.approx(4/5)

        assert TestEvaluationUtil.true_positive(self.ground_truth, FixedValueThresholding(0.8).apply(self.predicted, self.ground_truth)) == 3
        assert Precision(FixedValueThresholding(0.8)).compute(self.ground_truth, self.predicted) == pytest.approx(3/3)
        assert Recall(FixedValueThresholding(0.8)).compute(self.ground_truth, self.predicted) == pytest.approx(3/5)

        assert TestEvaluationUtil.true_positive(self.ground_truth, FixedValueThresholding(0.9).apply(self.predicted, self.ground_truth)) == 1
        assert Precision(FixedValueThresholding(0.9)).compute(self.ground_truth, self.predicted) == pytest.approx(1/1)
        assert Recall(FixedValueThresholding(0.9)).compute(self.ground_truth, self.predicted) == pytest.approx(1/5)

        assert TestEvaluationUtil.true_positive(self.ground_truth, FixedValueThresholding(1.0).apply(self.predicted, self.ground_truth)) == 0
        assert Precision(FixedValueThresholding(1.0)).compute(self.ground_truth, self.predicted) == pytest.approx(0.0)
        assert Recall(FixedValueThresholding(1.0)).compute(self.ground_truth, self.predicted) == pytest.approx(0.0)

        assert PrAUC().compute(self.ground_truth, self.predicted) == pytest.approx(1.0)
