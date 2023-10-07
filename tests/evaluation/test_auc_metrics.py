
import pytest
import numpy as np

from src.evaluation.auc_metrics import roc_auc, pr_auc
from src.evaluation.classification_metrics import precision, recall
from src.evaluation.thresholding import fixed_value_threshold
from tests.evaluation.TestEvaluationUtil import TestEvaluationUtil


class TestAUCMetrics:

    ground_truth: np.array = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1])
    predicted: np.array = np.array([0.2, 0.7, 0.6, 0.3, 0.8, 0.4, 0.9, 0.1, 0.8])
    nb_reps: int = 100
    length_synthetic: int = 1000

    def test_util(self):
        predicted_02 = fixed_value_threshold(self.ground_truth, self.predicted, 0.2)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_02) == 5
        assert TestEvaluationUtil.false_positive(self.ground_truth, predicted_02) == 3
        assert TestEvaluationUtil.false_negative(self.ground_truth, predicted_02) == 0
        assert TestEvaluationUtil.true_negative(self.ground_truth, predicted_02) == 1

        predicted_07 = fixed_value_threshold(self.ground_truth, self.predicted, 0.7)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_07) == 4
        assert TestEvaluationUtil.false_positive(self.ground_truth, predicted_07) == 0
        assert TestEvaluationUtil.false_negative(self.ground_truth, predicted_07) == 1
        assert TestEvaluationUtil.true_negative(self.ground_truth, predicted_07) == 4

    def tpr(self, threshold: float) -> float:
        predicted = fixed_value_threshold(self.ground_truth, self.predicted, threshold)
        tp = TestEvaluationUtil.true_positive(self.ground_truth, predicted)
        fn = TestEvaluationUtil.false_negative(self.ground_truth, predicted)
        return tp / (fn + tp)

    def fpr(self, threshold: float) -> float:
        predicted = fixed_value_threshold(self.ground_truth, self.predicted, threshold)
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

        assert roc_auc(self.ground_truth, self.predicted) == 1.0

    def test_pr_auc(self):
        predicted_0 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.0)
        assert precision(self.ground_truth, predicted_0) == pytest.approx(5/9)
        assert recall(self.ground_truth, predicted_0) == pytest.approx(5/5)

        predicted_01 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.1)
        assert precision(self.ground_truth, predicted_01) == pytest.approx(5/9)
        assert recall(self.ground_truth, predicted_01) == pytest.approx(5/5)

        predicted_02 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.2)
        assert precision(self.ground_truth, predicted_02) == pytest.approx(5/8)
        assert recall(self.ground_truth, predicted_02) == pytest.approx(5/5)

        predicted_03 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.3)
        assert precision(self.ground_truth, predicted_03) == pytest.approx(5/7)
        assert recall(self.ground_truth, predicted_03) == pytest.approx(5/5)

        predicted_04 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.4)
        assert precision(self.ground_truth, predicted_04) == pytest.approx(5/6)
        assert recall(self.ground_truth, predicted_04) == pytest.approx(5/5)

        predicted_05 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.5)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_05) == 5
        assert precision(self.ground_truth, predicted_05) == pytest.approx(5/5)
        assert recall(self.ground_truth, predicted_05) == pytest.approx(5/5)

        predicted_06 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.6)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_06) == 5
        assert precision(self.ground_truth, predicted_06) == pytest.approx(5/5)
        assert recall(self.ground_truth, predicted_06) == pytest.approx(5/5)

        predicted_07 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.7)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_07) == 4
        assert precision(self.ground_truth, predicted_07) == pytest.approx(4/4)
        assert recall(self.ground_truth, predicted_07) == pytest.approx(4/5)

        predicted_08 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.8)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_08) == 3
        assert precision(self.ground_truth, predicted_08) == pytest.approx(3/3)
        assert recall(self.ground_truth, predicted_08) == pytest.approx(3/5)

        predicted_09 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=0.9)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_09) == 1
        assert precision(self.ground_truth, predicted_09) == pytest.approx(1/1)
        assert recall(self.ground_truth, predicted_09) == pytest.approx(1/5)

        predicted_1 = fixed_value_threshold(self.ground_truth, self.predicted, threshold=1.0)
        assert TestEvaluationUtil.true_positive(self.ground_truth, predicted_1) == 0
        assert precision(self.ground_truth, predicted_1) == pytest.approx(0.0)
        assert recall(self.ground_truth, predicted_1) == pytest.approx(0.0)

        assert pr_auc(self.ground_truth, self.predicted) == pytest.approx(1.0)
