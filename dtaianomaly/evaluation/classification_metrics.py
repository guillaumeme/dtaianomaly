
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

from dtaianomaly.evaluation.thresholding import Thresholding
from dtaianomaly.evaluation.Metric import ThresholdingMetric


class Precision(ThresholdingMetric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        return precision_score(ground_truth_anomalies, self._get_anomaly_labels(predicted_anomaly_scores, ground_truth_anomalies))


class Recall(ThresholdingMetric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        return recall_score(ground_truth_anomalies, self._get_anomaly_labels(predicted_anomaly_scores, ground_truth_anomalies))


class Fbeta(ThresholdingMetric):

    def __init__(self, thresholding: Thresholding, beta: float = 1.0):
        super().__init__(thresholding)
        self.__beta: float = beta

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        return fbeta_score(ground_truth_anomalies, self._get_anomaly_labels(predicted_anomaly_scores, ground_truth_anomalies), beta=self.__beta)
