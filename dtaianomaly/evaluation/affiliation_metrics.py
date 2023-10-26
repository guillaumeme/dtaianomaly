
import numpy as np
from dtaianomaly.evaluation.Metric import ThresholdingMetric
from dtaianomaly.evaluation.thresholding import Thresholding

from dtaianomaly.evaluation.affiliation_util.generics import convert_vector_to_events
from dtaianomaly.evaluation.affiliation_util.metrics import precision_from_events, recall_from_events


class AffiliationPrecision(ThresholdingMetric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        ground_truth_events = convert_vector_to_events(ground_truth_anomalies)
        predicted_anomaly_labels = self._get_anomaly_labels(predicted_anomaly_scores, ground_truth_anomalies)
        predicted_events = convert_vector_to_events(predicted_anomaly_labels)
        t_range = (0, ground_truth_anomalies.shape[0])
        try:
            return precision_from_events(predicted_events, ground_truth_events, t_range)
        except ValueError:
            return np.nan


class AffiliationRecall(ThresholdingMetric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        ground_truth_events = convert_vector_to_events(ground_truth_anomalies)
        predicted_anomaly_labels = self._get_anomaly_labels(predicted_anomaly_scores, ground_truth_anomalies)
        predicted_events = convert_vector_to_events(predicted_anomaly_labels)
        t_range = (0, ground_truth_anomalies.shape[0])
        try:
            return recall_from_events(predicted_events, ground_truth_events, t_range)
        except ValueError:
            return np.nan


class AffiliationFBeta(ThresholdingMetric):

    def __init__(self, thresholding: Thresholding, beta: float = 1.0):
        super().__init__(thresholding)
        self.__beta: float = beta

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        ground_truth_events = convert_vector_to_events(ground_truth_anomalies)
        predicted_anomaly_labels = self._get_anomaly_labels(predicted_anomaly_scores, ground_truth_anomalies)
        predicted_events = convert_vector_to_events(predicted_anomaly_labels)
        t_range = (0, ground_truth_anomalies.shape[0])
        try:
            precision = precision_from_events(predicted_events, ground_truth_events, t_range)
            recall = recall_from_events(predicted_events, ground_truth_events, t_range)
        except ValueError:
            return np.nan
        return (1 + self.__beta**2) * (precision * recall) / (self.__beta**2 * precision + recall)
