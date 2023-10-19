import abc
import numpy as np

from dtaianomaly.evaluation.thresholding import Thresholding


class Metric(abc.ABC):

    @abc.abstractmethod
    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        raise NotImplementedError("The method 'compute(np.array, np.array)' must be implemented by a subclass of Metric!")


class ThresholdingMetric(Metric, abc.ABC):

    def __init__(self, thresholding: Thresholding):
        self.__thresholding: Thresholding = thresholding

    def _get_anomaly_labels(self, anomaly_scores: np.array, ground_truth_anomalies: np.array) -> np.array:
        return self.__thresholding.apply(anomaly_scores, ground_truth_anomalies)
