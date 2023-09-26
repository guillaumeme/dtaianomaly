
import numpy as np
from typing import Optional
from pyod.models.base import BaseDetector

from anomaly_detection.TimeSeriesAnomalyDetector import TimeSeriesAnomalyDetector
from anomaly_detection.utility.Windowing import Windowing


class PYODAnomalyDetector(TimeSeriesAnomalyDetector):

    def __init__(self, pyod_anomaly_detector: BaseDetector, windowing: Windowing):
        self.__pyod_anomaly_detector: BaseDetector = pyod_anomaly_detector
        self.__windowing: Windowing = windowing

    def fit(self, trend_data: np.ndarray, labels: Optional[np.array] = None) -> 'PYODAnomalyDetector':
        self.__pyod_anomaly_detector.fit(self.__windowing.create_windows(trend_data), labels)
        return self

    def decision_function(self, trend_data: np.ndarray) -> np.array:
        windowed_decision_scores = self.__pyod_anomaly_detector.decision_function(self.__windowing.create_windows(trend_data))
        return self.__windowing.reverse_windowing(windowed_decision_scores)
