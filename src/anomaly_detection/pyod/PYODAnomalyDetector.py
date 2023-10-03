
import numpy as np
import importlib
from typing import Optional, Dict
from pyod.models.base import BaseDetector

from src.anomaly_detection.TimeSeriesAnomalyDetector import TimeSeriesAnomalyDetector
from src.anomaly_detection.utility.Windowing import Windowing

_SUPPORTED_PYOD_ANOMALY_DETECTORS = {
    # key is the name to use when loading, value is the name of the module in PYOD
    'IForest': 'iforest',
    'LOF': 'lof',
    'KNN': 'knn'
}


class PYODAnomalyDetector(TimeSeriesAnomalyDetector):

    def __init__(self, pyod_anomaly_detector: BaseDetector, windowing: Windowing):
        self.__pyod_anomaly_detector: BaseDetector = pyod_anomaly_detector
        self.__windowing: Windowing = windowing

    def fit(self, trend_data: np.ndarray, labels: Optional[np.array] = None) -> 'PYODAnomalyDetector':
        self.__pyod_anomaly_detector.fit(self.__windowing.create_windows(trend_data))
        return self

    def decision_function(self, trend_data: np.ndarray) -> np.array:
        windowed_decision_scores = self.__pyod_anomaly_detector.decision_function(self.__windowing.create_windows(trend_data))
        return self.__windowing.reverse_windowing(windowed_decision_scores)

    @staticmethod
    def load(parameters: Dict[str, any]) -> 'TimeSeriesAnomalyDetector':

        # Check if the given anomaly detector is supported
        if parameters['pyod_model'] not in _SUPPORTED_PYOD_ANOMALY_DETECTORS:
            raise ValueError(f"The given anomaly detector '{parameters['pyod_model']}' is not supported yet, or is not a valid PYODAnomalyDetector!\n"
                             f"Supported PYODAnomalyDetectors are: {_SUPPORTED_PYOD_ANOMALY_DETECTORS.keys()}")

        # Load the module and class
        module = importlib.import_module(name='pyod.models.' + _SUPPORTED_PYOD_ANOMALY_DETECTORS[parameters['pyod_model']])
        pyod_anomaly_detector = getattr(module, parameters['pyod_model'])

        # Initialize the anomaly detector
        return PYODAnomalyDetector(
            pyod_anomaly_detector=pyod_anomaly_detector(**(parameters['pyod_model_parameters'] if 'pyod_model_parameters' in parameters else {})),
            windowing=Windowing(**(parameters['windowing'] if 'windowing' in parameters else {}))
        )
