
import numpy as np
import importlib
from typing import Optional, Dict, Union
import warnings

from dtaianomaly.anomaly_detection.TimeSeriesAnomalyDetector import TimeSeriesAnomalyDetector
from dtaianomaly.anomaly_detection.utility.TrainType import TrainType
from dtaianomaly.anomaly_detection.utility.Windowing import Windowing

try:
    from pyod.models.base import BaseDetector
except ImportError:
    raise ImportError("Install 'pyod' in order to use 'PyODAnomalyDetector'!")

_SUPPORTED_PYOD_ANOMALY_DETECTORS = {
    # key is the name to use when loading, value is the name of the module in PYOD
    'IForest': 'iforest',
    'LOF': 'lof',
    'KNN': 'knn',
    'OCSVM': 'ocsvm',
    'ABOD': 'abod',
    'ECOD': 'ecod'
}


class PyODAnomalyDetector(TimeSeriesAnomalyDetector):
    """
    Anomaly detector based on the PyOD library.
    """
    def __init__(self, pyod_anomaly_detector: Union[BaseDetector, str], windowing: Windowing):
        super().__init__()

        if type(pyod_anomaly_detector) == str:
            # Check if the given anomaly detector is supported
            if pyod_anomaly_detector not in _SUPPORTED_PYOD_ANOMALY_DETECTORS:
                raise ValueError(f"The given anomaly detector '{pyod_anomaly_detector}' is not supported yet, or is not a valid PyODAnomalyDetector!\n"
                                 f"Supported PYODAnomalyDetectors are: {_SUPPORTED_PYOD_ANOMALY_DETECTORS.keys()}")

            # Load the module and class
            module = importlib.import_module(name='pyod.models.' + _SUPPORTED_PYOD_ANOMALY_DETECTORS[pyod_anomaly_detector])
            self.__pyod_anomaly_detector = getattr(module, pyod_anomaly_detector)()
        else:
            self.__pyod_anomaly_detector: BaseDetector = pyod_anomaly_detector
        self.__windowing: Windowing = windowing

    def train_type(self) -> TrainType:
        return TrainType.UNSUPERVISED  # All PyOD anomaly detectors are unsupervised (https://pyod.readthedocs.io/en/latest/api_cc.html#pyod.models.base.BaseDetector.fit)

    def _fit(self, trend_data: np.ndarray, labels: Optional[np.array] = None) -> 'PyODAnomalyDetector':
        self.__pyod_anomaly_detector.fit(self.__windowing.create_windows(trend_data))
        return self

    def _decision_function(self, trend_data: np.ndarray) -> np.array:
        windowed_decision_scores = self.__pyod_anomaly_detector.decision_function(self.__windowing.create_windows(trend_data))
        return self.__windowing.reverse_windowing(windowed_decision_scores, trend_data.shape[0])

    def predict_confidence(self, trend_data: np.ndarray, contamination: float) -> np.array:
        warnings.warn('To compute the confidence of an anomaly detector, the train data should be i.i.d., '
                      'which is not the case for a PyOD anomaly detector that predicts anomaly scores for a '
                      'sliding window.')
        return super().predict_confidence(trend_data, contamination)

    @staticmethod
    def load(parameters: Dict[str, any]) -> 'TimeSeriesAnomalyDetector':

        # Check if the given anomaly detector is supported
        if parameters['pyod_model'] not in _SUPPORTED_PYOD_ANOMALY_DETECTORS:
            raise ValueError(f"The given anomaly detector '{parameters['pyod_model']}' is not supported yet, or is not a valid PYODAnomalyDetector!\n"
                             f"Supported PyODAnomalyDetectors are: {_SUPPORTED_PYOD_ANOMALY_DETECTORS.keys()}")

        # Load the module and class
        module = importlib.import_module(name='pyod.models.' + _SUPPORTED_PYOD_ANOMALY_DETECTORS[parameters['pyod_model']])
        pyod_anomaly_detector = getattr(module, parameters['pyod_model'])

        # Initialize the anomaly detector
        return PyODAnomalyDetector(
            pyod_anomaly_detector=pyod_anomaly_detector(**(parameters['pyod_model_parameters'] if 'pyod_model_parameters' in parameters else {})),
            windowing=Windowing(**(parameters['windowing'] if 'windowing' in parameters else {}))
        )
