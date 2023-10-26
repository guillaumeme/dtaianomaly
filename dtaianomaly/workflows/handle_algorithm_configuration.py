
import json
import importlib
from typing import Dict, Any, Union, Tuple
from dtaianomaly.anomaly_detection import *

AlgorithmConfiguration = Union[Dict[str, Any], str, Tuple[TimeSeriesAnomalyDetector, str]]


def handle_algorithm_configuration(algorithm_configuration: AlgorithmConfiguration) -> Tuple[TimeSeriesAnomalyDetector, str]:

    # Check if the given algorithm configuration is a tuple of the form (TimeSeriesAnomalyDetector, str)
    if isinstance(algorithm_configuration, tuple) \
            and len(algorithm_configuration) == 2 \
            and isinstance(algorithm_configuration[0], TimeSeriesAnomalyDetector) \
            and isinstance(algorithm_configuration[1], str):
        return algorithm_configuration

    # Read the algorithm configuration file if it is a string
    if type(algorithm_configuration) is str:
        configuration_file = open(algorithm_configuration, 'r')
        algorithm_configuration = json.load(configuration_file)
        configuration_file.close()

    # Load the specific anomaly detector class
    anomaly_detector_class_object: TimeSeriesAnomalyDetector = getattr(importlib.import_module('dtaianomaly.anomaly_detection'), algorithm_configuration['anomaly_detector'])

    # Load and return the specific anomaly detector instance
    return anomaly_detector_class_object.load(parameters=algorithm_configuration), algorithm_configuration['name']
