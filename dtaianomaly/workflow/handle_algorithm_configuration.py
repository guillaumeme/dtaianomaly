
import json
import importlib
from typing import Dict, Any, Union, Tuple
from dtaianomaly.anomaly_detection import *

AlgorithmConfiguration = Union[Dict[str, Any], str, Tuple[TimeSeriesAnomalyDetector, str]]


def handle_algorithm_configuration(algorithm_configuration: AlgorithmConfiguration) -> Dict[str, TimeSeriesAnomalyDetector]:

    # Check if the given algorithm configuration is a tuple of the form (TimeSeriesAnomalyDetector, str)
    if isinstance(algorithm_configuration, tuple) \
            and len(algorithm_configuration) == 2 \
            and isinstance(algorithm_configuration[0], TimeSeriesAnomalyDetector) \
            and isinstance(algorithm_configuration[1], str):
        return {algorithm_configuration[1]: algorithm_configuration[0]}

    if type(algorithm_configuration) is str:
        configuration_file = open(algorithm_configuration, 'r')
        algorithm_configuration = json.load(configuration_file)
        configuration_file.close()

    if 'algorithm_configurations' in algorithm_configuration:
        algorithms = {}
        sub_dir = algorithm_configuration['collection'] if 'sub_dir_name' in algorithm_configuration else ''
        for config in algorithm_configuration['algorithm_configurations']:
            algorithm, name = read_algorithm_configuration(config)
            algorithms[f'{sub_dir}/{name}'] = algorithm

    else:
        algorithm, name = read_algorithm_configuration(algorithm_configuration)
        algorithms = {name: algorithm}

    return algorithms


def read_algorithm_configuration(algorithm_configuration) -> Tuple[TimeSeriesAnomalyDetector, str]:

    # Read the algorithm configuration file if it is a string
    if type(algorithm_configuration) is str:
        configuration_file = open(algorithm_configuration, 'r')
        algorithm_configuration = json.load(configuration_file)
        configuration_file.close()

    # Load the specific anomaly detector class
    module_path = algorithm_configuration['module_path'] if 'module_path' in algorithm_configuration else 'dtaianomaly.anomaly_detection'
    anomaly_detector_class_object: TimeSeriesAnomalyDetector = getattr(importlib.import_module(module_path), algorithm_configuration['anomaly_detector'])

    # Load and return the specific anomaly detector instance, with the given hyperparameters
    return anomaly_detector_class_object.load(parameters=algorithm_configuration['hyperparameters']), algorithm_configuration['name']