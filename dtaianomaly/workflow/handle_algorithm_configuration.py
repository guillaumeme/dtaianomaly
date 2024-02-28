
import json
import importlib
from typing import Dict, Any, Union, Tuple
from dtaianomaly.anomaly_detection import *

AlgorithmConfiguration = Union[Dict[str, Any], str, Tuple[TimeSeriesAnomalyDetector, str]]


def handle_algorithm_configuration(algorithm_configuration: AlgorithmConfiguration) -> Dict[str, Tuple[TimeSeriesAnomalyDetector, AlgorithmConfiguration]]:

    # Check if the given algorithm configuration is a tuple of the form (TimeSeriesAnomalyDetector, str)
    if isinstance(algorithm_configuration, tuple) \
            and len(algorithm_configuration) == 2 \
            and isinstance(algorithm_configuration[0], TimeSeriesAnomalyDetector) \
            and isinstance(algorithm_configuration[1], str):
        return {algorithm_configuration[1]: (algorithm_configuration[0], {})}

    if type(algorithm_configuration) is str:
        configuration_file = open(algorithm_configuration, 'r')
        algorithm_configuration = json.load(configuration_file)
        configuration_file.close()

    # Fill in the template parameters
    algorithm_configuration = translate(algorithm_configuration)

    if 'algorithm_configurations' in algorithm_configuration:
        algorithms = {}
        sub_dir = algorithm_configuration['collection'] if 'collection' in algorithm_configuration else ''
        for config in algorithm_configuration['algorithm_configurations']:
            algorithm, name, used_config = read_algorithm_configuration(config)
            algorithms[f'{sub_dir}/{name}'] = (algorithm, used_config)

    else:
        algorithm, name, used_config = read_algorithm_configuration(algorithm_configuration)
        algorithms = {name: (algorithm, used_config)}

    return algorithms


def read_algorithm_configuration(algorithm_configuration) -> Tuple[TimeSeriesAnomalyDetector, str, Dict[str, Any]]:

    # Read the algorithm configuration file if it is a string
    if type(algorithm_configuration) is str:
        configuration_file = open(algorithm_configuration, 'r')
        algorithm_configuration = json.load(configuration_file)
        configuration_file.close()

    # Load the specific anomaly detector class
    module_path = algorithm_configuration['module_path'] if 'module_path' in algorithm_configuration else 'dtaianomaly.anomaly_detection'
    anomaly_detector_class_object: TimeSeriesAnomalyDetector = getattr(importlib.import_module(module_path), algorithm_configuration['anomaly_detector'])

    # Load and return the specific anomaly detector instance, with the given hyperparameters
    return anomaly_detector_class_object.load(parameters=algorithm_configuration['hyperparameters']), algorithm_configuration['name'], algorithm_configuration


def translate(input_config):
    config = {key: value for key, value in input_config.items() if not key.startswith('template_')}
    if len(config) == len(input_config):
        return input_config  # If there are no templates to fill in

    template_parameters = {key[len('template_'):]: value for key, value in input_config.items() if key.startswith('template_')}
    templates = [{}]
    for parameter, values in template_parameters.items():
        templates = [{**template, parameter: value} for value in values for template in templates]

    def _fill_in_template(template, dictionary):
        filled_in_dictionary = dictionary.copy()
        for key, value in dictionary.items():
            if isinstance(value, dict):
                filled_in_dictionary[key] = _fill_in_template(template, value)
            else:
                for template_key, template_value in template.items():
                    pattern = '{' + str(template_key) + '}'
                    if pattern == value:
                        filled_in_dictionary[key] = template_value
                    elif pattern in value:
                        filled_in_dictionary[key] = value.replace(pattern, str(template_value))

        return filled_in_dictionary

    configurations = []
    for template in templates:
        configuration = _fill_in_template(template, config)
        configuration['name'] = configuration['name'] + '#' + '#'.join([f'{key}={value}' for key, value in template.items()])
        configurations.append(configuration)

    return {
        'collection': config['name'],
        'algorithm_configurations': configurations
    }