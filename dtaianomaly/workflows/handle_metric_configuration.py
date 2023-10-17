
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

from dtaianomaly.evaluation.auc_metrics import roc_auc, pr_auc
from dtaianomaly.evaluation.classification_metrics import precision, recall, fbeta
from dtaianomaly.evaluation.thresholding import fixed_value_threshold, contamination_threshold, top_n_threshold, top_n_ranges_threshold

MetricConfiguration = Union[Dict[str, Dict[str, Any]], str]


# Functions with parameters: ground truth anomalies, predicted anomaly scores, metric parameters, thresholding function
__SUPPORTED_METRICS = {
    # Area under the curve metrics (do not require thresholding)
    'roc_auc': lambda ground_truth_anomalies, predicted_anomaly_scores, _, __:
        roc_auc(ground_truth_anomalies, predicted_anomaly_scores),
    'pr_auc': lambda ground_truth_anomalies, predicted_anomaly_scores, _, __:
        pr_auc(ground_truth_anomalies, predicted_anomaly_scores),

    # Basic metrics
    'precision': lambda ground_truth_anomalies, predicted_anomaly_scores, _, thresholding:
        precision(ground_truth_anomalies, thresholding(ground_truth_anomalies, predicted_anomaly_scores)),
    'recall': lambda ground_truth_anomalies, predicted_anomaly_scores, _, thresholding:
        recall(ground_truth_anomalies, thresholding(ground_truth_anomalies, predicted_anomaly_scores)),
    'fbeta': lambda ground_truth_anomalies, predicted_anomaly_scores, metric_parameters, thresholding:
        fbeta(ground_truth_anomalies, thresholding(ground_truth_anomalies, predicted_anomaly_scores), **metric_parameters)
}
__METRICS_WITHOUT_THRESHOLDING = ['roc_auc', 'pr_auc']

# Return a function that takes as input the ground truth anomalies and the predicted anomaly scores, and returns the labels of the predicted anomalies
__SUPPORTED_THRESHOLDING = {
    'fixed_value': lambda params: (lambda ground_truth_anomalies, predicted_anomaly_scores: fixed_value_threshold(ground_truth_anomalies, predicted_anomaly_scores, **params)),
    'contamination': lambda params: (lambda ground_truth_anomalies, predicted_anomaly_scores: contamination_threshold(ground_truth_anomalies, predicted_anomaly_scores, **params)),
    'top_n': lambda params: (lambda ground_truth_anomalies, predicted_anomaly_scores: top_n_threshold(ground_truth_anomalies, predicted_anomaly_scores, **params)),
    'top_n_ranges': lambda params: (lambda ground_truth_anomalies, predicted_anomaly_scores: top_n_ranges_threshold(ground_truth_anomalies, predicted_anomaly_scores, **params))
}


def handle_metric_configuration(metric_configuration: MetricConfiguration, predicted_proba: np.array, ground_truth: np.array) -> pd.Series:

    # Read the metric configuration file if it is a string
    if type(metric_configuration) is str:
        configuration_file = open(metric_configuration, 'r')
        metric_configuration = json.load(configuration_file)
        configuration_file.close()

    # Compute the metrics
    results = pd.Series(index=metric_configuration_to_names(metric_configuration))
    for metric, metric_parameters in metric_configuration.items():

        # Check if the given metric is supported
        if metric in __SUPPORTED_METRICS:
            metric_name = metric
        else:
            if 'metric_name' not in metric_parameters:
                raise ValueError(f"The metric parameters do not contain a 'metric_name' property for entry with key '{metric}'!\n"
                                 f"Given metric parameters: {metric_parameters}")
            if metric_parameters['metric_name'] not in __SUPPORTED_METRICS:
                raise ValueError(f"The given metric '{metric_parameters['metric_name']}' is not supported yet, or is not a valid metric!\n"
                                 f"Supported metrics are: {__SUPPORTED_METRICS.keys()}")
            metric_name = metric_parameters['metric_name']

        # Check if the given metric requires thresholding (and raise an error if no thresholding information is given if required)
        if metric_name in __METRICS_WITHOUT_THRESHOLDING:
            thresholding_function = None
        else:
            if 'thresholding_strategy' not in metric_parameters:
                raise ValueError(f"The metric parameters do not contain a 'thresholding_strategy' property for entry with key '{metric}'!\n"
                                 f"Given metric parameters: {metric_parameters}")
            if metric_configuration[metric]['thresholding_strategy'] not in __SUPPORTED_THRESHOLDING:
                raise ValueError(f"The given thresholding '{metric_parameters['thresholding_strategy']}' is not supported yet, or is not a valid thresholding!\n"
                                 f"Supported thresholding methods are: {__SUPPORTED_THRESHOLDING.keys()}")

            thresholding_parameters = metric_parameters['thresholding_parameters'] if 'thresholding_parameters' in metric_parameters else {}
            thresholding_function = __SUPPORTED_THRESHOLDING[metric_parameters['thresholding_strategy']](thresholding_parameters)

        # Compute the specific metric
        metric_parameters = metric_configuration[metric]['metric_parameters'] if 'metric_parameters' in metric_configuration[metric] else {}
        results[metric] = __SUPPORTED_METRICS[metric_name](ground_truth, predicted_proba, metric_parameters, thresholding_function)

    return results


def metric_configuration_to_names(metric_configuration: MetricConfiguration) -> List[str]:
    if type(metric_configuration) is str:
        configuration_file = open(metric_configuration, 'r')
        metric_configuration = json.load(configuration_file)
        configuration_file.close()

    metric_names = []
    for metric, metric_parameters in metric_configuration.items():
        if metric in __SUPPORTED_METRICS:
            metric_names.append(metric)
        else:
            metric_names.append(metric_parameters['metric_name'])
    return list(metric_configuration.keys())
