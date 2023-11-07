
import json
from typing import Dict, Any, Union, Callable

from dtaianomaly.evaluation.Metric import Metric
from dtaianomaly.evaluation.affiliation_metrics import AffiliationPrecision, AffiliationRecall, AffiliationFBeta
from dtaianomaly.evaluation.auc_metrics import RocAUC, PrAUC
from dtaianomaly.evaluation.classification_metrics import Precision, Recall, Fbeta
from dtaianomaly.evaluation.vus_metrics import RocVUS, PrVUS
from dtaianomaly.evaluation.thresholding import Thresholding, FixedValueThresholding, ContaminationThresholding, TopNThresholding, TopNRangesThresholding

MetricConfiguration = Union[Dict[str, Dict[str, Any]], str]


# Functions with parameters: ground truth anomalies, predicted anomaly scores, metric parameters, thresholding function
__SUPPORTED_METRICS: Dict[str, Callable[[Dict[str, Any], Thresholding], Metric]] = {
    # Affiliation metrics
    'affiliation_precision': lambda _, thresholding: AffiliationPrecision(thresholding),
    'affiliation_recall': lambda _, thresholding: AffiliationRecall(thresholding),
    'affiliation_fbeta': lambda metric_parameters, thresholding: AffiliationFBeta(thresholding, **metric_parameters),

    # Area under the curve metrics (do not require thresholding)
    'roc_auc': lambda _, __: RocAUC(),
    'pr_auc': lambda _, __: PrAUC(),

    # Basic metrics
    'precision': lambda _, thresholding: Precision(thresholding),
    'recall': lambda _, thresholding: Recall(thresholding),
    'fbeta': lambda metric_parameters, thresholding: Fbeta(thresholding, **metric_parameters),

    # Volume under the surface metrics
    'roc_vus': lambda metric_parameters, _: RocVUS(**metric_parameters),
    'pr_vus': lambda metric_parameters, _: PrVUS(**metric_parameters)
}
__METRICS_WITHOUT_THRESHOLDING = ['roc_auc', 'pr_auc', 'roc_vus', 'pr_vus']

# Return a function that takes as input the ground truth anomalies and the predicted anomaly scores, and returns the labels of the predicted anomalies
__SUPPORTED_THRESHOLDING: Dict[str, Callable[[Dict[str, Any]], Thresholding]] = {
    'fixed_value': lambda params: FixedValueThresholding(**params),
    'contamination': lambda params: ContaminationThresholding(**params),
    'top_n': lambda params: TopNThresholding(**params),
    'top_n_ranges': lambda params: TopNRangesThresholding(**params)
}


def handle_metric_configuration(metric_configuration: MetricConfiguration) -> Dict[str, Metric]:

    # Read the metric configuration file if it is a string
    if type(metric_configuration) is str:
        configuration_file = open(metric_configuration, 'r')
        metric_configuration = json.load(configuration_file)
        configuration_file.close()

    # Initialize the metrics
    results = {}
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
            thresholding = None
        else:
            if 'thresholding_strategy' not in metric_parameters:
                raise ValueError(f"The metric parameters do not contain a 'thresholding_strategy' property for entry with key '{metric}'!\n"
                                 f"Given metric parameters: {metric_parameters}")
            if metric_configuration[metric]['thresholding_strategy'] not in __SUPPORTED_THRESHOLDING:
                raise ValueError(f"The given thresholding '{metric_parameters['thresholding_strategy']}' is not supported yet, or is not a valid thresholding!\n"
                                 f"Supported thresholding methods are: {__SUPPORTED_THRESHOLDING.keys()}")

            thresholding_parameters = metric_parameters['thresholding_parameters'] if 'thresholding_parameters' in metric_parameters else {}
            thresholding = __SUPPORTED_THRESHOLDING[metric_parameters['thresholding_strategy']](thresholding_parameters)

        # Compute the specific metric
        metric_parameters = metric_configuration[metric]['metric_parameters'] if 'metric_parameters' in metric_configuration[metric] else {}
        results[metric] = __SUPPORTED_METRICS[metric_name](metric_parameters, thresholding)

    return results
