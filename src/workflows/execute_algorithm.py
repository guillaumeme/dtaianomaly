
import json
import pandas as pd
from typing import Union

from src.data_management import DataManager

from src.workflows.handle_data_configuration import DataConfigurationType, handle_data_configuration
from src.workflows.handle_algorithm_configuration import AlgorithmConfigurationType, handle_algorithm_configuration
from src.workflows.handle_metric import MetricConfigurationType, handle_metric_configuration, metric_configuration_to_names

ConfigurationType = Union[DataConfigurationType, AlgorithmConfigurationType, MetricConfigurationType]


def main(data_manager: DataManager,
         data_configuration: DataConfigurationType,
         algorithm_configuration: AlgorithmConfigurationType,
         metric_configuration: MetricConfigurationType) -> pd.DataFrame:

    data_manager = handle_data_configuration(data_manager, data_configuration)
    algorithm = handle_algorithm_configuration(algorithm_configuration)
    results = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(data_manager.get(), names=['collection_name', 'dataset_name']),
        columns=metric_configuration_to_names(metric_configuration)
    )

    for dataset_index in data_manager.get():
        # TODO what if there is no training data?
        data_train, ground_truth_train = data_manager.load_raw_data(dataset_index, train=False)
        algorithm.fit(data_train, ground_truth_train)

        data_test, ground_truth_test = data_manager.load_raw_data(dataset_index, train=False)
        predicted_proba = algorithm.predict_proba(data_test)

        results.loc[dataset_index] = handle_metric_configuration(metric_configuration, predicted_proba, ground_truth_test)

    return results
