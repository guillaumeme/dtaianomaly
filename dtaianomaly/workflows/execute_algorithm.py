
import pandas as pd
from typing import Union, Optional

from dtaianomaly.data_management import DataManager
from dtaianomaly.anomaly_detection.utility.TrainType import TrainType

from dtaianomaly.workflows.handle_data_configuration import DataConfigurationType, handle_data_configuration
from dtaianomaly.workflows.handle_algorithm_configuration import AlgorithmConfigurationType, handle_algorithm_configuration
from dtaianomaly.workflows.handle_metric import MetricConfigurationType, handle_metric_configuration, metric_configuration_to_names

ConfigurationType = Union[DataConfigurationType, AlgorithmConfigurationType, MetricConfigurationType]


def main(data_manager: DataManager,
         data_configuration: DataConfigurationType,
         algorithm_configuration: AlgorithmConfigurationType,
         metric_configuration: MetricConfigurationType,
         results_path: Optional[str] = None) -> pd.DataFrame:

    data_manager = handle_data_configuration(data_manager, data_configuration)
    algorithm = handle_algorithm_configuration(algorithm_configuration)
    algorithm_train_type = algorithm.train_type()
    results = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(data_manager.get(), names=['collection_name', 'dataset_name']),
        columns=metric_configuration_to_names(metric_configuration)
    )

    for dataset_index in data_manager.get():

        meta_data = data_manager.get_metadata(dataset_index)
        if not algorithm_train_type.can_solve_train_type_data(meta_data['train_type']):
            raise Exception(f"Algorithm type '{algorithm_train_type}' can not solve dataset type '{meta_data['train_type']}'!")

        # For supervised algorithms, the ground truth of the train data is required
        if algorithm_train_type == TrainType.SUPERVISED:
            data_train, ground_truth_train = data_manager.load_raw_data(dataset_index, train=True)
            algorithm.fit(data_train, ground_truth_train)

        # For semi-supervised algorithms, train data is required but not its ground truth
        elif algorithm_train_type == TrainType.SEMI_SUPERVISED:
            data_train, _ = data_manager.load_raw_data(dataset_index, train=True)
            algorithm.fit(data_train)

        # For unsupervised algorithms, use the train data to fit, if available, and otherwise use the test data
        else:
            if meta_data['train_type'] == 'unsupervised':
                data_train, _ = data_manager.load_raw_data(dataset_index, train=False)
            else:
                data_train, _ = data_manager.load_raw_data(dataset_index, train=True)
            algorithm.fit(data_train)

        # Execute on the test data
        data_test, ground_truth_test = data_manager.load_raw_data(dataset_index, train=False)
        predicted_proba = algorithm.predict_proba(data_test)

        # Write away the results
        results.loc[dataset_index] = handle_metric_configuration(metric_configuration, predicted_proba, ground_truth_test)

    # Save the results, if requested
    if results_path is not None:
        results.to_csv(results_path)

    return results
