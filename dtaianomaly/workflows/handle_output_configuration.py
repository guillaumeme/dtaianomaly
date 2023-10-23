
import os
import shutil
import json
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
from dtaianomaly.anomaly_detection import TimeSeriesAnomalyDetector

PlainOutputConfiguration = Union[Dict[str, Dict[str, Any]], str]


@dataclass
class OutputConfiguration:

    # The directory where everything should be saved
    directory_path: str
    algorithm_name: str

    # Whether the different intermediate information should be printed or not
    verbose: bool = False

    # Basic algorithm properties
    trace_time: bool = False
    trace_memory: bool = False

    # If the raw results should be saved as a file
    print_results: bool = False
    save_results: bool = False
    results_file: str = 'results.csv'

    # if a figure of the anomaly scores should be saved
    save_anomaly_scores_plot: bool = False
    anomaly_scores_directory: str = 'anomaly_score_plots'
    anomaly_scores_file_format: str = 'svg'
    show_anomaly_scores: str = 'overlay'
    show_ground_truth: Optional[str] = None

    # Raise an error of the train type of the algorithm does not match the train type of a dataset
    invalid_train_type_raise_error: bool = True

    @property
    def directory(self) -> str:
        return f'{self.directory_path}/{self.algorithm_name}'

    @property
    def results_path(self) -> str:
        return f'{self.directory}/{self.results_file}'

    @property
    def figure_directory_path(self) -> str:
        return f'{self.directory}/{self.anomaly_scores_directory}'

    def figure_path(self, dataset_index: str) -> str:
        return f'{self.figure_directory_path}/{dataset_index[0].lower()}_{dataset_index[1].lower()}.{self.anomaly_scores_file_format}'


def handle_output_configuration(plain_output_configuration: Union[PlainOutputConfiguration, OutputConfiguration], algorithm: TimeSeriesAnomalyDetector) -> OutputConfiguration:

    # If a proper output configuration is already given, then use that one
    if type(plain_output_configuration) is OutputConfiguration:
        output_configuration = plain_output_configuration

    # Otherwise, convert the json file or the plain configuration to an output configuration
    else:
        if type(plain_output_configuration) is str:
            configuration_file = open(plain_output_configuration, 'r')
            plain_output_configuration = json.load(configuration_file)
            configuration_file.close()
        output_configuration = OutputConfiguration(**plain_output_configuration, algorithm_name=algorithm.name)

    # Create the directory if it does not exist yet
    os.makedirs(output_configuration.directory, exist_ok=True)

    # Create a directory for the anomaly score plots if it does not exist yet, and clear it otherwise
    if os.path.exists(output_configuration.figure_directory_path):
        shutil.rmtree(output_configuration.figure_directory_path)
    os.mkdir(output_configuration.figure_directory_path)

    return output_configuration
