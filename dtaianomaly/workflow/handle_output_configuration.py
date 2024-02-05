
import os
import json
from typing import Dict, Any, Union, Optional, Tuple

PlainOutputConfiguration = Union[Dict[str, Dict[str, Any]], str]


class OutputConfiguration:
    """
    The output configuration, i.e., the information that can be outputted during a workflow.

    Properties
    ----------
    directory_path : str
        The name of the path where all the results should be stored.
    algorithm_name : str
        The name of the algorithm, which is used to store the information in a subdirectory.
    verbose : bool, default = False
        Whether to print intermediate information to the output stream.
    trace_time : bool, default = False
        If the running time of the algorithm should be traced.
    trace_memory : bool, default = False
        If the memory usage of the algorithm should be traced.
    print_results : bool, default = False
        if the final results should be printed to the output stream. Has no effect if ``verbose == False``.
    save_results : bool, default = False
        If the final results should be saved.
    constantly_save_results: bool, default = False
        If the intermediate results (of the algorihtm on each time series) should be saved.
    results_file : str, default = 'results.csv'
        The name for the file containing all the final results.
    save_anomaly_scores_plot : bool, default = False
        Whether a plot with the anomaly scores should be saved for each execution of the algorithm.
    anomaly_scores_plots_directory : str, default = 'anomaly_score_plots'
        The directory (within the algorihtm directory) where the plots should be saved.
    anomaly_scores_plots_file_format : str, default = 'svg'
        The format of the anomaly score plots.
    anomaly_scores_plots_show_anomaly_scores : str, default = 'overlay'
        How to visualize the anomaly scores in the anomaly score plots.
        See :py:func:`dtaianomaly.visualization.plot_anomaly_scores` for more details.
    anomaly_scores_plots_show_ground_truth : str, default = None
        How to visualize the ground truth in the anomaly score plots.
        See :py:func:`dtaianomaly.visualization.plot_anomaly_scores` for more details.
    save_anomaly_scores : bool, default = False
        Whether the raw anomaly scores (not just the quantitative evaluation) should be saved.
    anomaly_scores_directory : str, default = 'anomaly_scores'
        The directory (within the algorithm directory) where the raw anomaly scores should be saved.
    invalid_train_type_raise_error : bool, default = True
        Whether an error should be raised if the training type of the algorithm does not match
        the type of the time series.
    create_fit_predict_error_log : bool, default = True
        Whether an error log should be created for errors occurring during fitting or predicting
        with the anomaly detector.
    reraise_fit_predict_errors : bool, default = True
        If the errors encountered during fitting or predicting should be reraised. This thus
        effectively stops the workflow.

    Note
    ----
    To initialize an :py:class:`~dtaianomaly.worflow.OutputConfiguration`. The exact name above
    must be passed to the ``__init__`` method. This includes the two obligated parameters ``directory_path``
    and ``algorithm_name``, as well as the optional parameters. The order in which these parameters are
    given does not matter. An example is given below.

    >>> from dtaianomaly.workflow import OutputConfiguration
    >>> configuration = OutputConfiguration(
    ...     # First, the obligated parameters
    ...     directory_path='path/to/the/results',
    ...     algorithm_name='my_algorithm',
    ...     # Other, optional parameters
    ...     print_results=True,             # Show the results in the output stream
    ...     save_anomaly_scores_plot=True,  # Plot the anomaly scores
    ...     save_anomaly_scores=True,        # Plot the anomaly scores
    ...     # ...
    ... )
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

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
    constantly_save_results: bool = False
    results_file: str = 'results.csv'

    # If a figure of the anomaly scores should be saved
    save_anomaly_scores_plot: bool = False
    anomaly_scores_plots_directory: str = 'anomaly_score_plots'
    anomaly_scores_plots_file_format: str = 'svg'
    anomaly_scores_plots_show_anomaly_scores: str = 'overlay'
    anomaly_scores_plots_show_ground_truth: Optional[str] = None

    # If the raw anomaly scores should be saved or not
    save_anomaly_scores: bool = False
    anomaly_scores_directory: str = 'anomaly_scores'

    # Raise an error of the train type of the algorithm does not match the train type of the dataset
    invalid_train_type_raise_error: bool = True
    create_fit_predict_error_log: bool = True
    reraise_fit_predict_errors: bool = True

    @property
    def directory(self) -> str:
        return f'{self.directory_path}/{self.algorithm_name}'

    @property
    def results_path(self) -> str:
        return f'{self.directory}/{self.results_file}'

    def intermediate_results_path(self, dataset_index: Tuple[str, str]) -> str:
        return f'{self.directory}/tmp_intermediate_results_{self.dataset_index_to_str(dataset_index)}.csv'

    @property
    def anomaly_score_plots_directory_path(self) -> str:
        return f'{self.directory}/{self.anomaly_scores_plots_directory}'

    def anomaly_score_plot_path(self, dataset_index: Tuple[str, str]) -> str:
        return f'{self.anomaly_score_plots_directory_path}/{self.dataset_index_to_str(dataset_index)}.{self.anomaly_scores_plots_file_format}'

    @property
    def anomaly_scores_directory_path(self) -> str:
        return f'{self.directory}/{self.anomaly_scores_directory}'

    def anomaly_scores_path(self, dataset_index: Tuple[str, str]) -> str:
        return f'{self.anomaly_scores_directory_path}/{self.dataset_index_to_str(dataset_index)}'

    @property
    def error_log_dir(self):
        return f'{self.directory}/errors'

    def error_log_file(self, dataset_index: Tuple[str, str]) -> str:
        return f'{self.error_log_dir}/error_{self.dataset_index_to_str(dataset_index)}.txt'

    @staticmethod
    def dataset_index_to_str(dataset_index: Tuple[str, str]) -> str:
        return f'{dataset_index[0].lower()}_{dataset_index[1].lower()}'


def handle_output_configuration(plain_output_configuration: Union[PlainOutputConfiguration, OutputConfiguration], algorithm_name: str) -> OutputConfiguration:

    # If a proper output configuration is already given, then use that one
    if type(plain_output_configuration) is OutputConfiguration:
        output_configuration = plain_output_configuration
        output_configuration.algorithm_name = algorithm_name

    # Otherwise, convert the json file or the plain configuration to an output configuration
    else:
        if type(plain_output_configuration) is str:
            configuration_file = open(plain_output_configuration, 'r')
            plain_output_configuration = json.load(configuration_file)
            configuration_file.close()

        output_configuration = OutputConfiguration(**plain_output_configuration, algorithm_name=algorithm_name)

    # Create the directory if it does not exist yet
    os.makedirs(output_configuration.directory, exist_ok=True)

    # Create a directory for the anomaly score plots, if they should be saved
    if output_configuration.save_anomaly_scores_plot and not os.path.exists(output_configuration.anomaly_score_plots_directory_path):
        os.mkdir(output_configuration.anomaly_score_plots_directory_path)

    # Create a directory for the anomaly scores, if they should be saved
    if output_configuration.save_anomaly_scores and not os.path.exists(output_configuration.anomaly_scores_directory_path):
        os.mkdir(output_configuration.anomaly_scores_directory_path)

    # Create a directory for the error logs, if they should be created
    if output_configuration.create_fit_predict_error_log and not os.path.exists(output_configuration.error_log_dir):
        os.mkdir(output_configuration.error_log_dir)

    return output_configuration
