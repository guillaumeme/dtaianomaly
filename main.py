
import argparse
import json

from dtaianomaly.workflow import execute_algorithm
from dtaianomaly.data_management import DataManager


if __name__ == '__main__':

    # Create a parser and parse the arguments
    parser = argparse.ArgumentParser(description="Time series anomaly detection.")
    parser.add_argument('--seed', default=0, type=int,
                        help='The seed to set before detecting anomalies in every time series.')
    parser.add_argument('--n_jobs', default=1, type=int,
                        help='The number of jobs that to run in parallel, in which each job corresponds to '
                             'detecting anomalies in a single time series')
    parser.add_argument('--datasets_index_file', default='datasets.csv',
                        help='The path to the dataset index file containing metadata about the datasets.')
    parser.add_argument('--configuration_dir', default='configurations',
                        help='The directory containing the configuration files. The provided configurations '
                             'should be relative to this path.')
    parser.add_argument('--config',
                        help='The name of the configuration file for the experiment. This file'
                             'contains the paths to the configuration files for the data, the '
                             'algorithm, the metrics and the output. '
                             '(More information at https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/experiments.html)')
    parser.add_argument('--data',
                        help="The path of the configuration file regarding which data to read. "
                             "Ignored if the '--config' parameter is given. "
                             "(More information at https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/experiments.html)")
    parser.add_argument('--algorithm',
                        help="The path of the configuration file regarding the anomaly detector to load. "
                             "Ignored if the '--config' parameter is given. "
                             "(More information at https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/experiments.html)")
    parser.add_argument('--metric',
                        help="The path of the configuration file regarding the evaluation metrics. "
                             "Ignored if the '--config' parameter is given. "
                             "(More information at https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/experiments.html)")
    parser.add_argument('--output',
                        help="The path of the configuration file regarding the output of the workflow. "
                             "Ignored if the '--config' parameter is given. "
                             "(More information at https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/experiments.html)")
    args = parser.parse_args()

    # Variables for the different configurations
    data_configuration = None
    algorithm_configuration = None
    metric_configuration = None
    output_configuration = None

    # Check if a single configuration file was given
    if args.config is not None:
        configuration_file = open(args.configuration_dir + '/' + args.config, 'r')
        configuration = json.load(configuration_file)
        configuration_file.close()

        if 'data' in configuration:
            data_configuration = configuration['data']
        if 'algorithm' in configuration:
            algorithm_configuration = configuration['algorithm']
        if 'metric' in configuration:
            metric_configuration = configuration['metric']
        if 'output' in configuration:
            output_configuration = configuration['output']

    # Overwrite the configurations if specific values were given
    if args.data is not None:
        data_configuration = args.data
    if args.algorithm is not None:
        algorithm_configuration = args.algorithm
    if args.metric is not None:
        metric_configuration = args.metric
    if args.output is not None:
        output_configuration = args.output

    # Check if the configurations were given
    if data_configuration is None:
        raise ValueError("No data configuration was given!")
    if algorithm_configuration is None:
        raise ValueError("No algorithm configuration was given!")
    if metric_configuration is None:
        raise ValueError("No metric configuration was given!")
    if output_configuration is None:
        raise ValueError("No output configuration was given!")

    # Set the path correctly, if the configuration is a string (aka a path to a .json file)
    if type(data_configuration) is str:
        data_configuration = args.configuration_dir + '/' + data_configuration
    if type(algorithm_configuration) is str:
        algorithm_configuration = args.configuration_dir + '/' + algorithm_configuration
    if type(metric_configuration) is str:
        metric_configuration = args.configuration_dir + '/' + metric_configuration
    if type(output_configuration) is str:
        output_configuration = args.configuration_dir + '/' + output_configuration

    # Execute the algorithm
    execute_algorithm(
        data_manager=DataManager(args.datasets_index_file),
        data_configuration=data_configuration,
        algorithm_configuration=algorithm_configuration,
        metric_configuration=metric_configuration,
        output_configuration=output_configuration,
        seed=args.seed,
        n_jobs=args.n_jobs
    )
