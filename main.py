
import argparse
import json

from dtaianomaly.workflows import execute_algorithm
from dtaianomaly.data_management import DataManager


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Time series anomaly detection.")

    parser.add_argument('--data_dir', default='data',
                        help='The directory containing the datasets the datasets and the index file for the '
                             'datasets.')
    parser.add_argument('--datasets_index_file', default='datasets.csv',
                        help='The index file containing metadata about the datasets.')
    parser.add_argument('--configuration_dir', default='configurations',
                        help='The directory containing the configuration files. The provided configurations '
                             'should be relative to this path.')

    parser.add_argument('--config',
                        help='The name of the configuration file for the experiment. This file'
                             'contains the paths to the configuration files for the data, the '
                             'algorithm, the metrics and the output. '
                             '(More information at ...)')

    parser.add_argument('--data',
                        help="The path of the configuration file regarding which data to read. Ignored \n"
                             "More information at ... Ignored if the '--config' parameter is given.")
    parser.add_argument('--algorithm',
                        help="The path of the configuration file regarding the anomaly detector to load.\n"
                             "More information at ... Ignored if the '--config' parameter is given.")
    parser.add_argument('--metric',
                        help="The path of the configuration file regarding the evaluation metrics.\n"
                             "More information at ... Ignored if the '--config' parameter is given.")
    parser.add_argument('--output',
                        help="The path of the configuration file regarding the output of the workflow.\n"
                             "More information at ... Ignored if the '--config' parameter is given.")

    args = parser.parse_args()

    if args.config is not None:
        configuration_file = open(args.configuration_dir + args.config, 'r')
        configuration = json.load(configuration_file)
        configuration_file.close()
        data_configuration = configuration['data']
        data_configuration = args.configuration_dir + '/' + data_configuration if type(data_configuration) is str else data_configuration
        algorithm_configuration = configuration['algorithm']
        algorithm_configuration = args.configuration_dir + '/' + algorithm_configuration if type(algorithm_configuration) is str else algorithm_configuration
        metric_configuration = configuration['metric']
        metric_configuration = args.configuration_dir + '/' + metric_configuration if type(metric_configuration) is str else metric_configuration
        output_configuration = configuration['output']
        output_configuration = args.configuration_dir + '/' + output_configuration if type(output_configuration) is str else output_configuration

    else:
        data_configuration = args.configuration_dir + '/' + args.data
        algorithm_configuration = args.configuration_dir + '/' + args.algorithm
        metric_configuration = args.configuration_dir + '/' + args.metric
        output_configuration = args.configuration_dir + '/' + args.output

    # Execute the algorithm
    execute_algorithm(
        DataManager(args.data_dir, args.datasets_index_file),
        data_configuration,
        algorithm_configuration,
        metric_configuration,
        output_configuration
    )
