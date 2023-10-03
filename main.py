
import argparse

from src.workflows import execute_algorithm
from src.data_management import DataManager


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Time series anomaly detection.")

    # Parameters regarding the data
    parser.add_argument('--data_dir', dest='data_dir',
                        type=str, default='data',
                        help='The directory containing the datasets and the index file for the datasets')
    parser.add_argument('--datasets_index_file', dest='datasets_index_file',
                        type=str, default='datasets.csv',
                        help='The index file containing metadata about the datasets')

    # Parameter regarding the location of the experiment
    parser.add_argument('--experiment_dir', dest='experiment_dir',
                        type=str, default='experiments',
                        help='The directory containing the experiment files')

    # Parameter regarding the specific experiment to execute
    parser.add_argument('--experiment', dest='experiment',
                        type=str, default='test_univariate_algorithm_execution',
                        help='The name of the experiment to execute')

    # Parse the given arguments
    args = parser.parse_args()

    # Execute the algorithm
    results = execute_algorithm(
        DataManager(args.data_dir, args.datasets_index_file),
        args.experiment_dir + '/default_configurations/data/' + 'CalIt2.json',
        args.experiment_dir + '/default_configurations/algorithm/' + 'knn.json',
        args.experiment_dir + '/default_configurations/metrics/' + 'first_metrics.json'
    )
    print(results)
