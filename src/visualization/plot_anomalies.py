
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Union


def plot_data(
        trend_data: pd.DataFrame,
        axs: Optional[List[plt.Axes]] = None,
        file_path: Optional[str] = None,
        show_ground_truth: str = 'none') -> Union[plt.Figure, List[plt.Axes]]:

    # Check if valid axes are provided
    if axs is not None and len(axs) != trend_data.shape[1] - 1:
        raise ValueError("Number of axes must match number of attributes!")

    # Check if valid value is provided for 'show_ground_truth'
    if show_ground_truth not in ['none', 'inline', 'background']:
        raise ValueError("Parameter 'show_ground_truth' must be one of 'none', 'inline', or 'background'!")

    # Identify the attributes and set up the axis
    attributes = trend_data.columns.drop('is_anomaly')
    if axs is None:
        return_fig = True
        fig, axs = plt.subplots(nrows=len(attributes), sharex='all')
        axs = [axs] if len(attributes) == 1 else axs
    else:
        fig = None  # To avoid warning when saving the figure
        return_fig = False

    # Compute the ranges where anomalies occur
    anomaly_ranges = np.where(np.diff(trend_data['is_anomaly'], prepend=0) != 0)[0].reshape(-1, 2)

    # Plot each attribute
    for i in range(len(attributes)):
        attribute = attributes[i]
        trend_data[attribute].plot(ax=axs[i], ylabel=attribute)

        # Plot the ground truth
        if show_ground_truth != 'none':
            for start, end in anomaly_ranges:
                if show_ground_truth == 'background':
                    axs[i].axvspan(start, end, color='red', label='Anomaly')
                elif show_ground_truth == 'inline':
                    axs[i].plot(range(start, end), trend_data.iloc[start:end, :][attribute], color='red')

    # Save the figure if requested
    if axs is not None and file_path is not None:
        fig.savefig(file_path)

    return fig if return_fig else axs


def plot_anomaly_labels(trend_data: pd.DataFrame, anomaly_labels: np.array) -> None:
    pass


def plot_anomaly_scores(trend_data: pd.DataFrame, anomaly_scores: np.array) -> None:
    pass


def main():
    from src.data_management.DataManager import DataManager
    from src.workflows.handle_data_configuration import handle_data_configuration
    from src.workflows.handle_algorithm_configuration import handle_algorithm_configuration
    from src.evaluation.thresholding import contamination_threshold

    data_manager = DataManager('../../data')
    data_manager = handle_data_configuration(data_manager, '../../experiments/default_configurations/data/CalIt2.json')
    dataset_index = data_manager.get()[0]
    data_df = data_manager.load(dataset_index)
    trend_data, ground_truth = data_manager.load_raw_data(dataset_index)

    anomaly_detector = handle_algorithm_configuration('../../experiments/default_configurations/algorithm/knn.json')

    anomaly_detector.fit(trend_data)
    anomaly_scores = anomaly_detector.decision_function(trend_data)
    anomaly_labels = contamination_threshold(ground_truth, anomaly_scores, 0.1)

    fig = plot_data(data_df, show_ground_truth='inline')
    fig.show()


if __name__ == "__main__":
    main()
