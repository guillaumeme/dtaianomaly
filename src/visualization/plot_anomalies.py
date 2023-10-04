
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict


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
        trend_data[attribute].plot(ax=axs[i], title=attribute)

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


def plot_anomaly_scores(
        trend_data: pd.DataFrame,
        anomaly_scores: Union[np.array, Dict[str, np.array]],
        file_path: Optional[str] = None,
        show_ground_truth: str = 'none') -> plt.Figure:

    # Check if valid value is provided for 'show_ground_truth'
    if show_ground_truth not in ['none', 'compare', 'inline', 'background']:
        raise ValueError("Parameter 'show_ground_truth' must be one of 'none', 'compare', 'inline', or 'background'!")

    # Format the anomaly scores to a dict with as key the label to show in the plot
    formatted_anomaly_scores = {}
    if type(anomaly_scores) == dict:
        for key, value in anomaly_scores.items():
            formatted_anomaly_scores['Anomaly scores ' + key] = value
    else:
        formatted_anomaly_scores = {'Anomaly scores': anomaly_scores}

    # Plot the data
    fig, axs = plt.subplots(nrows=trend_data.shape[1] - 1 + len(formatted_anomaly_scores), sharex='all')
    plot_data(trend_data, axs[:trend_data.shape[1] - 1], show_ground_truth='none' if show_ground_truth == 'compare' else show_ground_truth)

    axs_counter = trend_data.shape[1] - 1
    for label, specific_anomaly_scores in formatted_anomaly_scores.items():
        # Plot the anomaly scores
        axs[axs_counter].set_title(label)
        axs[axs_counter].plot(specific_anomaly_scores, color='red', label='Predicted')
        if show_ground_truth == 'compare':
            axs[axs_counter].plot(trend_data['is_anomaly'], color='green', label='Ground truth')
            axs[axs_counter].legend()
        # Increment the counter
        axs_counter += 1

    # Save the figure if requested
    if file_path is not None:
        fig.savefig(file_path)

    return fig
