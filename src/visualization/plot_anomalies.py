
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict


def plot_data(
        trend_data: pd.DataFrame,
        file_path: Optional[str] = None,
        show_ground_truth: Optional[str] = None) -> plt.Figure:

    # Check if valid value is provided for 'show_ground_truth'
    if show_ground_truth not in [None, 'overlay', 'background']:
        raise ValueError("Parameter 'show_ground_truth' must be one of None, 'overlay', or 'background'!")

    # Identify the attributes and set up the axis
    attributes = trend_data.columns.drop('is_anomaly')
    fig, axs = plt.subplots(nrows=len(attributes), sharex='all')
    axs = [axs] if len(attributes) == 1 else axs

    # Plot each attribute
    for i in range(len(attributes)):
        attribute = attributes[i]
        if show_ground_truth is None:
            _plot_data(axs[i], trend_data[attribute])
        elif show_ground_truth == 'overlay':
            _plot_data_with_anomaly_overlay(axs[i], trend_data[attribute], trend_data['is_anomaly'].values)
        elif show_ground_truth == 'background':
            _add_anomalies_as_background(axs[i], trend_data['is_anomaly'])
            _plot_data(axs[i], trend_data[attribute])

    # Save the figure if requested
    if file_path is not None:
        fig.savefig(file_path)

    return fig


def plot_anomaly_scores(
        trend_data: pd.DataFrame,
        anomaly_scores: Union[np.array, Dict[str, np.array]],
        file_path: Optional[str] = None,
        show_anomaly_scores: str = 'separate',
        show_ground_truth: Optional[str] = None) -> plt.Figure:

    # Check if valid value is provided for 'show_anomaly_scores'
    if show_anomaly_scores not in ['separate', 'overlay']:
        raise ValueError("Parameter 'show_anomaly_scores' must be one of 'separate' or 'overlay'!")

    # Check if valid value is provided for 'show_ground_truth'
    if show_ground_truth not in [None, 'compare', 'overlay', 'background']:
        raise ValueError("Parameter 'show_ground_truth' must be one of None, 'compare', 'overlay', or 'background'!")

    # Check compatibilities of showing anomaly scores and ground truth
    if show_ground_truth == 'compare' and show_anomaly_scores == 'overlay':
        raise ValueError("'show_ground_truth' with value 'compare' and 'show_anomaly_scores' with value 'overlay' are incompatible!"
                         "Cannot show anomaly scores as overlay when comparing to ground truth!")
    if show_ground_truth == 'overlay' and show_anomaly_scores == 'overlay':
        raise ValueError("'show_ground_truth' with value 'overlay' and 'show_anomaly_scores' with value 'overlay' are incompatible!"
                         "Cannot show both ground truth and anomaly score as a overlay!")
    if type(anomaly_scores) == dict and show_anomaly_scores == 'overlay':
        raise ValueError("Cannot show anomaly scores as overlay when multiple anomaly scores are provided!")

    # Format the anomaly scores to a dict with as key the label to show in the plot
    formatted_anomaly_scores = anomaly_scores
    if type(anomaly_scores) != dict:
        formatted_anomaly_scores = {'Anomaly scores': anomaly_scores}

    # Set up the figure
    nb_rows = trend_data.shape[1] - 1
    nb_rows += len(formatted_anomaly_scores) if show_anomaly_scores == 'separate' else 0
    fig, axs = plt.subplots(nrows=nb_rows, sharex='all')
    axs = [axs] if nb_rows == 1 else axs

    # Plot each attribute
    attributes = trend_data.columns.drop('is_anomaly')
    for i in range(len(attributes)):
        attribute = attributes[i]
        if show_anomaly_scores == 'overlay':
            if show_ground_truth == 'background':
                _add_anomalies_as_background(axs[i], trend_data['is_anomaly'])
            _plot_data_with_anomaly_overlay(axs[i], trend_data[attribute], list(formatted_anomaly_scores.values())[0])
        elif show_ground_truth is None:
            _plot_data(axs[i], trend_data[attribute])
        elif show_ground_truth == 'overlay':
            _plot_data_with_anomaly_overlay(axs[i], trend_data[attribute], trend_data['is_anomaly'].values)
        elif show_ground_truth == 'background':
            _add_anomalies_as_background(axs[i], trend_data['is_anomaly'])
            _plot_data(axs[i], trend_data[attribute])

    # Plot the anomaly separate, if requested
    if show_anomaly_scores == 'separate':
        axs_counter = len(attributes)
        for label, specific_anomaly_scores in formatted_anomaly_scores.items():
            axs[axs_counter].set_title(label)
            if show_ground_truth == 'compare':
                axs[axs_counter].plot(specific_anomaly_scores, color='red', label='Predicted')
                axs[axs_counter].plot(trend_data['is_anomaly'], color='green', label='Ground truth')
                axs[axs_counter].legend()
            else:
                axs[axs_counter].plot(specific_anomaly_scores)
            # Increment the counter
            axs_counter += 1

    # Save the figure if requested
    if file_path is not None:
        fig.savefig(file_path)

    return fig


def _plot_data(ax: plt.Axes, trend_data_attribute: pd.Series) -> None:
    trend_data_attribute.plot(ax=ax, title=trend_data_attribute.name)


def _add_anomalies_as_background(ax: plt.Axes, anomaly_labels: pd.Series) -> None:
    anomaly_ranges = np.where(np.diff(anomaly_labels, prepend=0) != 0)[0].reshape(-1, 2)
    for start, end in anomaly_ranges:
        # Mark the point before and after the anomaly as well
        start_inclusive = start - 1
        end_inclusive = end + 1
        ax.axvspan(start_inclusive, end_inclusive, color='red', label='Anomaly', alpha=0.2)


def _plot_data_with_anomaly_overlay(ax: plt.Axes, trend_data_attribute: pd.Series, anomaly_scores: np.array, nb_bins=100) -> None:
    normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    binned_anomaly_scores = [np.floor(score * nb_bins) / nb_bins for score in normalized_scores]
    colormap = plt.get_cmap('RdYlGn', nb_bins).reversed()
    for i in range(0, trend_data_attribute.shape[0]):
        color = colormap(binned_anomaly_scores[i])
        ax.plot(trend_data_attribute.loc[i:i + 2], c=color)
