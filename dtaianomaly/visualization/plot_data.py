
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict


def plot_data(
        trend_data: pd.DataFrame,
        file_path: Optional[str] = None,
        show_ground_truth: Optional[str] = None) -> plt.Figure:
    """
    Visualizes the given trend data. For multivariate data, each attribute is plotted
    in a different axis.

    Parameters
    ----------
    trend_data : pd.DataFrame
        The trend data to visualize. This is a pandas dataframe, in which each
        row represents an observation, as returned by :py:class:`~dtaianomaly.data_management.DataManager`.
        This is a dataframe with as index the time stamps, a column ``'is_anomaly'``
        representing the anomaly labels, and the remaining columns containing the
        trend data itself.
    file_path : str, optional, default=None
        If provided, the resulting figure will be saved to this path. Otherwise, the
        file will not be saved.
    show_ground_truth : str, optional, default=None
        How to visualize the ground truth. Possible values:

            - ``None`` (default). Do not show the ground truth.

            - ``'overlay'``. Colour the anomalous segments in red and the normal segments in green.
              This option is slow for large time series, because a segment is drawn for each pair of
              consecutive observations.

            - ``'background'``. Add a coloured shade behind the plotted data at the anomalous regions.

    Raises
    ------
    ValueError
        If ``show_ground_truth`` is not a valid value.

    Returns
    -------
    plt.Figure
        The figure containing the visualization of the trend data.
    """
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
    """
    Plot the given anomaly scores for the given trend data.

    Parameters
    ----------
    trend_data : pd.DataFrame
        The trend data to visualize. This is a pandas dataframe, in which each
        row represents an observation, as returned by :py:class:`~dtaianomaly.data_management.DataManager`.
        This is a dataframe with as index the time stamps, a column ``'is_anomaly'``
        representing the anomaly labels, and the remaining columns containing the
        trend data itself.
    anomaly_scores : Union[np.array, Dict[str, np.array]]
        The anomaly scores to plot. If a dictionary is given, then multiple anomaly
        scores will be plotted (the values) which will be indicated by the corresponding
        key.
    file_path : str, optional, default=None
        If provided, the resulting figure will be saved to this path. Otherwise, the
        file will not be saved.
    show_anomaly_scores : str, default='separate'
        How to visualize the ground truth. Possible values:

            - ``'separate'`` (default). Show the anomaly scores in a separate axis. If multiple
              anomaly scores are given, a new axis will be added for each score.

            - ``'compare'``. Is equivalent to ``'separate'`` if only one score is given. If
              multiple scores are given, then only a single axis will be added, on which all
              the anomaly scores are plotted.

            - ``'overlay'``. Colour the time series with a gradient according to the anomaly
              score. The segments that are anomalous will be more red, while the normal segments
              will be more green. This option is slow for large time series, because a segment is
              drawn for each pair of consecutive observations.

    show_ground_truth : str, optional, default=None
        How to visualize the ground truth. Possible values:

            - ``None`` (default). Do not show the ground truth.

            - ``'compare'``. Add the ground truth to the separate axis which also contains the anomaly scores.

            - ``'overlay'``. Colour the anomalous segments in red and the normal segments in green. This option
              is slow for large time series, because a segment is drawn for each pair of consecutive observations.

            - ``'background'``. Add a coloured shade behind the plotted data at the anomalous regions.

    Raises
    ------
    ValueError
        - If ``show_anomaly_scores`` is not a valid value.
        - If ``show_ground_truth`` is not a valid value.
        - If ``show_ground_truth`` is ``'compare'``  and ``'show_anomaly_scores'`` is ``'overlay'``
          because it is not possible to compare with an overlay.
        - If ``show_ground_truth`` is ``'overlay'`` and ``'show_anomaly_scores'`` is ``'overlay'``
          because both scores can not be overlayed at the same time.
        - If ``'show_anomaly_scores'`` is ``'overlay'`` and ``anomaly_scores`` is a dictionary
          because it is not possible to overlay multiple anomaly scores simultaneously.

    Returns
    -------
    plt.Figure
        The figure containing the visualization of the trend data.
    """
    # Check if valid value is provided for 'show_anomaly_scores'
    if show_anomaly_scores not in ['separate', 'compare', 'overlay']:
        raise ValueError("Parameter 'show_anomaly_scores' must be one of 'separate', 'compare', or 'overlay'!")

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
    nb_rows += 1 if show_anomaly_scores == 'compare' else 0
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
        else:
            _plot_data(axs[i], trend_data[attribute])

    # Plot the anomaly separate, if requested
    if show_anomaly_scores == 'separate':
        axs_counter = len(attributes)
        for label, specific_anomaly_scores in formatted_anomaly_scores.items():
            axs[axs_counter].set_title(label)
            if show_ground_truth == 'compare':
                axs[axs_counter].plot(trend_data.index, specific_anomaly_scores, color='red', label='Predicted')
                axs[axs_counter].plot(trend_data['is_anomaly'], color='green', label='Ground truth')
                axs[axs_counter].legend()
            else:
                axs[axs_counter].plot(trend_data.index, specific_anomaly_scores)
            # Increment the counter
            axs_counter += 1

    if show_ground_truth == 'compare':
        for label, specific_anomaly_scores in formatted_anomaly_scores.items():
            axs[-1].plot(trend_data.index, specific_anomaly_scores, label=label)
        if show_ground_truth == 'compare':
            axs[-1].plot(trend_data['is_anomaly'], label='Ground truth')
        axs[-1].set_title('Anomaly scores')
        axs[-1].legend()

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
        start_inclusive = anomaly_labels.index[start - 1]
        end_inclusive = anomaly_labels.index[end + 1]
        ax.axvspan(start_inclusive, end_inclusive, color='red', label='Anomaly', alpha=0.2)


def _plot_data_with_anomaly_overlay(ax: plt.Axes, trend_data_attribute: pd.Series, anomaly_scores: np.array, nb_bins=100) -> None:
    if np.max(anomaly_scores) > np.min(anomaly_scores):
        normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    else:
        normalized_scores = np.zeros_like(anomaly_scores)
    binned_anomaly_scores = [np.floor(score * nb_bins) / nb_bins for score in normalized_scores]
    ax.set_title(str(trend_data_attribute.name))
    colormap = plt.get_cmap('RdYlGn', nb_bins).reversed()
    for i in range(0, trend_data_attribute.shape[0]):
        color = colormap(binned_anomaly_scores[i])
        # +2 because the second index is exclusive (and we want the closed interval [i, i+1])
        ax.plot(trend_data_attribute.iloc[i:i + 2], c=color)
