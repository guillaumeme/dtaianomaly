import matplotlib.pyplot as plt
import numpy as np


def plot_time_series_colored_by_score(X: np.ndarray, y: np.ndarray, ax: plt.Axes = None, nb_colors: int = 100, **kwargs) -> plt.Figure:
    """
    Plots the given time series, and color it according to the given scores.
    Higher scores will be colored red, and lower scores will be colored green.
    Thus, if the ground truth anomaly scores are passed, red corresponds to
    anomalies and green to normal observations.

    Parameters
    ----------
    X: np.ndarray of shape (n_samples, n_attributes)
        The time series to plot
    y: np.ndarray of shape (n_samples)
        The scores, according to which the plotted data should be colored.
    ax: plt.Axes, default=None
        The axes onto which the plot should be made. If None, then a new
        figure and axis will be created.
    nb_colors: int, default=100
        The number of colors to use for plotting the time series.
    **kwargs:
        Arguments to be passed to plt.Figure(), in case ``ax=None``.

    Returns
    -------
    fig: plt.Figure
        The figure containing the plotted data.

    Notes
    -----
    Each segment in the time series will be plotted independently. Thus,
    for time series with many observations, plotting the data using this
    method can cost a huge amount of time.
    """
    if ax is None:
        plt.figure(**kwargs)
        ax = plt.gca()
    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
    y_binned = [np.floor(score * nb_colors) / nb_colors for score in y_scaled]
    colormap = plt.get_cmap('RdYlGn', nb_colors).reversed()
    for i in range(0, X.shape[0]-1):
        color = colormap(y_binned[i])
        ax.plot([i, i+1], X[[i, i+1]], c=color)
    return plt.gcf()
