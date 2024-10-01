
import numpy as np


def sliding_window(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Constructs a sliding window for the given time series.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_attributes)
        The time series
    window_size: int
        The window size for the sliding windows.
    stride: int
        The stride, i.e., the step size for the windows.

    Returns
    -------
    windows: np.ndarray of shape ((n_samples - window_size)/stride + 1, n_attributes * window_size)
        The windows as a 2D numpy array. Each row corresponds to a
        window. For windows of multivariate time series are flattened
        to form a 1D array of length the number of attributes multiplied
        by the window size.
    """
    windows = [X[t:t+window_size].ravel() for t in range(0, X.shape[0] - window_size, stride)]
    windows.append(X[-window_size:].ravel())
    return np.array(windows)


def reverse_sliding_window(per_window_anomaly_scores: np.ndarray, window_size: int, stride: int, length_time_series: int) -> np.ndarray:
    """
    Reverses the sliding window, to convert the per-window anomaly
    scores into per-observation anomaly scores.

    For non-overlapping sliding windows, it is trivial to convert
    the per-window anomaly scores to per-observation scores, because
    each observation is linked to only one window. For overlapping
    windows, certain observations are linked to one or more windows
    (depending on the window size and stride), obstructing simply
    copying the corresponding per-window anomaly score to each window.
    In the case of multiple overlapping windows, the anomaly score
    of the observation is set to the mean of the corresponding
    per-window anomaly scores.

    Parameters
    ----------
    per_window_anomaly_scores: array-like of shape (n_windows)
    window_size: int
        The window size used for creating windows
    stride: int
        The stride, i.e., the step size used for creating windows
    length_time_series: int
        The original length of the time series.

    Returns
    -------
    anomaly_scores: np.ndarray of shape (length_time_series)
        The per-observation anomaly scores.
    """
    # Convert to array
    scores_time = np.empty(length_time_series)

    start_window_index = 0
    min_start_window = 0
    end_window_index = 0
    min_end_window = 0
    for t in range(length_time_series - window_size):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        while t >= min_end_window:
            end_window_index += 1
            min_end_window += stride
        scores_time[t] = np.mean(per_window_anomaly_scores[start_window_index:end_window_index])

    for t in range(length_time_series - window_size, length_time_series):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        scores_time[t] = np.mean(per_window_anomaly_scores[start_window_index:])

    return scores_time
