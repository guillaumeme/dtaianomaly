import numpy as np


class Windowing:

    def __init__(self, window_size: int, reduction: str = 'mean'):
        self.__window_size: int = window_size
        if reduction == 'mean':
            self.__reduction_function = np.mean
        elif reduction == 'median':
            self.__reduction_function = np.median
        elif reduction == 'max':
            self.__reduction_function = np.max
        elif reduction == 'sum':
            self.__reduction_function = np.sum
        else:
            raise ValueError(f"Reduction strategy '{reduction}' is invalid for reverse windowing! Allowed strategies are 'mean', 'median', 'max', and 'sum'.")

    @property
    def window_size(self) -> int:
        return self.__window_size

    def create_windows(self, trend_data: np.ndarray) -> np.ndarray:
        # Add a new dimension (of size 1) to the trend data if only one dimension exists
        if len(trend_data.shape) == 1:
            trend_data = trend_data.reshape(-1, 1)

        nb_windows = trend_data.shape[0] - self.__window_size + 1
        windowed_trend_data = np.empty((nb_windows, self.__window_size * trend_data.shape[1]), dtype=trend_data.dtype)
        for w in range(nb_windows):
            windowed_trend_data[w, :] = trend_data[w:w + self.__window_size, :].flatten()
        return windowed_trend_data

    def reverse_windowing(self, scores: np.array) -> np.array:
        total_time = scores.shape[0] + self.__window_size - 1
        scores_time = np.empty(total_time)

        # The first time points are covered by less than |self.__window_size| windows
        for t in range(self.__window_size):
            scores_time[t] = self.__reduction_function(scores[:t + 1])
        # The middle time points are covered by exactly |self.__window_size| windows
        for t in range(self.__window_size, total_time - self.__window_size + 1):
            scores_time[t] = self.__reduction_function(scores[t - self.__window_size + 1:t + 1])
        # And, the last time points are also covered by less than |self.__window_size| windows
        for t in range(total_time - self.__window_size + 1, total_time):
            scores_time[t] = self.__reduction_function(scores[t - self.__window_size + 1:])

        return scores_time
