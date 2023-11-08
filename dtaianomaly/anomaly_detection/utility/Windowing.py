import numpy as np


class Windowing:

    def __init__(self, window_size: int, stride: int = 1, reduction: str = 'mean'):
        self.__window_size: int = window_size
        self.__stride: int = stride
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

    @property
    def stride(self) -> int:
        return self.__stride

    def create_windows(self, trend_data: np.ndarray) -> np.ndarray:
        # Add a new dimension (of size 1) to the trend data if only one dimension exists
        if len(trend_data.shape) == 1:
            trend_data = trend_data.reshape(-1, 1)

        nb_windows = int(np.ceil((trend_data.shape[0] - self.window_size) / self.stride) + 1)
        windowed_trend_data = np.empty((nb_windows, self.__window_size * trend_data.shape[1]), dtype=trend_data.dtype)
        start_window = 0
        for w in range(nb_windows - 1):
            windowed_trend_data[w, :] = trend_data[start_window:start_window + self.window_size, :].flatten()
            start_window += self.stride
        windowed_trend_data[-1, :] = trend_data[-self.window_size:, :].flatten()
        return windowed_trend_data

    def reverse_windowing(self, scores: np.array, time_series_shape: int) -> np.array:
        scores_time = np.empty(time_series_shape)

        start_window_index = 0
        min_start_window = 0
        end_window_index = 0
        min_end_window = 0
        for t in range(time_series_shape - self.window_size):
            while min_start_window + self.window_size <= t:
                start_window_index += 1
                min_start_window += self.stride
            while t >= min_end_window:
                end_window_index += 1
                min_end_window += self.stride
            scores_time[t] = self.__reduction_function(scores[start_window_index:end_window_index])

        for t in range(time_series_shape - self.window_size, time_series_shape):
            while min_start_window + self.window_size <= t:
                start_window_index += 1
                min_start_window += self.stride
            scores_time[t] = self.__reduction_function(scores[start_window_index:])

        return scores_time
