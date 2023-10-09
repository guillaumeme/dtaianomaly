
import pytest
import numpy as np

from src.anomaly_detection.utility.Windowing import Windowing
from src.data_management.DataManager import DataManager


class TestWindowing:

    min_window_size = 20
    max_window_size = 500
    nb_reps = 50


class TestWindowingWindowSize(TestWindowing):

    def test_get_window_size(self):
        for random_window_size in np.random.choice(np.arange(self.min_window_size, self.max_window_size + 1), self.nb_reps):
            windowing = Windowing(window_size=random_window_size)
            assert windowing.window_size == random_window_size


class TestCreateWindows(TestWindowing):

    def test_univariate_small_window(self):
        # Setup
        time_series = np.arange(10).reshape(-1, 1)
        assert len(time_series.shape) == 2
        assert time_series.shape[1] == 1
        # Create windows
        windowing = Windowing(window_size=3)
        windows = windowing.create_windows(time_series)
        # Check size
        assert windows.shape[0] == 8
        for window in windows:
            assert window.shape[0] == 3
        # Check content
        assert np.array_equal(windows[0], np.array([0, 1, 2]))
        assert np.array_equal(windows[1], np.array([1, 2, 3]))
        assert np.array_equal(windows[2], np.array([2, 3, 4]))
        assert np.array_equal(windows[3], np.array([3, 4, 5]))
        assert np.array_equal(windows[4], np.array([4, 5, 6]))
        assert np.array_equal(windows[5], np.array([5, 6, 7]))
        assert np.array_equal(windows[6], np.array([6, 7, 8]))
        assert np.array_equal(windows[7], np.array([7, 8, 9]))

    def test_univariate_bigger_window(self):
        # Setup
        time_series = np.arange(3 * (self.max_window_size + 1)).reshape(-1, 1)
        assert len(time_series.shape) == 2
        assert time_series.shape[1] == 1
        for random_window_size in np.random.choice(np.arange(self.min_window_size, self.max_window_size + 1), self.nb_reps):
            # Create windows
            windowing = Windowing(window_size=random_window_size)
            windows = windowing.create_windows(time_series)
            # Check size
            assert windows.shape[0] == time_series.shape[0] - random_window_size + 1
            for window in windows:
                assert window.shape[0] == random_window_size
            # Check content
            for i in range(time_series.shape[0] - random_window_size + 1):
                assert np.array_equal(windows[i], np.arange(i, i + random_window_size))

    def test_univariate_single_dimension(self):
        # Setup
        time_series = np.arange(10)
        assert len(time_series.shape) == 1
        # Create windows
        windowing = Windowing(window_size=3)
        windows = windowing.create_windows(time_series)
        # Check size
        assert windows.shape[0] == 8
        for window in windows:
            assert window.shape[0] == 3
        # Check content
        assert np.array_equal(windows[0], np.array([0, 1, 2]))
        assert np.array_equal(windows[1], np.array([1, 2, 3]))
        assert np.array_equal(windows[2], np.array([2, 3, 4]))
        assert np.array_equal(windows[3], np.array([3, 4, 5]))
        assert np.array_equal(windows[4], np.array([4, 5, 6]))
        assert np.array_equal(windows[5], np.array([5, 6, 7]))
        assert np.array_equal(windows[6], np.array([6, 7, 8]))
        assert np.array_equal(windows[7], np.array([7, 8, 9]))

    def test_multivariate_2d_small_window(self):
        # Setup
        time_series = np.array([np.arange(10), np.arange(10)*10]).T
        assert len(time_series.shape) == 2
        assert time_series.shape[1] == 2
        # Create windows
        windowing = Windowing(window_size=3)
        windows = windowing.create_windows(time_series)
        # Check size
        assert windows.shape[0] == 8
        for window in windows:
            assert len(window.shape) == 1
            assert window.shape[0] == 3 * 2
        # Check content
        assert np.array_equal(windows[0], np.array([0, 0, 1, 10, 2, 20]))
        assert np.array_equal(windows[1], np.array([1, 10, 2, 20, 3, 30]))
        assert np.array_equal(windows[2], np.array([2, 20, 3, 30, 4, 40]))
        assert np.array_equal(windows[3], np.array([3, 30, 4, 40, 5, 50]))
        assert np.array_equal(windows[4], np.array([4, 40, 5, 50, 6, 60]))
        assert np.array_equal(windows[5], np.array([5, 50, 6, 60, 7, 70]))
        assert np.array_equal(windows[6], np.array([6, 60, 7, 70, 8, 80]))
        assert np.array_equal(windows[7], np.array([7, 70, 8, 80, 9, 90]))

    def test_multivariate_2d_bigger_window(self):
        # Setup
        time_series = np.array([np.arange(3 * (self.max_window_size + 1)), np.arange(3 * (self.max_window_size + 1)) * 10]).T
        assert len(time_series.shape) == 2
        assert time_series.shape[1] == 2
        for random_window_size in np.random.choice(np.arange(self.min_window_size, self.max_window_size + 1), self.nb_reps):
            # Create windows
            windowing = Windowing(window_size=random_window_size)
            windows = windowing.create_windows(time_series)
            # Check size
            assert windows.shape[0] == time_series.shape[0] - random_window_size + 1
            for window in windows:
                assert window.shape[0] == random_window_size*2
            # Check content
            for i in range(time_series.shape[0] - random_window_size + 1):
                for t in range(random_window_size):
                    assert windows[i][2*t] == i + t
                    assert windows[i][2*t + 1] == (i + t) * 10

    def test_multivariate_more_than_2d(self):
        dimensions = [3, 4, 5, 6]
        for dimension in dimensions:
            # Setup
            time_series = np.array([
                np.arange(3 * (self.max_window_size + 1)) * d ** 2 for d in range(dimension)
            ]).T
            assert len(time_series.shape) == 2
            assert time_series.shape[1] == dimension
            for random_window_size in np.random.choice(np.arange(self.min_window_size, self.max_window_size + 1), self.nb_reps // len(dimensions)):
                # Create windows
                windowing = Windowing(window_size=random_window_size)
                windows = windowing.create_windows(time_series)
                # Check size
                assert windows.shape[0] == time_series.shape[0] - random_window_size + 1
                for window in windows:
                    assert window.shape[0] == random_window_size * dimension
                # Check content
                for i in range(time_series.shape[0] - random_window_size + 1):
                    for t in range(random_window_size):
                        for d in range(dimension):
                            assert windows[i][dimension * t + d] == (i + t) * d**2

    def test_loaded_data(self):
        data_manager = DataManager('data')
        data_manager.select()
        dataset_index = data_manager.get(0)
        trend_data, ground_truth = data_manager.load_raw_data(dataset_index)
        windowing = Windowing(window_size=16)
        _ = windowing.create_windows(trend_data)
        # Nothing went wrong


class TestReverseWindowing(TestWindowing):

    def test_mean_reduction_small(self):
        scores = np.arange(8)
        windowing = Windowing(window_size=3, reduction='mean')
        reverse_windowing = windowing.reverse_windowing(scores)
        assert len(reverse_windowing.shape) == 1
        assert reverse_windowing.shape[0] == 10
        assert reverse_windowing[0] == 0    # [0]
        assert reverse_windowing[1] == 0.5  # [0, 1]
        assert reverse_windowing[2] == 1    # [0, 1, 2]
        assert reverse_windowing[3] == 2    # [1, 2, 3]
        assert reverse_windowing[4] == 3    # [2, 3, 4]
        assert reverse_windowing[5] == 4    # [3, 4, 5]
        assert reverse_windowing[6] == 5    # [4, 5, 6]
        assert reverse_windowing[7] == 6    # [5, 6, 7]
        assert reverse_windowing[8] == 6.5  # [6, 7]
        assert reverse_windowing[9] == 7    # [7]

    def test_mean_reduction_small2(self):
        # Mean and median is the for np.arange
        scores = 2**np.arange(8)
        print(scores)
        windowing = Windowing(window_size=3, reduction='mean')
        reverse_windowing = windowing.reverse_windowing(scores)
        assert len(reverse_windowing.shape) == 1
        assert reverse_windowing.shape[0] == 10
        assert reverse_windowing[0] == 1    # [1]
        assert reverse_windowing[1] == 1.5   # [1, 2]
        assert reverse_windowing[2] == pytest.approx(7/3)   # [1, 2, 4]
        assert reverse_windowing[3] == pytest.approx(14/3)    # [2, 4, 8]
        assert reverse_windowing[4] == pytest.approx(28/3)    # [4, 8, 16]
        assert reverse_windowing[5] == pytest.approx(56/3)   # [8, 16, 32]
        assert reverse_windowing[6] == pytest.approx(112/3)   # [16, 32, 64]
        assert reverse_windowing[7] == pytest.approx(224/3)   # [32 64, 128]
        assert reverse_windowing[8] == 96   # [64, 128]
        assert reverse_windowing[9] == 128  # [128]

    def test_median_reduction_small(self):
        scores = np.arange(8)
        windowing = Windowing(window_size=3, reduction='median')
        reverse_windowing = windowing.reverse_windowing(scores)
        assert len(reverse_windowing.shape) == 1
        assert reverse_windowing.shape[0] == 10
        assert reverse_windowing[0] == 0    # [0]
        assert reverse_windowing[1] == 0.5  # [0, 1]
        assert reverse_windowing[2] == 1    # [0, 1, 2]
        assert reverse_windowing[3] == 2    # [1, 2, 3]
        assert reverse_windowing[4] == 3    # [2, 3, 4]
        assert reverse_windowing[5] == 4    # [3, 4, 5]
        assert reverse_windowing[6] == 5    # [4, 5, 6]
        assert reverse_windowing[7] == 6    # [5, 6, 7]
        assert reverse_windowing[8] == 6.5  # [6, 7]
        assert reverse_windowing[9] == 7    # [7]

    def test_median_reduction_small2(self):
        # Mean and median is the for np.arange
        scores = 2**np.arange(8)
        print(scores)
        windowing = Windowing(window_size=3, reduction='median')
        reverse_windowing = windowing.reverse_windowing(scores)
        assert len(reverse_windowing.shape) == 1
        assert reverse_windowing.shape[0] == 10
        assert reverse_windowing[0] == 1    # [1]
        assert reverse_windowing[1] == 1.5  # [1, 2]
        assert reverse_windowing[2] == 2    # [1, 2, 4]
        assert reverse_windowing[3] == 4    # [2, 4, 8]
        assert reverse_windowing[4] == 8    # [4, 8, 16]
        assert reverse_windowing[5] == 16   # [8, 16, 32]
        assert reverse_windowing[6] == 32   # [16, 32, 64]
        assert reverse_windowing[7] == 64   # [32 64, 128]
        assert reverse_windowing[8] == 96   # [64, 128]
        assert reverse_windowing[9] == 128  # [128]

    def test_max_reduction_small(self):
        scores = np.arange(8)
        windowing = Windowing(window_size=3, reduction='max')
        reverse_windowing = windowing.reverse_windowing(scores)
        assert len(reverse_windowing.shape) == 1
        assert reverse_windowing.shape[0] == 10
        assert reverse_windowing[0] == 0    # [0]
        assert reverse_windowing[1] == 1    # [0, 1]
        assert reverse_windowing[2] == 2    # [0, 1, 2]
        assert reverse_windowing[3] == 3    # [1, 2, 3]
        assert reverse_windowing[4] == 4    # [2, 3, 4]
        assert reverse_windowing[5] == 5    # [3, 4, 5]
        assert reverse_windowing[6] == 6    # [4, 5, 6]
        assert reverse_windowing[7] == 7    # [5, 6, 7]
        assert reverse_windowing[8] == 7    # [6, 7]
        assert reverse_windowing[9] == 7    # [7]

    def test_sum_reduction_small(self):
        scores = np.arange(8)
        windowing = Windowing(window_size=3, reduction='sum')
        reverse_windowing = windowing.reverse_windowing(scores)
        assert len(reverse_windowing.shape) == 1
        assert reverse_windowing.shape[0] == 10
        assert reverse_windowing[0] == 0    # [0]
        assert reverse_windowing[1] == 1    # [0, 1]
        assert reverse_windowing[2] == 3    # [0, 1, 2]
        assert reverse_windowing[3] == 6    # [1, 2, 3]
        assert reverse_windowing[4] == 9    # [2, 3, 4]
        assert reverse_windowing[5] == 12   # [3, 4, 5]
        assert reverse_windowing[6] == 15   # [4, 5, 6]
        assert reverse_windowing[7] == 18   # [5, 6, 7]
        assert reverse_windowing[8] == 13   # [6, 7]
        assert reverse_windowing[9] == 7    # [7]


class TestWindowingReduction(TestWindowing):

    def test_invalid_reduction(self):
        with pytest.raises(ValueError):
            _ = Windowing(window_size=16, reduction='invalid_reduction')
        with pytest.raises(ValueError):
            _ = Windowing(window_size=16, reduction='average')
        with pytest.raises(ValueError):
            _ = Windowing(window_size=16, reduction='min')
        with pytest.raises(ValueError):
            _ = Windowing(window_size=16, reduction='something-random')

    def test_mean_reduction(self):
        _ = Windowing(window_size=16, reduction='mean')

    def test_median_reduction(self):
        _ = Windowing(window_size=16, reduction='median')

    def test_max_reduction(self):
        _ = Windowing(window_size=16, reduction='max')

    def test_sum_reduction(self):
        _ = Windowing(window_size=16, reduction='sum')
