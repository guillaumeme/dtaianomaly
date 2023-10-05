
import numpy as np

from src.anomaly_detection.utility.Windowing import Windowing


class TestWindowing:

    min_window_size = 20
    max_window_size = 500
    nb_reps = 50

    def test_get_window_size(self):
        for random_window_size in np.random.choice(np.arange(self.min_window_size, self.max_window_size + 1), self.nb_reps):
            windowing = Windowing(window_size=random_window_size)
            assert windowing.window_size == random_window_size


