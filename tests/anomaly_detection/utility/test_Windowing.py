
import unittest
import numpy as np

from src.anomaly_detection.utility.Windowing import Windowing


class TestWindowing(unittest.TestCase):

    def test_windowing_window_size(self):
        min_window_size = 20
        max_window_size = 500
        nb_reps = 50
        for random_window_size in np.random.choice(np.arange(min_window_size, max_window_size + 1), nb_reps):
            windowing = Windowing(window_size=random_window_size)
            self.assertEqual(windowing.window_size, random_window_size)
