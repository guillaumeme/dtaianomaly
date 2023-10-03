import unittest
import numpy as np

from src.evaluation.thresholding import count_nb_ranges


class TestCountConsecutiveOnes(unittest.TestCase):

    def test_empty_list(self):
        self.assertEqual(count_nb_ranges(np.array([])), 0)

    def test_no_ones(self):
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 0])), 0)

    def test_single_consecutive_ones(self):
        self.assertEqual(count_nb_ranges(np.array([0, 1, 0, 0, 0, 0])), 1)
        self.assertEqual(count_nb_ranges(np.array([0, 1, 0, 0, 1, 0, 0])), 2)
        self.assertEqual(count_nb_ranges(np.array([0, 1, 0, 0, 1, 0, 0, 1])), 3)

    def test_multiple_consecutive_ones(self):
        self.assertEqual(count_nb_ranges(np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0])), 3)
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0])), 2)
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])), 1)

    def test_consecutive_ones_at_start(self):
        self.assertEqual(count_nb_ranges(np.array([1, 1, 0, 0, 0, 1, 0])), 2)
        self.assertEqual(count_nb_ranges(np.array([0, 1, 0, 0, 0, 1, 0])), 2)
        self.assertEqual(count_nb_ranges(np.array([1, 0, 0, 0, 0, 1, 0])), 2)
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 0, 1, 0])), 1)

        self.assertEqual(count_nb_ranges(np.array([1, 1, 0, 0, 0, 0, 0])), 1)
        self.assertEqual(count_nb_ranges(np.array([0, 1, 0, 0, 0, 0, 0])), 1)
        self.assertEqual(count_nb_ranges(np.array([1, 0, 0, 0, 0, 0, 0])), 1)
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 0])), 0)

    def test_consecutive_ones_at_end(self):
        self.assertEqual(count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 1, 1])), 2)
        self.assertEqual(count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 1, 0])), 2)
        self.assertEqual(count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 0, 1])), 2)
        self.assertEqual(count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 0, 0])), 1)

        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 1, 1])), 1)
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 1, 0])), 1)
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 0, 1])), 1)
        self.assertEqual(count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 0, 0])), 0)








