import numpy as np

from src.evaluation.thresholding import count_nb_ranges


class TestCountConsecutiveOnes:

    def test_empty_list(self):
        assert count_nb_ranges(np.array([])) == 0

    def test_no_ones(self):
        assert count_nb_ranges(np.array([0, 0, 0, 0, 0])) == 0

    def test_single_consecutive_ones(self):
        assert count_nb_ranges(np.array([0, 1, 0, 0, 0, 0])) == 1
        assert count_nb_ranges(np.array([0, 1, 0, 0, 1, 0, 0])) == 2
        assert count_nb_ranges(np.array([0, 1, 0, 0, 1, 0, 0, 1])) == 3

    def test_multiple_consecutive_ones(self):
        assert count_nb_ranges(np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0])) == 3
        assert count_nb_ranges(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0])) == 2
        assert count_nb_ranges(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])) == 1

    def test_consecutive_ones_at_start(self):
        assert count_nb_ranges(np.array([1, 1, 0, 0, 0, 1, 0])) == 2
        assert count_nb_ranges(np.array([0, 1, 0, 0, 0, 1, 0])) == 2
        assert count_nb_ranges(np.array([1, 0, 0, 0, 0, 1, 0])) == 2
        assert count_nb_ranges(np.array([0, 0, 0, 0, 0, 1, 0])) == 1

        assert count_nb_ranges(np.array([1, 1, 0, 0, 0, 0, 0])) == 1
        assert count_nb_ranges(np.array([0, 1, 0, 0, 0, 0, 0])) == 1
        assert count_nb_ranges(np.array([1, 0, 0, 0, 0, 0, 0])) == 1
        assert count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 0])) == 0

    def test_consecutive_ones_at_end(self):
        assert count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 1, 1])) == 2
        assert count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 1, 0])) == 2
        assert count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 0, 1])) == 2
        assert count_nb_ranges(np.array([0, 1, 1, 0, 0, 0, 0, 0])) == 1

        assert count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 1, 1])) == 1
        assert count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 1, 0])) == 1
        assert count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 0, 1])) == 1
        assert count_nb_ranges(np.array([0, 0, 0, 0, 0, 0, 0, 0])) == 0








