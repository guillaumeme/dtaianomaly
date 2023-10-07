
import pytest
import numpy as np

from src.evaluation.thresholding import fixed_value_threshold, contamination_threshold, top_n_threshold, top_n_ranges_threshold, count_nb_ranges


class TestFixedValueThreshold:

    def test_invalid_threshold(self):
        scores = np.random.uniform(size=1000)
        assert np.all(fixed_value_threshold(np.ones_like(scores), scores, threshold=-1) == 1)
        assert np.all(fixed_value_threshold(np.ones_like(scores), scores, threshold=10) == 0)

    def test_threshold_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        assert np.array_equal(
            fixed_value_threshold(ground_truth, scores, threshold=0.0),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        assert np.array_equal(
            fixed_value_threshold(ground_truth, scores, threshold=0.1),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        assert np.array_equal(
            fixed_value_threshold(ground_truth, scores, threshold=0.15),
            np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]))
        assert np.array_equal(
            fixed_value_threshold(ground_truth, scores, threshold=0.5),
            np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(
            fixed_value_threshold(ground_truth, scores, threshold=0.8),
            np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(
            fixed_value_threshold(ground_truth, scores, threshold=1.0),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    def test_threshold_not_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        labels_without_threshold = fixed_value_threshold(ground_truth, scores)
        labels_with_threshold = fixed_value_threshold(ground_truth, scores, threshold=0.7)  # 4 anomalies in ground truth, and the 4th highest score is 0.7
        assert np.array_equal(labels_without_threshold, np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(labels_without_threshold, labels_with_threshold)
        assert np.sum(labels_without_threshold) == 4

    def test_type(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        for threshold in np.arange(21) / 20:
            assert fixed_value_threshold(ground_truth, scores, threshold=threshold).dtype == int


class TestContaminationThreshold:

    def test_invalid_contamination(self):
        scores = np.random.uniform(size=1000)
        assert np.all(contamination_threshold(np.ones_like(scores), scores, contamination=-1) == 0)
        assert np.all(contamination_threshold(np.ones_like(scores), scores, contamination=10) == 1)

    def test_contamination_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        assert np.array_equal(
            contamination_threshold(ground_truth, scores, contamination=0.0),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert np.array_equal(
            contamination_threshold(ground_truth, scores, contamination=1/15),
            np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(
            contamination_threshold(ground_truth, scores, contamination=6/15),
            np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]))
        assert np.array_equal(
            contamination_threshold(ground_truth, scores, contamination=9/15),
            np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]))
        assert np.array_equal(
            contamination_threshold(ground_truth, scores, contamination=15/15),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        assert np.array_equal(
            contamination_threshold(ground_truth, scores, contamination=1.0),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    def test_contamination_not_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        labels_without_contamination = contamination_threshold(ground_truth, scores)
        labels_with_contamination = contamination_threshold(ground_truth, scores, contamination=4/15)
        assert np.array_equal(labels_without_contamination, np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(labels_without_contamination, labels_with_contamination)
        assert np.sum(labels_without_contamination) == 4

    def test_nb_anomalies(self):
        scores = np.random.uniform(size=1000)
        for contamination in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            assert np.mean(contamination_threshold(np.ones_like(scores), scores, contamination=contamination)) == pytest.approx(contamination, abs=1e-3)

    def test_type(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        for contamination in np.arange(21) / 20:
            assert contamination_threshold(ground_truth, scores, contamination=contamination).dtype == int


class TestTopNThreshold:

    def test_invalid_n(self):
        scores = np.random.uniform(size=1000)
        assert np.all(top_n_threshold(np.ones_like(scores), scores, top_n=-1) == 0)
        assert np.all(top_n_threshold(np.ones_like(scores), scores, top_n=scores.shape[0]+1) == 1)

    def test_n_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=0),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=1),
            np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=2),
            np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=4),
            np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=6),
            np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]))
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=9),
            np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]))
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=12),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        assert np.array_equal(
            top_n_threshold(ground_truth, scores, top_n=15),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    def test_n_not_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        labels_without_n = top_n_threshold(ground_truth, scores)
        labels_with_n = top_n_threshold(ground_truth, scores, top_n=4)
        assert np.array_equal(labels_without_n, np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]))
        assert np.array_equal(labels_without_n, labels_with_n)
        assert np.sum(labels_without_n) == 4

    def test_nb_anomalies(self):
        scores = np.random.uniform(size=1000)
        for n in range(0, scores.shape[0] + 1):
            assert np.sum(top_n_threshold(np.ones_like(scores), scores, top_n=n)) >= n

    def test_type(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        for top_n in range(0, len(scores) + 1):
            assert top_n_threshold(ground_truth, scores, top_n=top_n).dtype == int


class TestTopNRangesThreshold:

    def test_invalid_n(self):
        scores = np.random.uniform(size=1000)
        assert np.all(top_n_threshold(np.ones_like(scores), scores, top_n=-1) == 0)
        assert np.all(top_n_threshold(np.ones_like(scores), scores, top_n=scores.shape[0] + 1) == 1)

    def test_n_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        assert np.array_equal(
            top_n_ranges_threshold(ground_truth, scores, top_n=0),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        # The only threshold that exists with one range is 0.1
        assert np.array_equal(
            top_n_ranges_threshold(ground_truth, scores, top_n=1),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        # No threshold exists with two ranges, thus a threshold is chosen such that there are fewer ranges, which means only 1 range
        assert np.array_equal(
            top_n_ranges_threshold(ground_truth, scores, top_n=2),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        assert np.array_equal(
            top_n_ranges_threshold(ground_truth, scores, top_n=3),
            np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]))
        assert np.array_equal(
            top_n_ranges_threshold(ground_truth, scores, top_n=4),
            np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]))
        # No threshold exists such that there are 5 or more ranges, thus the maximum number of ranges equals 4
        for n in range(5, scores.shape[0]):
            assert np.array_equal(
                top_n_ranges_threshold(ground_truth, scores, top_n=n),
                top_n_ranges_threshold(ground_truth, scores, top_n=4))

    def test_n_not_given(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        labels_without_n = top_n_ranges_threshold(ground_truth, scores)
        labels_with_n = top_n_ranges_threshold(ground_truth, scores, top_n=3)
        assert np.array_equal(labels_without_n, np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]))
        assert np.array_equal(labels_without_n, labels_with_n)
        assert count_nb_ranges(labels_without_n) == 3

    def test_nb_anomalies(self):
        scores = np.random.uniform(size=1000)
        for n in range(0, scores.shape[0] + 1, 50):
            assert count_nb_ranges(top_n_ranges_threshold(np.ones_like(scores), scores, top_n=n)) <= n

    def test_type(self):
        ground_truth = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 0.8, 0.4])
        for top_n in range(0, len(scores) + 1):
            assert top_n_ranges_threshold(ground_truth, scores, top_n=top_n).dtype == int


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








