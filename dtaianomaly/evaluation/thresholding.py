
import numpy as np
from typing import Optional


import abc


class Thresholding(abc.ABC):

    @abc.abstractmethod
    def apply(self, scores: np.array, ground_truth: np.array) -> np.array:
        raise NotImplementedError("Abstract method ")


class FixedValueThresholding(Thresholding):

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    def apply(self, scores: np.array, ground_truth: np.array):
        if self.threshold is not None:
            threshold = self.threshold
        else:
            nb_ground_truth_anomalies = int(np.sum(ground_truth))
            threshold = np.partition(scores, -nb_ground_truth_anomalies)[-nb_ground_truth_anomalies]

        if threshold <= 0.0:
            return np.ones_like(scores, dtype=np.int16)
        elif threshold >= 1.0:
            return np.zeros_like(scores, dtype=np.int16)
        else:
            return np.array(scores >= threshold, dtype=np.int16)


class ContaminationThresholding(Thresholding):

    def __init__(self, contamination: Optional[float] = None):
        self.contamination = contamination

    def apply(self, scores: np.array, ground_truth: np.array):
        contamination = self.contamination if self.contamination is not None else np.sum(ground_truth) / len(ground_truth)

        if contamination <= 0.0:
            return np.zeros_like(scores, dtype=np.int16)
        elif contamination >= 1.0:
            return np.ones_like(scores, dtype=np.int16)
        else:
            return np.array(scores >= np.quantile(scores, 1 - contamination), dtype=np.int16)


class TopNThresholding(Thresholding):

    def __init__(self, top_n: Optional[int] = None):
        self.top_n = top_n

    def apply(self, scores: np.array, ground_truth: np.array):
        top_n = self.top_n if self.top_n is not None else np.sum(ground_truth)

        if top_n <= 0:
            return np.zeros_like(scores, dtype=np.int16)
        elif top_n >= len(scores):
            return np.ones_like(scores, dtype=np.int16)
        else:
            threshold = np.partition(scores, -top_n)[-top_n]
            return np.array(scores >= threshold, dtype=np.int16)


class TopNRangesThresholding(Thresholding):

    def __init__(self, top_n: Optional[int] = None):
        self.top_n = top_n

    def apply(self, scores: np.array, ground_truth: np.array):
        """
        This function computes a threshold such that there are top_n ranges of anomalies. For this a threshold in the
        given scores is computed. If multiple such thresholds exist, then the smallest threshold is chosen. If no
        threshold exists such that there are top_n ranges, then the (smallest) threshold that gives the most possible
        ranges is chosen. If no **top_n** is given, then the ground truth number of anomalies is used to compute
        the number of ranges.

        :param ground_truth:
        :param scores:
        :return:
        """
        top_n = self.top_n if self.top_n is not None else count_nb_ranges(ground_truth)

        if top_n <= 0:
            return np.zeros_like(scores, dtype=np.int16)
        elif top_n >= len(scores):
            return np.ones_like(scores, dtype=np.int16)
        else:
            thresholds = np.sort(np.unique(scores))
            nb_ranges = np.array([count_nb_ranges(scores >= threshold) for threshold in thresholds])
            index_threshold = np.argmax(np.where(nb_ranges <= top_n, nb_ranges, 0))
            return np.array(scores >= thresholds[index_threshold], dtype=np.int16)


def count_nb_ranges(labels) -> int:
    return np.sum(np.diff(labels, prepend=0) == 1)
