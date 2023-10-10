
import numpy as np
from typing import Optional


def fixed_value_threshold(ground_truth: np.array, scores: np.array, threshold: Optional[float] = None) -> np.array:
    # If no threshold is given, use the ground truth number of anomalies to compute a threshold (which boils down to top_n_threshold)
    if threshold is None:
        return top_n_threshold(ground_truth, scores)

    if threshold <= 0.0:
        return np.ones_like(scores, dtype=np.int16)
    elif threshold >= 1.0:
        return np.zeros_like(scores, dtype=np.int16)
    else:
        return np.array(scores >= threshold, dtype=np.int16)


def contamination_threshold(ground_truth: np.array, scores: np.array, contamination: Optional[float] = None) -> np.array:
    # If no contamination is given, compute the contamination based on the ground truth
    if contamination is None:
        contamination = np.sum(ground_truth) / len(ground_truth)

    if contamination <= 0.0:
        return np.zeros_like(scores, dtype=np.int16)
    elif contamination >= 1.0:
        return np.ones_like(scores, dtype=np.int16)
    else:
        return np.array(scores >= np.quantile(scores, 1 - contamination), dtype=np.int16)


def top_n_threshold(ground_truth: np.array, scores: np.array, top_n: Optional[int] = None) -> np.array:
    # If no n is given, use the ground truth number of anomalies to compute a threshold
    if top_n is None:
        top_n = np.sum(ground_truth)
    return contamination_threshold(ground_truth, scores, contamination=top_n / len(ground_truth))


def top_n_ranges_threshold(ground_truth: np.array, scores: np.array, top_n: Optional[int] = None) -> np.array:
    """
    This function computes a threshold such that there are top_n ranges of anomalies. For this a threshold in the
    given scores is computed. If multiple such thresholds exist, then the smallest threshold is chosen. If no
    threshold exists such that there are top_n ranges, then the (smallest) threshold that gives the most possible
    ranges is chosen. If no **top_n** is given, then the ground truth number of anomalies is used to compute
    the number of ranges.

    :param ground_truth:
    :param scores:
    :param top_n:
    :return:
    """
    if top_n is None:
        top_n = count_nb_ranges(ground_truth)

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
