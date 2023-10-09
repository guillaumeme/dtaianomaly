
import numpy as np


class TestEvaluationUtil:

    @staticmethod
    def true_positive(ground_truth: np.array, predicted: np.array) -> int:
        return np.count_nonzero(ground_truth & predicted)

    @staticmethod
    def false_positive(ground_truth: np.array, predicted: np.array) -> int:
        return np.count_nonzero(np.logical_not(ground_truth) & predicted)

    @staticmethod
    def false_negative(ground_truth: np.array, predicted: np.array) -> int:
        return np.count_nonzero(ground_truth & np.logical_not(predicted))

    @staticmethod
    def true_negative(ground_truth: np.array, predicted: np.array) -> int:
        return np.count_nonzero(np.logical_not(ground_truth) & np.logical_not(predicted))
