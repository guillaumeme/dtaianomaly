
import numpy as np
import sklearn


def precision(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return sklearn.metrics.precision_score(ground_truth_anomalies, predicted_anomalies)


def recall(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return sklearn.metrics.recall_score(ground_truth_anomalies, predicted_anomalies)


def f1(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return sklearn.metrics.f1_score(ground_truth_anomalies, predicted_anomalies)
