
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def precision(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return precision_score(ground_truth_anomalies, predicted_anomalies)


def recall(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return recall_score(ground_truth_anomalies, predicted_anomalies)


def f1(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return f1_score(ground_truth_anomalies, predicted_anomalies)
