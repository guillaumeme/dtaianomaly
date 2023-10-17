
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score


def precision(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return precision_score(ground_truth_anomalies, predicted_anomalies)


def recall(ground_truth_anomalies: np.array, predicted_anomalies: np.array) -> float:
    return recall_score(ground_truth_anomalies, predicted_anomalies)


def fbeta(ground_truth_anomalies: np.array, predicted_anomalies: np.array, beta: float = 1.0) -> float:
    return fbeta_score(ground_truth_anomalies, predicted_anomalies, beta=beta)
