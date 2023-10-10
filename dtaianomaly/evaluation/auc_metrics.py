
import numpy as np
import sklearn


def roc_auc(ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
    return sklearn.metrics.roc_auc_score(ground_truth_anomalies, predicted_anomaly_scores)


def pr_auc(ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
    precision, recall, _ = sklearn.metrics.precision_recall_curve(ground_truth_anomalies, predicted_anomaly_scores)
    return sklearn.metrics.auc(recall, precision)
