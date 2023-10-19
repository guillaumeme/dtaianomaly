
import numpy as np
import sklearn

from dtaianomaly.evaluation.Metric import Metric


class RocAUC(Metric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        return sklearn.metrics.roc_auc_score(ground_truth_anomalies, predicted_anomaly_scores)


class PrAUC(Metric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        precision, recall, _ = sklearn.metrics.precision_recall_curve(ground_truth_anomalies, predicted_anomaly_scores)
        return sklearn.metrics.auc(recall, precision)
