
import numpy as np
import sklearn

from dtaianomaly.evaluation.Metric import Metric


class RocAUC(Metric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        # Check if there are any anomalies in the ground truth
        if np.sum(ground_truth_anomalies) == 0:
            return 0.0

        return sklearn.metrics.roc_auc_score(ground_truth_anomalies, predicted_anomaly_scores)


class PrAUC(Metric):

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:
        # Check if there are any anomalies in the ground truth
        if np.sum(ground_truth_anomalies) == 0:
            return 0.0

        precision, recall, _ = sklearn.metrics.precision_recall_curve(ground_truth_anomalies, predicted_anomaly_scores)
        return sklearn.metrics.auc(recall, precision)
