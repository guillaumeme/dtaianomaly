
import numpy as np
from dtaianomaly.evaluation.Metric import Metric


class RocVUS(Metric):

    def __init__(self, max_window_size: int = 250, num_thresholds: int = 250):
        self.__max_window_size: int = max_window_size
        self.__num_thresholds: int = num_thresholds

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:

        # Check if there are any anomalies in the ground truth
        if np.sum(ground_truth_anomalies) == 0:
            return 0.0

        score_sorted = -np.sort(-predicted_anomaly_scores)
        thresholds = score_sorted[np.linspace(0, predicted_anomaly_scores.shape[0] - 1, num=self.__num_thresholds, dtype=int)]

        auc_3d = 0

        P = np.sum(ground_truth_anomalies)

        for window_size in range(self.__max_window_size + 1):

            labels = extend_postive_range(ground_truth_anomalies, window_size)

            L = range_convers_new(labels)
            TPR_list = [0]
            FPR_list = [0]
            Precision_list = [1]

            for threshold in thresholds:
                pred = (predicted_anomaly_scores >= threshold)
                TPR, FPR, Precision = TPR_FPR_RangeAUC(labels, pred, P, L)

                TPR_list.append(TPR)
                FPR_list.append(FPR)
                Precision_list.append(Precision)

            TPR_list.append(1)
            FPR_list.append(1)  # otherwise, range-AUC will stop earlier than (1,1)

            tpr = np.array(TPR_list)
            fpr = np.array(FPR_list)

            width = fpr[1:] - fpr[:-1]
            height = (tpr[1:] + tpr[:-1]) / 2
            auc_range = np.sum(width * height)
            auc_3d += auc_range

        return auc_3d / (self.__max_window_size + 1)


class PrVUS(Metric):

    def __init__(self, max_window_size: int = 250, num_thresholds: int = 250):
        self.__max_window_size: int = max_window_size
        self.__num_thresholds: int = num_thresholds

    def compute(self, ground_truth_anomalies: np.array, predicted_anomaly_scores: np.array) -> float:

        if np.sum(ground_truth_anomalies) == 0:
            return 0.0

        score_sorted = -np.sort(-predicted_anomaly_scores)
        thresholds = score_sorted[np.linspace(0, predicted_anomaly_scores.shape[0] - 1, num=self.__num_thresholds, dtype=int)]

        ap_3d = 0

        P = np.sum(ground_truth_anomalies)

        for window in range(self.__max_window_size + 1):

            labels = extend_postive_range(ground_truth_anomalies, window)

            L = range_convers_new(labels)
            TPR_list = [0]
            FPR_list = [0]
            Precision_list = [1]

            for threshold in thresholds:
                pred = (predicted_anomaly_scores >= threshold)
                TPR, FPR, Precision = TPR_FPR_RangeAUC(labels, pred, P, L)

                TPR_list.append(TPR)
                FPR_list.append(FPR)
                Precision_list.append(Precision)

            TPR_list.append(1)
            FPR_list.append(1)  # otherwise, range-AUC will stop earlier than (1,1)

            tpr = np.array(TPR_list)
            prec = np.array(Precision_list)

            width_PR = tpr[1:-1] - tpr[:-2]
            height_PR = (prec[1:] + prec[:-1]) / 2
            AP_range = np.sum(width_PR * height_PR)
            ap_3d += AP_range

        return ap_3d / (self.__max_window_size + 1)


# https://github.com/TheDatumOrg/VUS/tree/main original version
def RangeAUC_volume(labels_original, score, windowSize):
    score_sorted = -np.sort(-score)
    thresholds = score_sorted[np.linspace(0, len(score) - 1, 250).astype(int)]

    auc_3d = []
    ap_3d = []

    P = np.sum(labels_original)

    for window in range(windowSize+1):

        labels = extend_postive_range(labels_original, window)

        L = range_convers_new(labels)
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]

        for threshold in thresholds:
            pred = (score >= threshold)
            TPR, FPR, Precision = TPR_FPR_RangeAUC(labels, pred, P, L)

            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)

        TPR_list.append(1)
        FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)

        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)

        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        auc_range = np.sum(width * height)
        auc_3d.append(auc_range)

        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = (prec[1:] + prec[:-1])/2
        AP_range = np.sum(width_PR * height_PR)
        ap_3d.append(AP_range)

    return sum(auc_3d)/(windowSize+1), sum(ap_3d)/(windowSize+1)


def extend_postive_range(x, window=5):
    label = x.copy().astype(float)
    L = range_convers_new(label)  # index of non-zero segments
    length = len(label)
    for k in range(len(L)):
        s = L[k][0]
        e = L[k][1]

        x1 = np.arange(e, min(e + window // 2, length))
        label[x1] += np.sqrt(1 - (x1 - e) / (window))

        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / (window))

    label = np.minimum(np.ones(length), label)
    return label


def range_convers_new(label):
    """
    input: arrays of binary values
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    """
    L = []
    i = 0
    j = 0
    while j < len(label):
        # print(i)
        while label[i] == 0:
            i += 1
            if i >= len(label):
                break
        j = i + 1
        # print('j'+str(j))
        if j >= len(label):
            if j == len(label):
                L.append((i, j - 1))

            break
        while label[j] != 0:
            j += 1
            if j >= len(label):
                L.append((i, j - 1))
                break
        if j >= len(label):
            break
        L.append((i, j - 1))
        i = j
    return L


def TPR_FPR_RangeAUC(labels, pred, P, L):
    product = labels * pred

    TP = np.sum(product)

    P_new = (P + np.sum(labels)) / 2  # so TPR is neither large nor small
    recall = min(TP / P_new, 1)

    existence = 0
    for seg in L:
        if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:
            existence += 1

    existence_ratio = existence / len(L)

    TPR_RangeAUC = recall * existence_ratio

    FP = np.sum(pred) - TP

    N_new = len(labels) - P_new
    FPR_RangeAUC = FP / N_new

    Precision_RangeAUC = TP / np.sum(pred)

    return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC
