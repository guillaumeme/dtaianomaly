
import numpy as np
from typing import Optional, Dict

from dtaianomaly.anomaly_detection import TrainType
from dtaianomaly.anomaly_detection.TimeSeriesAnomalyDetector import TimeSeriesAnomalyDetector
from dtaianomaly.anomaly_detection.utility.Windowing import Windowing

try:
    import stumpy
except ImportError:
    raise ImportError("Install 'stumpy' in order to use 'STOMP' anomaly detector!")


class STOMP(TimeSeriesAnomalyDetector):
    """
    Use the STOMP algorithm to detect anomalies in a time series [Zhu2016matrixII]_. STOMP is a fast and scalable algorithm for computing
    the matrix profile, which measures the distance from each sequence to the most similar other sequence. Consequently,
    the matrix profile can be used to quantify how anomalous a subsequence is, because it has a large distance to all
    other subsequences.

    Parameters
    ----------
    window_size : int
        The window size to use for computing the matrix profile.
    reduction : str, default='mean'
        How to reduce the window-based anomaly score to a time-based anomaly score. See also :class:`Windowing <dtaianomaly.anomaly_detection.Windowing>`.
    normalize : bool, default=True
        Whether to z-normalize the time series before computing the matrix profile.
    p : float, default=2.0
        The norm to use for computing the matrix profile.
    k : int, default=1
        The k-th nearest neighbor to use for computing the sequence distance in the matrix profile.

    Example
    -------
    >>> from dtaianomaly.anomaly_detection import STOMP
    >>> from dtaianomaly.data_management import DataGenerator
    >>> time_series = DataGenerator.random_time_series(length=5000, seed=0)
    >>> stomp = STOMP(window_size=64)
    >>> stomp.decision_function(time_series)
    array([8.48835586, 8.48704886, 8.54577643, ..., 8.26465958, 8.22832889,
           8.35702386])

    Note
    ----
    If the given time series is multivariate, the matrix profile is computed for each dimension separately and then
    summed up.

    Reference
    ---------
    .. [Zhu2016matrixII] Y. Zhu et al., “Matrix Profile II: Exploiting a Novel Algorithm and GPUs to Break the One Hundred Million Barrier
                         for Time Series Motifs and Joins,” in 2016 IEEE 16th International Conference on Data Mining (ICDM), Barcelona, Spain:
                         IEEE, Dec. 2016, pp. 739–748. `10.1109/ICDM.2016.0085 <https://ieeexplore.ieee.org/document/7837898>`_.
    """

    def __init__(self,
                 window_size: int,
                 reduction: str = 'mean',
                 normalize: bool = True,
                 p: float = 2.0,
                 k: int = 1):
        super().__init__()
        self.__windowing: Windowing = Windowing(window_size=window_size, stride=1, reduction=reduction)
        self.__normalize: bool = normalize
        self.__p: float = p
        self.__k: int = k

    def train_type(self) -> TrainType:
        return TrainType.UNSUPERVISED

    def _fit(self, trend_data: np.ndarray, labels: Optional[np.array] = None):
        return self  # STOMP does not require any fitting

    def _decision_function(self, trend_data: np.ndarray) -> np.array:
        if trend_data.shape[1] == 1:
            matrix_profile = stumpy.stump(trend_data.squeeze(), m=self.__windowing.window_size, normalize=self.__normalize, p=self.__p, k=self.__k)
            matrix_profile = matrix_profile[:, self.__k - 1]
        else:
            matrix_profiles, _ = stumpy.mstump(trend_data.transpose(), m=self.__windowing.window_size, discords=True, normalize=self.__normalize, p=self.__p)
            matrix_profile = matrix_profiles.sum(axis=0)

        return self.__windowing.reverse_windowing(matrix_profile, trend_data.shape[0])

    @staticmethod
    def load(parameters: Dict[str, any]) -> 'TimeSeriesAnomalyDetector':
        return STOMP(**parameters['hyperparameters'])


def main():
    from dtaianomaly.anomaly_detection import STOMP
    from dtaianomaly.data_management import DataGenerator
    time_series = DataGenerator.random_time_series(length=5000, seed=0)
    stomp = STOMP(window_size=64)
    decision_scores = stomp.decision_function(time_series)
    print(decision_scores)


if __name__ == '__main__':
    main()
