
import numpy as np
from typing import Optional, Dict

from dtaianomaly.anomaly_detection.TimeSeriesAnomalyDetector import TimeSeriesAnomalyDetector
from dtaianomaly.anomaly_detection.utility.TrainType import TrainType

import math
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from sklearn.preprocessing import MinMaxScaler


_SUPPORTED_TSB_UAD_ANOMALY_DETECTORS = [
    # Default models
    "DAMP", "SAND", "IForest", "LOF", "MatrixProfile", "PCA", "POLY", "OCSVM", "LSTM", "AE", "CNN",
    # Models for which permission should be asked
    "Series2Graph",
    "NORMA",
]


class TSBUADAnomalyDetector(TimeSeriesAnomalyDetector):

    def __init__(self, model: str, window: Optional[int] = None, model_parameters: Optional[Dict[str, any]] = None):
        super().__init__()
        if model not in _SUPPORTED_TSB_UAD_ANOMALY_DETECTORS:
            raise ValueError(f"The given anomaly detector '{model}' is not supported by TSB UAD!\n"
                             f"Supported PYODAnomalyDetectors are: {_SUPPORTED_TSB_UAD_ANOMALY_DETECTORS}")
        self.__model: str = model
        self.__window: Optional[int] = window
        self.__model_parameters: Dict[str, any] = model_parameters or {}

    def train_type(self) -> TrainType:
        return TrainType.UNSUPERVISED

    def _fit(self, trend_data: np.ndarray, labels: Optional[np.array] = None) -> 'TSBUADAnomalyDetector':
        if len(trend_data.shape) > 1 and trend_data.shape[1] > 1:
            raise ValueError('TSB_UAD only accepts univariate time series!')
        return self

    def _decision_function(self, trend_data: np.ndarray) -> np.array:

        if len(trend_data.shape) > 1 and trend_data.shape[1] > 1:
            raise ValueError('TSB_UAD only accepts univariate time series!')

        model = self.__model
        sliding_window = self.__window
        data = trend_data[:, 0] if len(trend_data.shape) > 0 else trend_data

        if model == 'IForest':
            from TSB_UAD.models.iforest import IForest

            if sliding_window:
                X_data = Window(window=sliding_window).convert(data).to_numpy()
            else:
                sliding_window = find_length(data)
                X_data = Window(window=sliding_window).convert(data).to_numpy()

            clf = IForest(**self.__model_parameters)
            x = X_data
            clf.fit(X_data)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

        elif model == 'DAMP':
            from TSB_UAD.models.damp import DAMP

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            clf = DAMP(m=sliding_window, sp_index=sliding_window + 1, **self.__model_parameters)
            x = data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

        elif model == 'SAND':
            from TSB_UAD.models.sand import SAND

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            clf = SAND(pattern_length=sliding_window, subsequence_length=4 * sliding_window)
            x = data
            clf.fit(x, overlaping_rate=int(1.5 * sliding_window))
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        elif model == 'Series2Graph':
            from .series2graph import Series2Graph

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            s2g = Series2Graph(pattern_length=sliding_window, **self.__model_parameters)
            s2g.fit(data)
            query_length = 2 * sliding_window
            s2g.score(query_length=query_length, dataset=data)

            score = s2g.decision_scores_
            score = np.array([score[0]] * math.ceil(query_length // 2) + list(score) + [score[-1]] * (query_length // 2))
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        elif model == 'LOF':
            from TSB_UAD.models.lof import LOF

            if sliding_window:
                X_data = Window(window=sliding_window).convert(data).to_numpy()
            else:
                sliding_window = find_length(data)
                X_data = Window(window=sliding_window).convert(data).to_numpy()

            clf = LOF(**self.__model_parameters)
            x = X_data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

        elif model == 'MatrixProfile':
            from TSB_UAD.models.matrix_profile import MatrixProfile

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            clf = MatrixProfile(window=sliding_window)
            x = data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

        elif model == 'NORMA':
            from .norma import NORMA

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            clf = NORMA(pattern_length=sliding_window, nm_size=3 * sliding_window, **self.__model_parameters)
            x = data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            score = np.array([score[0]] * ((sliding_window - 1) // 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

        elif model == 'PCA':
            from TSB_UAD.models.pca import PCA

            if sliding_window:
                X_data = Window(window=sliding_window).convert(data).to_numpy()
            else:
                sliding_window = find_length(data)
                X_data = Window(window=sliding_window).convert(data).to_numpy()

            clf = PCA(**self.__model_parameters)
            x = X_data
            clf.fit(x)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))

        elif model == 'POLY':
            from TSB_UAD.models.poly import POLY

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            clf = POLY(window=sliding_window, **self.__model_parameters)
            x = data
            clf.fit(x)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        elif model == 'OCSVM':
            from TSB_UAD.models.ocsvm import OCSVM

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            data_train = data[:int(0.1 * len(data))]
            data_test = data
            X_train = Window(window=sliding_window).convert(data_train).to_numpy()
            X_test = Window(window=sliding_window).convert(data_test).to_numpy()

            X_train_ = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_train.T).T
            X_test_ = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_test.T).T
            clf = OCSVM(**self.__model_parameters)
            clf.fit(X_train_, X_test_)
            score = clf.decision_scores_
            score = np.array([score[0]] * math.ceil((sliding_window - 1) / 2) + list(score) + [score[-1]] * ((sliding_window - 1) // 2))
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        elif model == 'LSTM':
            from TSB_UAD.models.lstm import lstm

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            data_train = data[:int(0.1 * len(data))]
            data_test = data

            clf = lstm(slidingwindow=sliding_window, **self.__model_parameters)
            clf.fit(data_train, data_test)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        elif model == 'AE':
            from TSB_UAD.models.AE_mlp2 import AE_MLP2

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            data_train = data[:int(0.1 * len(data))]
            data_test = data

            clf = AE_MLP2(slidingWindow=sliding_window, **self.__model_parameters)
            clf.fit(data_train, data_test)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        elif model == 'CNN':
            from TSB_UAD.models.cnn import cnn

            if sliding_window:
                sliding_window = sliding_window
            else:
                sliding_window = find_length(data)

            data_train = data[:int(0.1 * len(data))]
            data_test = data

            clf = cnn(slidingwindow=sliding_window, **self.__model_parameters)
            clf.fit(data_train, data_test)
            measure = Fourier()
            measure.detector = clf
            measure.set_param()
            clf.decision_function(measure=measure)
            score = clf.decision_scores_
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

        else:
            raise ValueError(f"The given anomaly detector '{model}' is not supported by TSB UAD!\n"
                             f"Supported PYODAnomalyDetectors are: {_SUPPORTED_TSB_UAD_ANOMALY_DETECTORS}")

        return score

    @staticmethod
    def load(parameters: Dict[str, any]) -> 'TimeSeriesAnomalyDetector':
        return TSBUADAnomalyDetector(
            model=parameters['tsb_uad_model'],
            window=parameters['window'] if 'window' in parameters else None,
            model_parameters=parameters['model_parameters'] if 'model_parameters' in parameters else None,
        )
