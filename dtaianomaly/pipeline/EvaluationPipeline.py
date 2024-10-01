
from typing import List, Dict, Union

import numpy as np

from dtaianomaly.utils import is_valid_list, is_valid_array_like
from dtaianomaly.preprocessing import Preprocessor
from dtaianomaly.anomaly_detection import BaseDetector
from dtaianomaly.evaluation import ProbaMetric
from dtaianomaly.pipeline import Pipeline


class EvaluationPipeline:
    """
    Pipeline to combine a base pipeline, and a set of metrics. Used
    in the workflow. The given :py:class:`~dtaianomaly.preprocessing.Preprocessor`
    and :py:class:`~dtaianomaly.anomaly_detection.BaseDetector` are
    combined into a :py:class:`~dtaianomaly.pipeline.Pipeline` object.

    Parameters
    ----------
    preprocessor: Preprocessor or list of Preprocessors
        The preprocessors to include in this evaluation pipeline.
    detector: BaseDetector
        The anomaly detector to include in this evaluation pipeline.
    metrics: list of Probametric objects
        The evaluation metrics to compute in this evaluation pipeline.
    """
    pipeline: Pipeline
    metrics: List[ProbaMetric]

    def __init__(self,
                 preprocessor: Union[Preprocessor, List[Preprocessor]],
                 detector: BaseDetector,
                 metrics: Union[ProbaMetric, List[ProbaMetric]]):
        if not (isinstance(metrics, ProbaMetric) or is_valid_list(metrics, ProbaMetric)):
            raise TypeError("metrics should be a list of ProbaMetric objects")
        self.pipeline = Pipeline(preprocessor=preprocessor, detector=detector)
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def run(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Run the pipeline and evaluate performance.

          Parameters
        ----------
        X: array-like of shape (n_samples, n_attributes)
            Input time series.
        y: array-like of shape (n_samples)
            The ground truth labels.

        Returns
        -------
        performances: Dict[str, float]
            The evaluation of the performance metrics. The keys are
            string descriptors of the performance metrics, with values
            the corresponding performance score.
        """
        if not is_valid_array_like(X):
            raise ValueError("X is not a valid array-like!")
        if not is_valid_array_like(y):
            raise ValueError("X is not a valid array-like!")

        self.pipeline.fit(X, y)
        probas = self.pipeline.predict_proba(X=X)
        X_, y_ = self.pipeline.preprocessor.transform(X, y)
        return {
            str(metric): metric.compute(y_true=y_, y_pred=probas)
            for metric in self.metrics
        }
