
"""
This module contains the all functionality to effectively detect anomalies. It can be imported as follows:

.. code-block:: python

   from dtaianomaly import anomaly_detection

The most important class in this module is the :py:class:`~dtaianomaly.anomaly_detection.TimeSeriesAnomalyDetector`
class, which offers a generic interface to use any anomaly detector in ``dtaianomaly``.
"""

from .TimeSeriesAnomalyDetector import TimeSeriesAnomalyDetector

from .utility import Windowing
from .utility import TrainType

from .pyod import PyODAnomalyDetector
from .matrix_profile import STOMP
from .tsbuad import TSBUADAnomalyDetector
