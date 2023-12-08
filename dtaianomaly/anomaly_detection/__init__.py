
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

try:
    from .pyod import PyODAnomalyDetector
except ImportError:
    pass  # In case not all dependencies were installed

try:
    from .matrix_profile import STOMP
except ImportError:
    pass
