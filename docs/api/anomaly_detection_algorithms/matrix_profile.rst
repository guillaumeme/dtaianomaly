Matrix Profile based anomaly detection
======================================

The `Matrix Profile <https://www.cs.ucr.edu/~eamonn/MatrixProfile.html>`_ has
the potential to revolutionize time series data mining because of its generality,
versatility, simplicity and scalability. The matrix profile is a meta-time series
which computes, for each subsequence, the distance to its nearest neighbor. This
has implications for many, perhaps *most*, time series data mining tasks, including
motif discovery, semantic segmentation, etc. In particular, ``dtaianomaly`` exploits
it use for time series anomaly detection. If a subsequence has a large distance to
all other subsequences in the time series (its value in the matrix profile is high),
then it is likely an anomaly.

STOMP
-----

.. autoclass:: dtaianomaly.anomaly_detection.STOMP

