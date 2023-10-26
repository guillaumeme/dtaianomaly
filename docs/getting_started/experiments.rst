Large Scale Experiments
=======================

To qualitatively evaluate an anomaly detection algorithm, it must be applied to a
large set of time series, in which its performance can be measured. ``dtaianomaly``
provides workflows to execute such experimental evaluation. A workflow can be set
up by providing a number of ``json`` configuration files. This has two advantages:
1. Reproducibility is guaranteed by storing the configuration files.
2. Because ``json`` boils down to dictionaries, which allows to programmatically set
up workflows.

Below we discuss :ref:`how to set up and execute a workflow <Executing a workflow>` and
:ref:`how the configuration files are formatted <Configuration files>`.

Executing a workflow
--------------------



Configuration files
-------------------

Below is described how the configuration files should be formatted. Additionally, some
configurations are available `here <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/tree/main/configurations>`_.
These can be used directly in your workflow, or as reference to set up a custom
workflow.

Datasets
~~~~~~~~

The dataset configuration has a single key: ``'select'``, with as value a list
of nested dictionaries. These dictionaries are used as argument to seperate calls
of the :py:meth:`DataManager.select` method. Recall that each criteria within a
call (within a dictionary in this case) is combined with an ``AND`` operator, and
each call (each element in the list) is combined with an ``OR`` operator. More
information can be found on :ref:`this webpage <data_management/Selecting time series>`.
Here we show a configuration file to select all time series in the ``'Demo'`` collection
and all time series with at most 10 000 observations in the ``'KDD-TSAD'`` collection:
```json
{
  "select": [
    {
      "collection_name": "Demo"
    },
    {
      "collection_name": "KDD-TSAD",
      "length" : [0, 10000]
    }
  ]
}

.. note::

    Note that the in the end, only the available time series are selected using the
    :py:meth:`DataManager.filter_selected_datasets` method, because it makes no sense
    to execute a workflow on time series that are not available.

Algorithms
~~~~~~~~~~

The `TimeSeriesAnomalyDetector` has a static `load(Dict[str, Any])` method to load
specific anomaly detector and its parameters. The anomaly detector is indicated
through the `anomaly_detector` keyword. This equals the class name of the anomaly
detector. The various anomaly detectors differ (slightly) in how their configuration
should look like. Therefore, the different types are discussed below.

#### PyOD anomaly detectors

A PyOD anomaly detector, indicated by the `PyODAnomalyDetector` name, has three additional
parameters:
1. `pyod_model`: The name of the PyOD model, for example `IForest` or `LOF`.
2. `pyod_model_parameters`: The parameters of the PyOD model, for example the number of trees
   for the `IForest` model or the number of neighbors for the `LOF` model. This property is
   optional, and default parameters are used if it isn't given.
3. `windowing`: A dictionary mapping the parameter names of a `Windowing` object onto the
   corresponding values such as `window_size`.

```json
{
  "anomaly_detector": "PyODAnomalyDetector",
  "pyod_model": "IForest",
  "pyod_model_parameters": {
    "n_estimators": 100
  },
  "windowing": {
    "window_size": 100
  }
}
```

Metrics
~~~~~~~

The metric configuration is the simplest one. The keys indicate the name of
the metric to compute, for example the `auc_roc` metric for Area Under the
Receiver Operator Curve or `precision` precision. The value can contain additional
information to compute the metric

Some parameters can handle a continuous scoring function (such as `auc_roc`), but
others require some thresholding. The `thresholding_strategy` parameter indicates
how the continuous predicted probabilities should be converted to anomaly labels
(e.g., `contamination` for a fixed contamination rate). The parameters required
for thresholding can be provided through the `thresholding_parameters` property
(e.g., the specific contamination rate to employ). If no `thresholding_parameters`
are given, then the parameters are obtained from the ground truth.

Certain metrics may require additional parameters to compute, such as the $f$-score.
These additional parameters can be provided through the `metric_parameters` property.

If the metric name does not correspond a known metric, then the `metric_name` parameter
is searched. This allows to compute the same metric twice, but with different parameters.
The key of the entry is used to indicate the result of computing the given metric.

```json
{
  "auc_roc": { },
  "precision": {
    "thresholding_strategy": "contamination",
    "thresholding_parameters": {
      "contamination": 0.1
    }
  },
  "precision2": {
    "metric_name": "precision",
    "thresholding_strategy": "contamination",
    "thresholding_parameters": {
      "contamination": 0.2
    }
  }
}
```

Output
~~~~~~

