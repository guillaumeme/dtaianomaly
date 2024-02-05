Large Scale Experiments
=======================

To qualitatively evaluate an anomaly detection algorithm, it must be applied to a
large set of time series, in which its performance can be measured. ``dtaianomaly``
provides workflows to execute such experimental evaluation. A workflow can be set
up by providing a number of ``json`` configuration files. This has two advantages:
1. Reproducibility is guaranteed by storing the configuration files.
2. Because ``json`` boils down to dictionaries, which allows to programmatically set
up workflows.

Below we discuss :ref:`how to set up and execute a workflow <large_scale_experiments_workflow>`
and :ref:`how the configuration files are formatted <large_scale_experiments_configuration>`.


.. _large_scale_experiments_workflow:

Executing a workflow
--------------------

A workflow can be executed through the :py:meth:`~dtaianomaly.workflow.execute_algorithm` method.
This method requires a :py:class:'dtaianomaly.data_management.DataManager' to read the
data, as well as the different configurations described below. These configurations
can be a string representing the path to the ``json`` file, or a Python dictionary.

Alternatively, we provide a command line interface to execute a workflow. The interface
is provided in `main.py <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/blob/main/main.py>`_
in `GitLab <https://gitlab.kuleuven.be/u0143709/dtaianomaly>`_. Download the file and
navigate to the directory in which the file is stored. Then, execute the following command
to start a workflow:

.. code-block:: bash

    python main.py
        --seed <seed to set before detecting anomalies in a time series>
        --n_jobs <The number of time series to detect anomalies in, in parallel>
        --datasets_index_file <path to the dataset index file>
        --configuration_dir <path to the directory containing all the configuration files>
        --config <path to the general configuration file, relative to 'configuration_dir'>
        --data <path to the data configuration file, relative to 'configuration_dir'>
        --algorithm <path to the algorithm configuration file, relative to 'configuration_dir'>
        --metric <path to the metric configuration file, relative to 'configuration_dir'>
        --output <path to the output configuration file, relative to 'configuration_dir'>


The ``--config`` argument requires a path to a ``json`` file. This file should contain for
properties: ``'data'``, ``'algorithm'``, ``'metric'`` and ``'output'``. Each of these
refers to a path to the corresponding configuration file (relative to ``--configuration_dir``).
If a certain property is not provided in the general configuration file, then the command line
interface looks for the value given to the corresponding property. Note that this ensures
priority to the ``--config`` argument.

.. _large_scale_experiments_configuration:

Configuration files
-------------------

Below is described how the configuration files should be formatted. Additionally, some
configurations are available `here <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/tree/main/configurations>`_.
These can be used directly in your workflow, or as reference to set up a custom
workflow.

Datasets
~~~~~~~~

The dataset configuration has a single key: ``'select'``, with as value a list
of nested dictionaries. These dictionaries are used as argument to separate calls
of the :py:meth:`DataManager.select` method. Recall that each criteria within a
call (within a dictionary in this case) is combined with an ``AND`` operator, and
each call (each element in the list) is combined with an ``OR`` operator. More
information can be found on :ref:`here <select_time_series>`.
Here, we show a configuration file to select all time series in the ``'Demo'`` collection
and all time series with at most 10 000 observations in the ``'KDD-TSAD'`` collection:

.. code-block:: json

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

    In the end, only the available time series are selected using the
    :py:meth:`DataManager.filter_selected_datasets` method, because it makes no sense
    to execute a workflow on time series that are not available.

Algorithms
~~~~~~~~~~

It is possible to either execute a single algorithm or a large set of algorithms. Below,
we first describe how the configuration of a single algorithm looks like. Afterwards, the
configuration for multiple algorithms is described, which uses single-algorithm configurations.

The algorithm configuration requires three key-value pairs:
1. ``'name'``: the semantic name to use when referring to the anomaly detector. This can
be useful to use when quantifying the effect of hyperparameters, for example.
2. ``'anomaly_detector'``: the class name of the specific anomaly detector to use. For example, for
a :py:class:`PyODAnomalyDetector`, the value would be ``'PyODAnomalyDetector'``.
3. ``'hyper_parameters'``: the hyperparameters to use for the specific anomaly detector. This should
be a dictionary, and the arguments are passed on :py:meth:`TimeSeriesAnomalyDetector.load` method of
the anomaly detector given by the ``'anomaly_detector'``.

Below we show the configuration file for a :py:class:`PyodAnomalyDetector`, which uses an
IForest to detect anomalies.

.. code-block:: json

    {
      "name": "IForest (w=64)",
      "anomaly_detector": "PyODAnomalyDetector",
      "hyperparameters": {
        "pyod_model": "IForest",
        "windowing": {
          "window_size": 64
        },
        "pyod_model_parameters": {
          "n_estimators": 100
        }
      }
    }

.. note::

    While the general structure of the anomaly detector configuration is relatively
    fixed (i.e., keys ``'name'``, ``'anomaly_detector'`` and ``'hyperparameters'``),
    the hyperparameters dictionary can vary depending on the anomaly detector. We
    therefore refer to the documentation of the :py:meth:`TimeSeriesAnomalyDetector.load`
    method of the specific anomaly detector you want to load for more information.


.. _note_module_path_in_algorithm_config:
.. note::

    The workflow searches for the correct file in :py:mod:`dtaianomaly.anomaly_detection`
    module. Because of this, a custom anomaly detector will not be recognized by the
    workflow by default. For this, you need to include an item ``'module_path'`` in the
    configuration, which dictates where the custom anomaly detector is located,
    relative from where the workflow has been started. Also make sure that the directory
    containing your code contains a (potentially empty) ``__init__.py`` file, to make
    sure the directory is recognized as a module.

Oftentimes, it is desirable to setup an experiment with a large number
of datasets and algorithms. This can be done by adapting the configuration
file for the algorithms as follows. Only one key is obligated to be in the
configuration: ``'algorithm_configurations'``. The corresponding value is
a list of configurations for single anomaly detectors, as described above.
The concrete value can either be a string representing the path of the
``json`` configuration file (relative to where the workflow started) or
directly a dictionary with correct configurations (or a combination of both!).
Additionally, there is an optional key ``'collection'``. When provided, a
sub directory will be created, in which all the results of the workflow
are saved, allowing to group all information related to the workflow-run.

.. code-block:: json

    {
      "collection" : "multiple_algorithms",
      "algorithm_configurations": [
          "configurations/algorithm/iforest_16.json",
          "configurations/algorithm/iforest_64.json"
      ]
    }


Metrics
~~~~~~~

The metric configuration dictates which metrics should be computed to measure algorithm
performance. This allows to only have to detect the anomalies once for a specific
method, and then compute all metrics of interest. The configuration is a dictionary,
in which the keys indicate the name of the metric to compute, for example the ``'auc_roc'``
metric for `Area Under the Receiver Operator Curve <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
or ``'precision'`` for `precision <https://en.wikipedia.org/wiki/Precision_and_recall>`_.
We first show an example and discuss the structure in more detail below.

.. code-block:: json

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
      "fbeta": {
        "metric_parameters": {
          "beta": 1.0
        },
        "thresholding_strategy": "contamination",
        "thresholding_parameters": {
          "contamination": 0.1
        }
      }
    }

Some parameters can handle a continuous scoring function (such as ``'auc_roc'``), but
others require some thresholding. The ``'thresholding_strategy'`` parameter indicates
how the continuous predicted probabilities should be converted to anomaly labels
(e.g., ``'contamination'`` for a fixed contamination rate). The parameters required
for thresholding can be provided through the ``'thresholding_parameters'`` property
(e.g., the specific contamination rate). If no ``'thresholding_parameters'`` are
given, then the parameters are obtained from the ground truth. We refer to
:ref:`this page <evaluation_thresholding_strategies>` for more information on thresholding.

Certain metrics may require additional parameters to compute, such as the $f$-score.
These additional parameters can be provided through the ``'metric_parameters'`` property.
This property has as value a dictionary, with the keys indicating the parameter name and
as value the concrete parameter value.

If the metric name does not correspond a known metric, then the ``'metric_name'`` parameter
is searched. This allows to compute the same metric twice, but with different parameters
(i.e., other thresholding strategy or other metric parameters). The key of the entry is used
to indicate the result of computing the given metric.

Output
~~~~~~

The output configuration indicates what should be outputted during the workflow. This
configuration is based on the :py:class:`OutputConfiguration <dtaianomaly.workflow.OutputConfiguration>` class. The key-value
pairs in the configuration file correspond to the properties of this class and their
corresponding value. Below we show an example of such a configuration file. Consider
for example the first property, the value ``'results'`` will be assigned to the
``'directory_path'`` property of the :py:class:`OutputConfiguration <dtaianomaly.workflow.OutputConfiguration>` class.We refer
to the documentation of the :py:class:`OutputConfiguration <dtaianomaly.workflow.OutputConfiguration>` class for more information
on the different properties and their default values.

.. code-block:: json

    {
      "directory_path": "results",
      "verbose": true,

      "trace_time": true,
      "trace_memory": true,

      "print_results": true,
      "save_results": true,
      "constantly_save_results": true,
      "results_file": "results.csv",

      "save_anomaly_scores_plot": true,
      "anomaly_scores_directory": "anomaly_score_plots",
      "anomaly_scores_file_format": "svg",
      "show_anomaly_scores": "overlay",
      "show_ground_truth": null,

      "invalid_train_type_raise_error": true
    }



