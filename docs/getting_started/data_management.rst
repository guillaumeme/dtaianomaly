Data Management
===============

It is very simple to load data within ``dtaianomaly``. This is done through the
:py:class:`DataManager` class. It allows to dynamically select time series that
satisfy certain criteria, without having to load all time series. Below are the
core functionalities discussed. For more information, please refer to the
:doc:`documentation <../api/data_management>`.

.. note::
    The reasoning of :py:class:`DataManager` is inspired by
    `TimeEval <https://github.com/HPI-Information-Systems/TimeEval/tree/main>`_.

Dataset index
-------------

The :py:class:`DataManager` requires a dataset index to work properly. This file
contains the relevant metadata regarding all the time series can be used. An example
of such index file is shown in `datasets.csv <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/blob/main/data/datasets.csv>`_.
The metadata includes features such the length of the time series, the dimension,
and the number of anomalies. This index file also contains the path to the specific
train and test data. The location of these datafiles should be relative to the location
of the index file. The below snippet shows how to initialize :py:class:`DataManager`.

.. code-block:: python

    from dtaianomaly.data_management import DataManager
    data_manager = DataManager(
       # The location of all the datasets
       data_dir='data',
       # The location of the dataset index file, relative to the `data_dir`
       datasets_index_file='datasets.csv'
    )
..

.. note::
    A large set of time series that follow this structure can be found on
    `This webpage <https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html>`__.


.. _select_time_series:

Selecting time series
---------------------

One of the main features of the :py:class:`DataManager` is the ability to select time series
based on certain criteria. This is done through the :py:meth:`DataManager.select` method. This method
takes a dictionary as argument, which specifies the criteria that should be met. For example, we can
select the time series from the ``'Demo'`` collection as follows:

.. code-block:: python

    data_manager.select({"collection_name": "Demo"})
..

For numerical properties (e.g., ``'length'``), a single element indicates an exact match and a
tuple indicates a range. For categorical properties (e.g., ``strings``), the value is either a
single element indicating an exact match or a list of values indicating multiple possible exact
matches. For booleans, the value is either ``True`` or ``False`` (no list is allowed, because then
simply all datasets would be selected). For example:

.. code-block:: python

   data_manager.select({
      # The collection name must exactly match either 'DAPHNET' or 'KDD-TSAD'
      "collection_name": ["DAPHNET", "KDD-TSAD"],
       # The data may not have trend
      "trend": "no_trend",
      # The length must be between 0 and 10000
      "length": [0, 10000],
      # There should be exactly one anomaly
      "num_anomalies": 1,
      # The index should be in datetime format
      "datetime_index": True
   })
..

The properties within a single :py:meth:`DataManager.select` call are treated as an ``AND``-operation,
i.e., all above properties must hold for a dataset to be selected. To perform an ``OR``-operation,
multiple calls to :py:meth:`DataManager.select` can be used, as is shown in below example. Notice
that it is possible to chain the different calls, because the :py:meth:`DataManager.select` method
returns a reference to the :py:class:`DataManager` object itself:

.. code-block:: python

    data_manager.select({
       # Select all datasets from the DAPHNET collection with a length at most 10 000 samples
       # and contamination between 0.05 and 0.1, ...
       "collection_name": "DAPHNET",
       "length": [0, 10000],
       "contamination": (0.05, 0.1)
    }).select({  # Chain the calls
       # ... or any the datasets from the DAPHNET collection that has a contamination below 0.05.
       "collection_name": "DAPHNET",
       "contamination": (0.00, 0.05)
    })
..

One option would be to build the dataset index file from scratch. However, this might become
tedious. Another option is to simply copy the file from `GitLab <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/blob/main/data/datasets.csv>`_!
But now the problem could be that certain datasets are in the index file that you do not have
locally available. Therefore, we provide the :py:meth:`DataManager.filter_available_datasets`
function. This method will deselect all datasets that are not available (the path to the dataset
does not exist). A simple example is shown here:

.. code-block:: python

    # Select all *available* datasets with at most 10 000 observations
    data_manager.select({"length": [0, 10000]).filter_available_datasets()
..

Obtaining the time series
-------------------------

The :py:meth:`DataManager.select` method updates the internal state of the :py:class:`DataManager`
to maintain the selected time series. The :py:meth:`DataManager.get` method can be used to obtain
an index to the selected time series, which can be used to load the :ref:`time series <data_management_load_time_series>`.
The below snippet illustrates how to obtain all and only a single data set index:

.. code-block:: python

    all_selected_datasets = data_manager.get()
    selected_dataset_at_0 = data_manager.get(index=0)
..

.. _data_management_load_time_series:

Loading the time series
-----------------------

The :py:class:`DataManager` provides two methods to load the time series, using an index
obtained through :py:meth:`DataManager.get`.

1. :py:meth:`DataManager.load` loads the data as a pandas data frame, in which the index
is the time stamp of each observation, and each column represents an attribute of the time
series. The dataframe also contains a column ``'is_anomaly'`` which equals the round truth
anomaly labels.

2. :py:meth:`DataManager.load_raw_data` loads the data as a numpy ndarray of size
``(n_samples, n_features)``. This method also returns the ground truth labels as a numpy
array. This method is preferred for applying anomaly detection, because the data is already
in the correct format.

Both methods have an boolean parameter ``train`` which indicates whether the train or
test data should be returned. If the train data is requested, but no train trend
data exists, an error is raised. The below snippet illustrates these methods:

.. code-block:: python

    # Load the train and test data as a dataframe
    train_trend_data_df = data_manager.load(dataset_index, train=True)
    test_trend_data_df = data_manager.load(dataset_index, train=False)
    # Load the train and test data as a numpy ndarray
    train_trend_data, train_ground_truth = data_manager.load_raw_data(dataset_index, train=True)
    test_trend_data, test_ground_truth = data_manager.load_raw_data(dataset_index, train=False)
..