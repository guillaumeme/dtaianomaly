# Time Series Anomaly Detection

[![pipeline status](https://gitlab.kuleuven.be/u0143709/time-series-anomaly-detection/badges/main/pipeline.svg)](https://gitlab.kuleuven.be/u0143709/time-series-anomaly-detection/-/commits/main)
[![coverage report](https://gitlab.kuleuven.be/u0143709/time-series-anomaly-detection/badges/main/coverage.svg)](https://gitlab.kuleuven.be/u0143709/time-series-anomaly-detection/-/commits/main)

> **_IMPORTANT:_** `dtaianomaly` is still a work in progress. Therefore, many changes 
> are still expected. Feel free to [contact us](#contact) if there are any suggestions!

A simple-to-use Python package for the development and analysis of time series anomaly 
detection techniques. 

## Table of Contents
1. [Installation](#installation): How to install `dtaianomaly`.
2. [Usage](#usage): How to use `dtaianomaly`, both in your own code and through configuration files
3. [More examples](#more-examples) A list of more in-depth examples. 
4. [Contact](#contact): How to get in touch with us.

## Installation

You can install `dtaianomaly` using `pip`:

```
pip install dtaianomaly
```

## Usage

### In code

Here we show how you can use `dtaianomaly` in your own code. We first show how to load 
datasets using the `DataManager`. If you already have a time series as a `np.ndarray`
of size `(n_samples, n_features)`, you can skip this step. Second, we show how to use 
the `TimeSeriesAnomalyDetector` class to detect anomalies in the data. Third, we 
show how to quantitatively evaluate the results of the anomaly detection algorithm.
Because time series are inherently something visual, we also show how to use `dtaianomaly`
to visualize the results of the anomaly detection algorithm. [This jupyter notebook](notebooks/README_demo.ipynb)
contains all the code cells shown below.

#### 1. Loading data

Data can be read using the `DataManager` class. Below we give a simple example of loading 
data using the `DataManager` class. More information regarding how to structure the datasets 
and how to select datasets with certain properties can be found in the [data](data) folder.

> The reasoning of `DataManager` is inspired by [TimeEval](https://github.com/HPI-Information-Systems/TimeEval/tree/main).

```python
from dtaianomaly.data_management import DataManager

# Initialize the data manager
data_manager = DataManager(data_dir='data', datasets_index_file='datasets.csv')

# Select all datasets
data_manager.select({'collection_name': 'Demo', 'dataset_name': 'Demo1'}) 
# Get the index of the first selected dataset
dataset_index = data_manager.get(0)  
# Load the trend data (as a numpy ndarray) and the anomaly labels
trend_data, labels = data_manager.load_raw_data(dataset_index, train=False)
```

#### 2. Detecting anomalies

The `TimeSeriesAnomalyDetector` class is the main class of `dtaianomaly` as it is the base
of all time series anomaly detection algorithms. The main methods of this class are:

1. `fit(trend_data: np.ndarray, labels: np.array = None)` to fit the anomaly detector. The 
   `labels`  parameter is optional and should only be given to supervised time series anomaly 
    detection algorithms. 
2. `decision_function(trend_data: np.ndarray)` to compute the raw anomaly scores of every 
   measurement the time series. The scores are a value in the range $[0, +\infty[$, in which 
   a absolute value of the anomaly score indicates how anomalous an observation is. 
3. `predict_proba(trend_data: np.ndarray, normalization: str = 'unify')` converts the raw
   anomaly scores to a probability of an observation being anomalous (thus in range $[0, 1]$). The `normalization` 
   parameter indicates how the raw anomaly scores should be normalized.

Here we show a simple example to detect anomalies in time series. Specifically, we use an 
`IForest` (as implemented in [PyOD](https://github.com/yzhao062/pyod)), but adapted for 
time series using a sliding widow of size 16. 

```python
from dtaianomaly.anomaly_detection import PyODAnomalyDetector, Windowing

# Initialize the anomaly detector
# Here we use an IForest with a sliding window of size 16
anomaly_detector = PyODAnomalyDetector('IForest', Windowing(window_size=100))

# Fit the anomaly detector 
anomaly_detector.fit(trend_data)
# Compute the raw anomaly scores of an observation (in range [0, infinity])
raw_anomaly_scores = anomaly_detector.decision_function(trend_data)
# Compute the probability of an observation being an anomaly (in range [0, 1])
anomaly_probabilities = anomaly_detector.predict_proba(trend_data)
```

In this example, `anomaly_detector` can be any of the implemented anomaly detection algorithms.
This allows for abstraction using the `TimeSeriesAnomalyDetector` class, which can be used to 
implement pre- and post-processing steps for anomaly detection algorithms.

#### 3. Evaluating results

The `evaluation` module contains functions to evaluate the results of the anomaly detector, as
shown below. Some methods use continuous anomaly scores (such as the area under the precision-recall
curve), while others require discrete anomaly labels (such as the F1 score). Therefore, we provide
several thresholding methods, such as `fixed_value_threshold`. 

```python
from dtaianomaly.evaluation import Fbeta, PrAUC, FixedValueThresholding

# Compute the F1 score, for which discrete anomaly labels are required
f1_score = Fbeta(FixedValueThresholding(), beta=1.0).compute(labels, raw_anomaly_scores)
 
# Compute the area under the precision-recall curve
pr_auc_score = PrAUC().compute(labels, raw_anomaly_scores)
```

#### 4. Visualizing the results

To easily visualize the results of the anomaly detection algorithm (beyond numerical results), 
we provide methods to visualize the data and the anomaly scores. A simple example is shown below.

```python
from dtaianomaly.visualization import plot_anomaly_scores

# Load the trend data as a pandas DataFrame
trend_data_df = data_manager.load(dataset_index, train=False)
plot_anomaly_scores(trend_data_df, raw_anomaly_scores)
```
![Anomaly scores](notebooks/README_demo.svg)

### Using configuration files

One of the best ways to guarantee reproducibility of your experiments is to use configuration
files. Therefore, we implemented a simple way to provide configuration files for the evaluation
of time series anomaly detection algorithms. The configurations are formatted in `json` format, 
but can also be passed directly as a dictionary. Below we show how to use a configuration files to 
execute an algorithm. Checkout the [experiments](experiments) folder for more information regarding
the format of the configuration files and examples.

```python
from dtaianomaly.workflows import execute_algorithm

results = execute_algorithm(
   data_manager=data_manager,
   # Give configurations as the location of the configuration file ...
   data_configuration='experiments/default_configurations/data/Demo.json',
   algorithm_configuration='experiments/default_configurations/algorithm/iforest.json',
   # ... or directly as a dictionary
   metric_configuration={
     "pr_auc": { },
     "precision": {
       "thresholding_strategy": "fixed_value_threshold",
       "thresholding_parameters": {
         "threshold": 0.05
       }
     }
   }
)
`````

## More examples
More examples will be added in the [notebooks](notebooks) directory soon!
- [PyOD anomaly detectors](notebooks/pyod_anomaly_detectors.ipynb): Compares different anomaly detection algorithms 
  implemented in the PyOD library on a simple time series, showing how to easily initialize a `PyODAnomalyDetector` 
  and compare multiple methods. 
- [Compare normalization](notebooks/compare_normalization.ipynb): Compares different normalization 
  methods for anomaly scores, showing how to easily compare multiple methods.
- [Analyze decision scores](notebooks/analyze_decision_scores.ipynb): Vizually illustrates the decision
  scores of various anomlay detectors.

## Contact
Feel free to email [louis.carpentier@kuleuven.be](mailto:louis.carpentier@kuleuven.be) if 
there are any questions, remarks, ideas, ...
