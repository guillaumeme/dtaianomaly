# Experiments

Experiments **will** be defined as a directory containing a config file. 
However, this feature is not yet implemented. Loading specific datasets, 
algorithms and metrics through config files is already possible. The format
of these files are described below. 

## Format of config files

All config files are formatted through `json`. This is a dictionary-based 
format, which allows the workflows to be run in code by providing `Python`
dictionaries. Some default configuration files are provided in the [default 
configuration directory](default_configurations).

### Dataset configuration 

The dataset configuration has a single key: `select`. This key contains a list
of nested dictionaries. These indicate the properties of the datasets to select. 
Each element in the list corresponds to a single call of the `select` function of
the `DataManager`, and consequently all properties must be satisfied (`AND`). The 
different entries of the list correspond to different calls of the `select` function,
and consequently if any of the entries is satisfied, then the dataset is selected
(`OR`).

For instance, below are all the datasets in the `Demo` collection selected and the 
datasets with at most 10 000 observations from the `KDD-TSAD` collection.
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
```

> Click [here](../data/README.md#selecting-datasets) to get more information regarding 
> how to select datasets. 

### Algorithm configuration

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

### Metric configuration

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
