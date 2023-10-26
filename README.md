# Time Series Anomaly Detection

[![pipeline status](https://gitlab.kuleuven.be/u0143709/dtaianomaly/badges/main/pipeline.svg)](https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/commits/main)
[![coverage report](https://gitlab.kuleuven.be/u0143709/dtaianomalybadges/main/coverage.svg)](https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/commits/main)
[![Downloads](https://static.pepy.tech/badge/dtaianomaly)](https://pepy.tech/project/dtaianomaly)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dtaianomaly.svg)](https://pypi.python.org/pypi/dtaianomaly/)
[![PyPI license](https://img.shields.io/pypi/l/dtaianomaly.svg)](https://pypi.python.org/pypi/dtaianomaly/)


> **_IMPORTANT:_** `dtaianomaly` is still a work in progress. Therefore, many changes 
> are still expected. Feel free to [contact us](#contact) if there are any suggestions!

A simple-to-use Python package for the development and analysis of time series anomaly 
detection techniques. Here we describe the main usage of `dtaianomaly`, but be sure to
check out the [documentation](https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/) 
for more information. 

## Installation
The easiest way to install `dtaianomaly` is through [PyPi](https://pypi.org/project/dtaianomaly/):
```
pip install dtaianomaly
```

## Features
The three key features of `dtaianomaly` are as follows:
1. **Large scale experiments.** To evaluate anomaly detection methods, it is crucial to
   be able to perform large scale experiments. However, it is also crucial to ensure 
   reproducibility of the obtained results. ´dtaianomaly´ provides a simple way to evaluate
   an anomaly detector on a large set of time sets. This allows to both (1) quantitatively 
   evaluate the method by measuring the performance, runtime and memory usage, and (2)
   qualitatively evaluate the method by visually inspecting the detected anomalies. 
   This is achieved by using configuration files, which ensure that identical settings
   are used for the experiments. We refer to the [documentation](https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/experiments.html) for more details regarding
   how to set up and run experiments.
2. **Develop anomaly detectors.** The models in `dtaianomaly` are all centered around the
   `TimeSeriesAnomalyDetector` class, which provides an interface to detecting anomalies in
   time series. The main advantage of this abstract class is that anomaly detectors can be
   handled in an abstract manner. This allows to develop wrapper approaches around existing 
   methods, without needing information about the specific algorithm. On top of this, by 
   only implementing the interface of `TimeSeriesAnomalyDetector`, it is possible to develop
   new methods that can be used within the entire infrastructure of `dtaianomaly`. An example
   of how to implement a custom anomaly detector is given in [this notebook](notebooks/custom_anomaly_detector.ipynb).
3. **Detect anomalies through a simple API.** Once an anomaly detector is developed and
   validated, it can be used to detect anomalies in time series. `dtaianomaly` provides 
   a simple API (through `fit` and `predict` methods) to detect anomalies. The below code 
   snippet illustrates how to detect anomalies in only a few lines of code. The complete 
   example can be found in [this notebook](notebooks/README_demo.ipynb). The main advantage 
   of ´dtaianomaly´ is that anomaly detectors can be used as a wrapper approach. Thus, you 
   do not need to know anything about the ´TimeSeriesAnomalyDetector´ in order to
   detect anomalies. 

```python
from dtaianomaly.anomaly_detection import PyODAnomalyDetector, Windowing

trend_data = ... # Some time series as a numpy array of shape (n_samples, n_features)

# Initialize an IForest that takes as features each window of 100 observations
anomaly_detector = PyODAnomalyDetector('IForest', Windowing(window_size=100))

# Fit the anomaly detector 
anomaly_detector.fit(trend_data)
# Compute the raw anomaly scores of an observation (in range [0, infinity])
raw_anomaly_scores = anomaly_detector.decision_function(trend_data)
# Compute the probability of an observation being an anomaly (in range [0, 1])
anomaly_probabilities = anomaly_detector.predict_proba(trend_data)
```
![Anomaly scores](notebooks/README_demo.svg)

## Examples
Several examples of how `dtaianomaly` can be used are provided in the [notebooks](notebooks). Here
we list some of the most important ones to get to know `dtaianomaly`:
- [Quantitatively evaluate an anomaly detector](notebooks/execute_workflow.ipynb): Shows how to 
  quantitatively evaluate an anomaly detector on a benchmark set of time series. 
- [Custom anomaly detector](notebooks/custom_anomaly_detector.ipynb): Illustrates a simple example 
  of a custom anomaly detector implemented in `dtaianomaly`. 
- [PyOD anomaly detectors](notebooks/analyze_pyod_anomaly_detectors.ipynb): Compares different anomaly detection algorithms 
  implemented in the [PyOD](https://pyod.readthedocs.io/en/latest/) library, showing how you can use an anomaly 
  detector as a wrapper approach. 

## Dependencies
Time series are represented as [NumPy](https://numpy.org/)-arrays to detect anomalies, but can 
also be represented as [Pandas](https://pandas.pydata.org/) DataFrames for visualization. Anomaly
detection algorithms use the [PyOD](https://pyod.readthedocs.io/en/latest/) library. We use 
[matplotlib](https://matplotlib.org/) for visualization. 

`dtaianomaly` also depends on [scikit-learn](https://scikit-learn.org/stable/) and 
[scipy](https://www.scipy.org/), but we plan on removing these dependencies in the near future.

## Contact
Feel free to email to [louis.carpentier@kuleuven.be](mailto:louis.carpentier@kuleuven.be) if 
there are any questions, remarks, ideas, ...

## License
    Copyright (c) 2023 KU Leuven, DTAI Research Group
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
