# Data

Data can be read using the `DataManager` class. Its functionality is to select datasets
based on certain properties and to load the raw data of the selected datasets. The `DataManager`
is initialized as follows
 ```python
from dtaianomaly.data_management import DataManager
data_manager = DataManager(
   # The location of all the datasets
   data_dir='data', 
   # The location of the dataset index file, relative to the `data_dir`
   datasets_index_file='datasets.csv'
)
 ```

## Dataset index

The `DataManager` class requires a dataset index, which contains all the metadata regarding
the specific datasets. An example is shown in [datasets.csv](datasets.csv). The `DataManager`
reads and uses it to select datasets based on specific properties, as well as to load the 
data from the correct location (through the `train_path` and `test_path` properties). The 
location of the data is relative to the location of the dataset index file. 

There are many properties present in the dataset index, such as the `length`, `dimensions`, 
and `num_anomalies`. Two special properties are `collection_name` and `dataset_name`. These 
are used to index the different datasets. For example, the `get()` function returns a list of 
tuples of `(collection_name, dataset_name)` of the [selected](#Selecting-datasets). The 
`load(dataset_index)` and `load_raw_data(dataset_index)` function requires a such a tuple to 
read the correct data. 

## Selecting datasets

The `DataManager` allows to easily select datasets based on certain properties. This is done 
through the `select(dataset_properties: Dict[str, Any])` method. The `dataset_properties` dictate
which properties should be satisfied. The keys of `dataset_properties` equal the name of the
property in the [dataset index](#dataset-index). The value equals what the property should satisfy.
The `select` function does not return anything, but instead updates the internal state of the 
`DataManager`. For example:
 ```python
data_manager.select({"collection_name": "Demo"})
 ```

For numerical properties (e.g., `length`), the value is a single element indicating an exact match,
or a tuple indicating a range. For categorical properties (e.g., strings), the value is either a 
single element indicating an exact match or a list of values indicating multiple possible exact matches.
For booleans, the value is either `True` or `False` (no list is allowed, because then simply all 
datasets would be selected). For example:
 ```python
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
 ```

The properties within a single `select()` call are treated as an `AND`-operation, i.e., all
above properties must hold for a dataset to be selected. to perform an `OR`-operation, you can 
use multiple separate `select()` calls. For example:
 ```python
# Select all datasets from the DAPHNET collection with a length at most 10 000 samples
# and contamination between 0.05 and 0.1, ...
data_manager.select({
   "collection_name": "DAPHNET",
   "length": [0, 10000],
   "contamination": (0.05, 0.1)
})
# ... or any the datasets from the DAPHNET collection that has a contamination below 0.05.
data_manager.select({
   "collection_name": "DAPHNET",
   "contamination": (0.00, 0.05)
})
```

## Filtering the available datasets

One option would be to build the dataset index file from scratch. However, this might become 
tedious. Another option is to simply copy the file from this repository! But now the problem 
could be that certain datasets are in the index file that you do not have locally available. 
Therefore, we provide the `filter_available_datasets()` function. This method will deselect 
all datasets that are not available (the path to the dataset does not exist). A simple 
example is shown here:
 ```python
# Select all datasets with at most 10 000 observations
data_manager.select({"length": [0, 10000])
# But remove those that are not locally available
data_manager.filter_available_datasets()
```

## Get the selected datasets

The dataset index of the selected datasets can be obtained through the `get(index: int = None)`
method. If no `index` property is given, then the dataset index of all selected datasets is 
returned as a list. Otherwise, the index specifies which dataset index  should be returned. 
 ```python
all_selected_datasets = data_manager.get()
selected_dataset_at_0 = data_manager.get(index=0)
```

## Load the datasets

There are two methods to load the data, given a dataset index (obtained through `get()`). 
The `load(dataset_index, train: bool = False)` loads the data of the dataset as a pandas 
dataframe. The index of this dataframe is equals the time stamp (either an actual date 
time or a simple enumeration). The `is_anomaly` column indicates whether the sample is
and thus is the ground truth. The other columns contain the different attributes of the 
time series. The second method is `load_raw_data(dataset_index, train: bool = False)`, 
which will load the time series as a numpy ndarray of size `(n_samples, n_features)`.
This method returns two values: the time series itself and the ground truth labels as 
numpy array. Both methods have an optional parameter `train` which indicates whether 
the train or test data should be returned. If the train data is requested, but no trend
data exists, an error is raised. 
 ```python
# Load the train and test data as a dataframe
train_trend_data_df = data_manager.load(dataset_index, train=True)
test_trend_data_df = data_manager.load(dataset_index, train=False)
# Load the train and test data as a numpy ndarray
train_trend_data, train_ground_truth = data_manager.load_raw_data(dataset_index, train=True)
test_trend_data, test_ground_truth = data_manager.load_raw_data(dataset_index, train=False)
```
