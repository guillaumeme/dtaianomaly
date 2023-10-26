import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Union

DatasetIndex = Tuple[str, str]


_COLUMN_NAMES = {
    'train_path', 'test_path', 'dataset_type', 'datetime_index', 'split_at',
    'train_type', 'train_is_normal', 'input_type', 'length', 'dimensions',
    'contamination', 'num_anomalies', 'min_anomaly_length', 'median_anomaly_length',
    'max_anomaly_length', 'mean', 'stddev', 'trend', 'stationarity', 'period_size'
}


# Inspired by Timeeval (https://github.com/HPI-Information-Systems/TimeEval/tree/main)
class DataManager:

    def __init__(self, datasets_index_file: str = 'datasets.csv'):

        # Check if the index file exists
        if not os.path.exists(datasets_index_file):
            raise ValueError(f"There is no file '{datasets_index_file}'!")

        # Separately save the directory of the datasets index file
        self.__data_dir: str = os.path.dirname(os.path.abspath(datasets_index_file))
        self.__datasets_index_file: str = datasets_index_file

        # Check if the index file contains the correct index columns
        with open(self.__datasets_index_file, 'r') as file:
            columns = file.readline().strip().split(',')
            if 'collection_name' not in columns and 'dataset_name' not in columns:
                raise IndexError(f"The dataset index file does not contain columns 'collection_name' and 'dataset_name'!")

            # Check if the columns are correct
            columns.remove('collection_name')
            columns.remove('dataset_name')
            if set(columns) != _COLUMN_NAMES:
                raise ValueError(f"The dataset index file does not contain the correct columns!\n"
                                 f"Columns in the file: {columns}\n"
                                 f"Required columns: {_COLUMN_NAMES}")

        # Read the dataset index file
        self.__datasets_index: pd.DataFrame = pd.read_csv(self.__datasets_index_file, index_col=['collection_name', 'dataset_name'])

        # Read the dataset index file
        self.__selected_datasets: pd.Series = pd.Series(index=self.__datasets_index.index, data=False)

    def clear(self) -> 'DataManager':
        # Set all values in the selected datasets to False
        self.__selected_datasets[:] = False
        return self

    def select(self, dataset_properties: Optional[Dict[str, any]] = None) -> 'DataManager':

        # If no dataset properties are given, select all datasets
        if dataset_properties is None:
            self.__selected_datasets = pd.Series(index=self.__datasets_index.index, data=True)
            return self

        # Keep track of the datasets that match all the given properties
        newly_selected_datasets = np.ones(self.__datasets_index.shape[0], dtype=bool)

        # Filter the datasets that do not match the given properties
        for dataset_property, value in dataset_properties.items():
            # Make sure the given property is valid
            if not (dataset_property in self.__datasets_index.columns or dataset_property in self.__datasets_index.index.names):
                raise ValueError(f"The dataset property '{dataset_property}' is not a valid property for a dataset!\n"
                                 f"Valid properties are: {self.__datasets_index.columns.tolist()}")

            # Filter the index explicitly
            if dataset_property == 'collection_name':
                newly_selected_datasets &= self.__datasets_index.index.get_level_values('collection_name').isin([value] if isinstance(value, str) else value)
            elif dataset_property == 'dataset_name':
                newly_selected_datasets &= self.__datasets_index.index.get_level_values('dataset_name').isin([value] if isinstance(value, str) else value)

            # Filter the remaining values implicitly, based on the type of the column
            else:
                # For boolean properties, there should only be given one value: either True or False
                if self.__datasets_index[dataset_property].dtype == bool:
                    if not isinstance(value, bool):
                        raise ValueError(f"The dataset property '{dataset_property}' is a boolean, but {value} is not a boolean!")
                    newly_selected_datasets &= (self.__datasets_index.loc[:, dataset_property] == value)

                # For number-like properties, either a single number (for exact match) or a list of two numbers (for a range) should be given
                elif self.__datasets_index[dataset_property].dtype == np.int64 or self.__datasets_index[dataset_property].dtype == np.float64:
                    if isinstance(value, list) or isinstance(value, tuple):
                        if len(value) != 2:
                            raise ValueError(f"For dataset property '{dataset_property}', the value should either express a number (exact match), or a list of two numbers (min and max)!")
                        if not (isinstance(value[0], int) or isinstance(value[0], float)) or not (isinstance(value[1], int) or isinstance(value[1], float)):
                            raise ValueError(f"Both attributes in a ranged value for dataset property '{dataset_property}' should be either an int or a float!")
                        newly_selected_datasets &= (self.__datasets_index.loc[:, dataset_property] >= value[0]) & (self.__datasets_index.loc[:, dataset_property] <= value[1])
                    else:
                        if not (isinstance(value, int) or isinstance(value, float)):
                            raise ValueError(f"The dataset property '{dataset_property}' is a number (float or int), but {value} is not a number!")
                        newly_selected_datasets &= (self.__datasets_index.loc[:, dataset_property] == value)

                # Otherwise, either a single value (for exact match) or a list of values (for multiple exact matches) should be given
                else:
                    iterable_value = isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set)
                    newly_selected_datasets &= self.__datasets_index.loc[:, dataset_property].isin(value if iterable_value else [value])

        # Set the flags of the selected datasets to True
        self.__selected_datasets |= newly_selected_datasets

        return self

    def filter_available_datasets(self) -> 'DataManager':
        for dataset_index in self.__selected_datasets.index:

            # Skip datasets that are already not selected
            if not self.__selected_datasets[dataset_index]:
                continue

            # Obtain the metadata, in particular to get the train and test path
            metadata = self.get_metadata(dataset_index)

            # For train data, also check if there should be training data (or if there is only test data)
            if pd.notna(self.get_metadata(dataset_index)['train_path']):
                if not os.path.exists(self.__data_dir + '/' + metadata['train_path']):
                    self.__selected_datasets[dataset_index] = False
                    continue  # Don't have to check the test data anymore

            # The test data must exist
            if not os.path.exists(self.__data_dir + '/' + metadata['test_path']):
                self.__selected_datasets[dataset_index] = False

        return self

    def get(self, index: Optional[int] = None) -> Union[List[DatasetIndex], DatasetIndex]:
        selected_datasets = self.__selected_datasets[self.__selected_datasets].index
        if index is None:
            return selected_datasets.tolist()
        else:
            if index < 0 or index >= selected_datasets.shape[0]:
                raise IndexError(f"Invalid index of a dataset given to the DataManager: '{index}'!"
                                 f"Only {selected_datasets.shape[0]} datasets are selected, thus '0 <= index < {selected_datasets.shape[0]}' must be satisfied")
            return selected_datasets[index]

    def get_metadata(self, dataset_index: DatasetIndex) -> pd.Series:
        self.check_index_exists(dataset_index)
        return self.__datasets_index.loc[dataset_index, :]

    def load(self, dataset_index: DatasetIndex, train: bool = False) -> pd.DataFrame:
        self.check_index_selected(dataset_index)
        path = self.check_data_exists(dataset_index, train)
        return pd.read_csv(path, index_col='timestamp')

    def load_raw_data(self, dataset_index: DatasetIndex, train: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        data_df = self.load(dataset_index, train)
        return data_df[data_df.columns].drop(columns='is_anomaly').values, data_df['is_anomaly'].values

    def check_index_exists(self, dataset_index: DatasetIndex) -> None:
        if dataset_index not in self.__datasets_index.index:
            raise ValueError(f"The dataset '{dataset_index}' does not exist!")

    def check_index_selected(self, dataset_index: DatasetIndex) -> None:
        self.check_index_exists(dataset_index)
        if not self.__selected_datasets.loc[dataset_index]:
            raise ValueError(f"The dataset '{dataset_index}' has not been selected!")

    def check_data_exists(self, dataset_index: DatasetIndex, train: bool = False) -> str:
        if train and pd.isna(self.__datasets_index.loc[dataset_index, 'train_path']):
            raise ValueError(f"Train data is requested, but there is no train data for the dataset with index '{dataset_index}'!")

        path = (self.__data_dir + '/' + self.__datasets_index.loc[dataset_index, 'train_path' if train else 'test_path'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"The {'train' if train else 'test'} data of dataset '{dataset_index}' does not exist!")
        return path

    def add_dataset(self,
                    collection_name: str,
                    dataset_name: str,
                    test_data: pd.DataFrame,
                    test_path: str,
                    dataset_type: str,
                    train_type: str,
                    train_is_normal: bool,
                    trend: str,
                    stationarity: str,
                    train_data: Optional[pd.DataFrame] = None,
                    train_path: Optional[str] = None,
                    period_size: Optional[int] = None,
                    split_at: Optional[int] = None) -> None:

        # Check if the dataset already exists
        dataset_index = (collection_name, dataset_name)
        if dataset_index in self.__datasets_index.index:
            raise ValueError(f"There already exists a dataset with index '{dataset_index}'!")

        # Check the test data and train data if given
        check_data(test_data, test_path, 'test')
        if train_data is not None and train_path is not None:
            check_data(train_data, train_path, 'train')
            if set(train_data) != set(test_data):  # Includes the 'is_anomaly' column
                raise ValueError(f"The train data and test data should have the same number of attributes!")
        elif train_data is not None and train_path is None:
            raise ValueError(f"Train data is given, but there is no train data path given!")
        elif train_data is None and train_path is not None:
            raise ValueError(f"Train data path is given, but there is no train data given!")

        length_anomaly_sequences = []
        current_length = 0
        for anomaly_label in test_data['is_anomaly']:
            if anomaly_label == 1:
                # If the label is 1, then the current anomaly sequence increased in length
                current_length += 1
            else:
                # If the label is 0, then a sequence has ended if the current length is larger than 0
                if current_length > 0:
                    length_anomaly_sequences.append(current_length)
                    current_length = 0
        if current_length > 0:
            length_anomaly_sequences.append(current_length)

        # Create a new row for the dataset index
        new_row = pd.Series(index=_COLUMN_NAMES, data={
            'train_path': train_path,
            'test_path': test_path,
            'dataset_type': dataset_type,
            'datetime_index': set(test_data.index) != set(range(test_data.shape[0])),
            'split_at': split_at,
            'train_type': train_type,
            'train_is_normal': train_is_normal,
            'input_type': 'multivariate' if test_data.shape[1] > 2 else 'univariate',  # '2' to take 'is_anomaly' into account
            'length': test_data.shape[0],
            'dimensions': test_data.shape[1] - 1,  # Exclude the 'is_anomaly' column
            'contamination': test_data['is_anomaly'].sum() / test_data.shape[0],
            'num_anomalies': len(length_anomaly_sequences),
            'min_anomaly_length': min(length_anomaly_sequences) if len(length_anomaly_sequences) > 0 else 0,
            'median_anomaly_length': np.median(length_anomaly_sequences) if len(length_anomaly_sequences) > 0 else 0,
            'max_anomaly_length': max(length_anomaly_sequences) if len(length_anomaly_sequences) > 0 else 0,
            'mean': np.mean(test_data.mean(axis=0).drop('is_anomaly')),
            'stddev': np.mean(test_data.std(axis=0).drop('is_anomaly')),
            'trend': trend,
            'stationarity': stationarity,
            'period_size': period_size
        })

        # Add the row to the dataset index
        self.__datasets_index.loc[dataset_index, :] = new_row
        self.__datasets_index.sort_index(inplace=True)  # Easier for humans to find stuff
        self.__selected_datasets.at[dataset_index] = False
        self.__selected_datasets.sort_index(inplace=True)

        # Save the index and the data
        self.__datasets_index.to_csv(self.__datasets_index_file)
        if not os.path.exists(self.__data_dir + '/' + os.path.dirname(test_path)):
            os.makedirs(self.__data_dir + '/' + os.path.dirname(test_path))
        test_data.to_csv(self.__data_dir + '/' + test_path)
        if train_data is not None:
            if not os.path.exists(os.path.dirname(self.__data_dir + '/' + train_path)):
                os.makedirs(os.path.dirname(self.__data_dir + '/' + train_path))
            train_data.to_csv(self.__data_dir + '/' + train_path)

    def remove_dataset(self, dataset_index: DatasetIndex) -> None:
        self.check_index_exists(dataset_index)
        metadata = self.get_metadata(dataset_index)

        # Remove the data files
        if pd.notna(metadata['train_path']):
            if os.path.exists(self.__data_dir + '/' + metadata['train_path']):
                os.remove(self.__data_dir + '/' + metadata['train_path'])
        if os.path.exists(self.__data_dir + '/' + metadata['test_path']):
            os.remove(self.__data_dir + '/' + metadata['test_path'])

        # Update the index file
        self.__datasets_index.drop(index=dataset_index, inplace=True)
        self.__datasets_index.sort_index(inplace=True)  # Easier for humans to find stuff
        self.__datasets_index.to_csv(self.__datasets_index_file)

        # Update the selected datasets
        self.__selected_datasets.drop(index=dataset_index, inplace=True)
        self.__selected_datasets.sort_index(inplace=True)


def check_data(data: pd.DataFrame, path: str, name: str) -> None:
    # Check if the path already exists
    if os.path.exists(path):
        raise ValueError(f"The given {name} path '{path}' already exist!")
    # Check if the index has the correct name
    if data.index.name != 'timestamp':
        raise ValueError(f"The index of the given {name} data does not have the name 'timestamp', but the name '{data.index.name}'!")
    # Check if there is an is_anomaly column
    if 'is_anomaly' not in data.columns:
        raise ValueError(f"The given {name} data does not contain the column 'is_anomaly'!")
    # Check if the 'is_anomaly' column only contains binary labels (both integers and floats)
    if not data['is_anomaly'].isin([0, 1, 0.0, 1.0, True, False]).all():
        raise ValueError(f"The 'is_anomaly' column of the {name} data should only contain binary labels ('0' or '1'), but contains values!")
