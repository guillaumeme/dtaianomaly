import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List

DatasetIndex = Tuple[str, str]


# Inspired by Timeeval (https://github.com/HPI-Information-Systems/TimeEval/tree/main)
class DataManager:

    def __init__(self, data_dir: str, datasets_index_file: str = 'datasets.csv'):
        self.__data_dir: str = data_dir
        self.__datasets_index_file: str = datasets_index_file
        self.__datasets_index: pd.DataFrame = pd.read_csv(self.__data_dir + '/' + self.__datasets_index_file, index_col=['collection_name', 'dataset_name'])
        self.__selected_datasets: pd.DataFrame = pd.DataFrame(index=self.__datasets_index.index, columns=['selected'], data=False)

    def select(self, dataset_properties: Optional[Dict[str, any]] = None) -> None:
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

                # For string-like properties, either a single string (for exact match) or a list of strings (for multiple exact matches) should be given
                elif all((v is None) or isinstance(v, str) for v in self.__datasets_index[dataset_property]):
                    newly_selected_datasets &= self.__datasets_index.loc[:, dataset_property].isin([value] if isinstance(value, str) else value)

                else:
                    raise NotImplementedError(f"The type of property '{dataset_property}' equals '{self.__datasets_index[dataset_property].dtype}', but this type is not yet supported!")

        # Set the flags of the selected datasets to True
        self.__selected_datasets['selected'] |= newly_selected_datasets

    def get(self) -> List[DatasetIndex]:
        return self.__selected_datasets[self.__selected_datasets['selected']].index.tolist()

    def get_metadata(self, dataset_index: DatasetIndex) -> pd.Series:
        self.check_index_exists(dataset_index)
        return self.__datasets_index.loc[dataset_index, :]

    def load(self, dataset_index: DatasetIndex, train: bool = False) -> pd.DataFrame:
        self.check_index_selected(dataset_index)
        path = (self.__data_dir + '/' + self.__datasets_index.loc[dataset_index, 'train_path' if train else 'test_path'])
        return pd.read_csv(path, index_col='timestamp')

    def load_raw_data(self, dataset_index: DatasetIndex, train: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        data_df = self.load(dataset_index, train)
        return data_df[data_df.columns].drop(columns='is_anomaly').values, data_df['is_anomaly'].values

    def check_index_exists(self, dataset_index: DatasetIndex) -> None:
        if dataset_index not in self.__datasets_index.index:
            raise ValueError(f"The dataset '{dataset_index}' does not exist!")

    def check_index_selected(self, dataset_index: DatasetIndex) -> None:
        self.check_index_exists(dataset_index)
        if not self.__selected_datasets.loc[dataset_index, 'selected']:
            raise ValueError(f"The dataset '{dataset_index}' has not been selected!")
