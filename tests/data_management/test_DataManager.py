
import pytest
import numpy as np
import pandas as pd
import shutil
import os
from dtaianomaly.data_management.DataManager import DataManager


@pytest.fixture
def dataset_index_file() -> pd.DataFrame:
    return pd.read_csv('data/datasets.csv', index_col=['collection_name', 'dataset_name'])


class TestInitialize:

    def test_invalid_data_directory_path(self):
        with pytest.raises(ValueError):
            _ = DataManager('invalid_path', 'datasets.csv')

    def test_invalid_dataset_index_file_path(self):
        with pytest.raises(ValueError):
            _ = DataManager('data', 'invalid_datasets.csv')

    def test_invalid_dataset_index_file_index(self, tmp_path):
        dataset_index_path = tmp_path / "datasets.csv"
        pd.DataFrame(columns=['collection_name_invalid', 'dataset_name_invalid', 'some_other_column']).to_csv(dataset_index_path, index=False)

        with pytest.raises(IndexError):
            _ = DataManager(str(tmp_path), "datasets.csv")

    def test_invalid_dataset_index_file_missing_properties(self, tmp_path, dataset_index_file):
        dataset_index_path = tmp_path / "datasets.csv"

        required_columns = dataset_index_file.columns
        for _ in range(20):
            columns = list(np.random.choice(required_columns, np.random.randint(1, len(required_columns)-1), replace=False))
            columns.extend(['collection_name', 'dataset_name'])
            pd.DataFrame(columns=columns).to_csv(dataset_index_path, index=False)

            with pytest.raises(ValueError):
                _ = DataManager(str(tmp_path), "datasets.csv")

    def test_invalid_dataset_index_file_extra_properties(self, tmp_path, dataset_index_file):
        dataset_index_path = tmp_path / "datasets.csv"

        required_columns = list(dataset_index_file.columns)
        required_columns.extend(['collection_name', 'dataset_name'])
        required_columns.append('some_other_column')
        pd.DataFrame(columns=required_columns).to_csv(dataset_index_path, index=False)

        with pytest.raises(ValueError):
            _ = DataManager(str(tmp_path), "datasets.csv")

    def test_valid_parameters(self):
        _ = DataManager('data', 'datasets.csv')


class TestSelectGet:

    def test_clear(self, data_manager, dataset_index_file):
        data_manager.select()
        assert len(data_manager.get()) > 0

        data_manager.clear()
        assert len(data_manager.get()) == 0

        data_manager.select({'collection_name': 'Demo'})
        assert len(data_manager.get()) > 0
        assert len(data_manager.get()) < len(dataset_index_file)  # To make sure that not all datasets are selected

    def test_select_all(self, data_manager, dataset_index_file):
        data_manager.select()
        selected_datasets = data_manager.get()
        assert len(selected_datasets) == len(dataset_index_file)
        for selected_dataset_index in selected_datasets:
            assert selected_dataset_index in dataset_index_file.index

    def test_select_invalid_property(self, data_manager):
        with pytest.raises(ValueError):
            data_manager.select({'invalid_property': 'invalid_value'})

    def test_select_collection_name_non_existing(self, data_manager):
        data_manager.select({'collection_name': 'INVALID_COLLECTION_NAME'})
        assert len(data_manager.get()) == 0

    def test_select_collection_name_single(self, data_manager):
        data_manager.select({'collection_name': 'Demo'})
        for selected_dataset_index in data_manager.get():
            assert selected_dataset_index[0] == 'Demo'

    def test_select_collection_name_multiple(self, data_manager):
        data_manager.select({'collection_name': ['Demo', 'KDD-TSAD']})
        for selected_dataset_index in data_manager.get():
            assert selected_dataset_index[0] in ['Demo', 'KDD-TSAD']

    def test_select_dataset_name_non_existing(self, data_manager):
        data_manager.select({'dataset_name': 'INVALID_DATASET_NAME'})
        assert len(data_manager.get()) == 0

    def test_select_dataset_name_single(self, data_manager):
        data_manager.select({'dataset_name': 'Demo1'})
        for selected_dataset_index in data_manager.get():
            assert selected_dataset_index[1] == 'Demo1'

    def test_select_dataset_name_multiple(self, data_manager):
        data_manager.select({'dataset_name': ['Demo1', 'Demo2']})
        for selected_dataset_index in data_manager.get():
            assert selected_dataset_index[1] in ['Demo1', 'Demo2']

    def test_select_index(self, data_manager):
        data_manager.select({'collection_name': 'Demo', 'dataset_name': 'Demo1'})
        for selected_dataset_index in data_manager.get():
            assert selected_dataset_index[0] == 'Demo'
            assert selected_dataset_index[1] == 'Demo1'

    def test_select_bool_non_bool_value(self, data_manager):
        with pytest.raises(ValueError):
            data_manager.select({'datetime_index': None})

    def test_select_bool_multiple_values(self, data_manager):
        with pytest.raises(ValueError):
            data_manager.select({'datetime_index': [True, False]})

    def test_select_bool_property(self, data_manager):
        data_manager.select({'datetime_index': True})
        for selected_dataset_index in data_manager.get():
            assert data_manager.get_metadata(selected_dataset_index)['datetime_index']

        data_manager.clear()
        data_manager.select({'datetime_index': False})
        for selected_dataset_index in data_manager.get():
            assert not data_manager.get_metadata(selected_dataset_index)['datetime_index']

    def test_select_numerical_non_numeric_value(self, data_manager):
        with pytest.raises(ValueError):
            data_manager.select({'length': None})
        data_manager.select({'length': 1000})  # Does not rais an error
        data_manager.select({'contamination': 0.1})  # Does not rais an error

    def test_select_numerical_non_numeric_range(self, data_manager):
        with pytest.raises(ValueError):
            data_manager.select({'length': (None, 1000)})
        with pytest.raises(ValueError):
            data_manager.select({'length': (100, None)})
        with pytest.raises(ValueError):
            data_manager.select({'length': (None, 1000.0)})
        with pytest.raises(ValueError):
            data_manager.select({'length': (100.0, None)})
        with pytest.raises(ValueError):
            data_manager.select({'length': (None, None)})

        data_manager.select({'length': (100, 1000)})   # Does not rais an error
        data_manager.select({'length': (100.0, 1000)})   # Does not rais an error
        data_manager.select({'length': (100, 1000.0)})   # Does not rais an error
        data_manager.select({'length': (100.0, 1000.0)})   # Does not rais an error

    def test_select_numerical_long_list(self, data_manager):
        with pytest.raises(ValueError):
            data_manager.select({'length': [100, 1000, 10000]})

    def test_select_numerical_not_existing(self, data_manager):
        data_manager.select({'dimensions': 0})
        assert len(data_manager.get()) == 0

    def test_select_numerical_single(self, data_manager):
        for dimension in range(1, 10):
            data_manager.select({'dimensions': dimension})
            for selected_dataset_index in data_manager.get():
                assert data_manager.get_metadata(selected_dataset_index)['dimensions'] == dimension
            data_manager.clear()

    def test_select_numerical_range(self, data_manager):
        min_contamination = 0.01
        max_contamination = 0.1
        data_manager.select({'contamination': (min_contamination, max_contamination)})
        for selected_dataset_index in data_manager.get():
            assert min_contamination <= data_manager.get_metadata(selected_dataset_index)['contamination'] <= max_contamination

    def test_select_other_single_value(self, data_manager):
        for train_type in ['supervised', 'unsupervised', 'semi-supervised']:
            data_manager.select({'train_type': train_type})
            for selected_dataset_index in data_manager.get():
                assert data_manager.get_metadata(selected_dataset_index)['train_type'] == train_type
            data_manager.clear()

    def test_select_other_multiple_list(self, data_manager):
        train_types = ['unsupervised', 'semi-supervised']
        assert isinstance(train_types, list)
        data_manager.select({'train_type': train_types})
        for selected_dataset_index in data_manager.get():
            assert data_manager.get_metadata(selected_dataset_index)['train_type'] in train_types

    def test_select_other_multiple_set(self, data_manager):
        train_types = {'unsupervised', 'semi-supervised'}
        assert isinstance(train_types, set)
        data_manager.select({'train_type': train_types})
        for selected_dataset_index in data_manager.get():
            assert data_manager.get_metadata(selected_dataset_index)['train_type'] in train_types

    def test_select_other_multiple_tuple(self, data_manager):
        train_types = ('unsupervised', 'semi-supervised')
        assert isinstance(train_types, tuple)
        data_manager.select({'train_type': train_types})
        for selected_dataset_index in data_manager.get():
            assert data_manager.get_metadata(selected_dataset_index)['train_type'] in train_types

    def test_select_multiple_properties(self, data_manager):
        data_manager.select({
            'collection_name': 'KDD-TSAD',
            'dimensions': 1,
            'length': (100, 20000),
            'median_anomaly_length': (50, 200),
        })
        for selected_dataset_index in data_manager.get():
            # All must be satisfied
            assert selected_dataset_index[0] == 'KDD-TSAD'
            assert data_manager.get_metadata(selected_dataset_index)['dimensions'] == 1
            assert 100 <= data_manager.get_metadata(selected_dataset_index)['length'] <= 20000
            assert 50 <= data_manager.get_metadata(selected_dataset_index)['median_anomaly_length'] <= 200

    def test_select_multiple_calls(self, data_manager):
        data_manager.select({
            'collection_name': 'KDD-TSAD',
            'length': (100, 20000),
        })
        data_manager.select({
            'collection_name': 'Daphnet',
            'num_anomalies': (2, 9),
        })
        for selected_dataset_index in data_manager.get():
            satisfied_first = selected_dataset_index[0] == 'KDD-TSAD' and 100 <= data_manager.get_metadata(selected_dataset_index)['length'] <= 20000
            satisfied_second = selected_dataset_index[0] == 'Daphnet' and 2 <= data_manager.get_metadata(selected_dataset_index)['num_anomalies'] <= 9
            # Either selection should be satisfied
            assert satisfied_first or satisfied_second

    def test_filter_available_datasets(self, data_manager):
        data_manager.select({'collection_name': ['Demo', 'KDD-TSAD', 'Exathlon']})
        nb_initially_selected_datasets = len(data_manager.get())

        data_manager.filter_available_datasets()
        nb_finally_selected_datasets = len(data_manager.get())

        # There should be fewer datasets after filtering
        assert nb_finally_selected_datasets <= nb_initially_selected_datasets

        for selected_dataset_index in data_manager.get():
            try:
                data_manager.check_data_exists(selected_dataset_index, train=True)
            except ValueError:
                pass  # If the train data does not exist
            data_manager.check_data_exists(selected_dataset_index, train=False)

    def test_get_index(self, data_manager):
        data_manager.select()
        selected_datasets = data_manager.get()
        for i in range(len(selected_datasets)):
            assert selected_datasets[i] == selected_datasets[i]

    def test_get_negative_index(self, data_manager):
        with pytest.raises(IndexError):
            data_manager.get(-1)

    def test_get_too_large_index(self, data_manager):
        data_manager.select()
        selected_datasets = data_manager.get()
        with pytest.raises(IndexError):
            data_manager.get(len(selected_datasets))


class TestLoad:

    def test_load_index_does_not_exist(self, data_manager):
        invalid_dataset_index = ('invalid_collection_name', 'invalid_dataset_name')
        with pytest.raises(ValueError):
            data_manager.load(invalid_dataset_index)
        with pytest.raises(ValueError):
            data_manager.load(invalid_dataset_index)

    def test_load_dataset_not_selected(self, data_manager, dataset_index_file):
        dataset_index = dataset_index_file.index[0]
        data_manager.check_index_exists(dataset_index)
        with pytest.raises(ValueError):
            data_manager.check_index_selected(dataset_index)
        with pytest.raises(ValueError):
            data_manager.load(dataset_index)

    def test_load_data_not_available(self, data_manager, dataset_index_file):
        data_manager.select({'collection_name': 'CalIt2'})
        dataset_index = dataset_index_file.index[0]
        data_manager.check_index_exists(dataset_index)
        data_manager.check_index_selected(dataset_index)
        with pytest.raises(FileNotFoundError):
            data_manager.check_data_exists(dataset_index)
        with pytest.raises(FileNotFoundError):
            data_manager.load(dataset_index)

    def test_load(self, data_manager):
        data_manager.select({'collection_name': 'Demo'})
        dataset_index = data_manager.get(0)
        trend_data_df = data_manager.load(dataset_index)

        assert isinstance(trend_data_df, pd.DataFrame)
        assert trend_data_df.shape[0] > 0
        assert trend_data_df.shape[1] == 2  # single attribute + labels

        assert 'is_anomaly' in trend_data_df.columns
        assert trend_data_df.index.name == 'timestamp'

    def test_load_raw_data(self, data_manager):
        data_manager.select({'collection_name': 'Demo'})
        dataset_index = data_manager.get(0)
        trend_data_df = data_manager.load(dataset_index)
        trend_data, ground_truth = data_manager.load_raw_data(dataset_index)

        assert isinstance(trend_data, np.ndarray)
        assert trend_data.shape[0] == trend_data_df.shape[0]
        assert trend_data.shape[1] == trend_data_df.shape[1] - 1  # Remove `is_anomaly` column
        for i in range(trend_data.shape[1]):
            assert np.array_equal(trend_data[:, i], trend_data_df.iloc[:, i].values)

        assert isinstance(ground_truth, np.ndarray)
        assert ground_truth.shape[0] == trend_data_df.shape[0]
        assert len(ground_truth.shape) == 1
        assert np.array_equal(ground_truth, trend_data_df['is_anomaly'].values)

    def test_load_all_data(self, data_manager):
        data_manager.select({'collection_name': ['Demo', 'KDD-TSAD', 'Exathlon']})
        data_manager.filter_available_datasets()

        for dataset_index in data_manager.get():

            try:
                data_manager.load(dataset_index, train=True)
                data_manager.load_raw_data(dataset_index, train=True)
            except ValueError:
                pass  # If the train data does not exist

            data_manager.load(dataset_index, train=False)
            data_manager.load_raw_data(dataset_index, train=False)


class TestAddData:

    @staticmethod
    def add_data_to_data_manager(
            data_manager: DataManager,
            collection_name: str = "some_collection",
            dataset_name: str = "some_dataset",
            test_data: pd.DataFrame = pd.DataFrame(),
            test_path: str = "some_test_path.csv",
            dataset_type: str = "",
            train_type: str = "",
            train_is_normal: bool = True,
            trend: str = "",
            stationarity: str = "",
            train_data: pd.DataFrame = None,
            train_path: str = None,
            period_size: int = None,
            split_at: int = None):
        data_manager.add_dataset(
            collection_name=collection_name,
            dataset_name=dataset_name,
            test_data=test_data,
            test_path=test_path,
            dataset_type=dataset_type,
            train_type=train_type,
            train_is_normal=train_is_normal,
            trend=trend,
            stationarity=stationarity,
            train_data=train_data,
            train_path=train_path,
            period_size=period_size,
            split_at=split_at)

    @pytest.fixture
    def tmp_data_manager(self, tmp_path) -> DataManager:
        shutil.copy('data/datasets.csv', tmp_path / 'datasets.csv')
        shutil.copytree('data/demo', tmp_path / 'demo')  # To make sure that there is some data
        return DataManager(str(tmp_path), 'datasets.csv')

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        raw_data = np.random.rand(1000, 2)
        data = pd.DataFrame(raw_data, columns=['attribute1', 'attribute2'])
        anomaly_scores = np.zeros(1000)
        anomaly_scores[100:200] = 1
        anomaly_scores[500:550] = 1
        anomaly_scores[850:900] = 1
        data['is_anomaly'] = anomaly_scores
        data.index.name = 'timestamp'
        return data

    def test_already_existing_dataset_index(self, tmp_data_manager, data):
        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        dataset_index = tmp_data_manager.get(0)
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name=dataset_index[0], dataset_name=dataset_index[1])

    def test_test_path_already_exists(self, tmp_data_manager, data, tmp_path):
        data.to_csv(tmp_path / 'file.csv')  # Create the file already
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, test_path=str(tmp_path / 'file.csv'))

    def test_invalid_test_index_name(self, tmp_data_manager, data):
        dataset = data.copy()
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=dataset, test_path="test_data.csv")  # This works
        dataset.index.name = 'invalid_index_name'
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat2', test_data=dataset, test_path="test_data2.csv")

    def test_no_test_anomaly_column(self, tmp_data_manager, data):
        dataset = data.copy()
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=dataset, test_path="test_data.csv")  # This works
        dataset.drop(columns=['is_anomaly'], inplace=True)
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat2', test_data=dataset, test_path="test_data2.csv")

    def test_test_invalid_anomaly_column(self, tmp_data_manager, data):
        dataset = data.copy()
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=dataset, test_path="test_data.csv")  # This works
        dataset.at[0, 'is_anomaly'] = 0.5
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat2', test_data=dataset, test_path="test_data2.csv")

    def test_train_path_already_exists(self, tmp_data_manager, data, tmp_path):
        data.to_csv(tmp_path / 'file_train.csv')  # Create the file already
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, train_path=str(tmp_path / 'file_train.csv'))

    def test_invalid_train_index_name(self, tmp_data_manager, data):
        dataset_test = data.copy()
        dataset_train = data.copy()
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=dataset_test, test_path="test_data.csv", train_data=dataset_train, train_path='train_data.csv')  # This works
        dataset_train.index.name = 'invalid_index_name'
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat2', test_data=dataset_test, test_path="test_data2.csv", train_data=dataset_train, train_path='train_data2.csv')

    def test_no_train_anomaly_column(self, tmp_data_manager, data):
        dataset_test = data.copy()
        dataset_train = data.copy()
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=dataset_test, test_path="test_data.csv", train_data=dataset_train, train_path='train_data.csv')  # This works
        dataset_train.drop(columns=['is_anomaly'], inplace=True)
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat2', test_data=dataset_test, test_path="test_data2.csv", train_data=dataset_train, train_path='train_data2.csv')

    def test_train_invalid_anomaly_column(self, tmp_data_manager, data):
        dataset_test = data.copy()
        dataset_train = data.copy()
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat1', test_data=dataset_test, test_path="test_data.csv", train_data=dataset_train, train_path='train_data.csv')  # This works
        dataset_train.at[0, 'is_anomaly'] = 0.5
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat2', test_data=dataset_test, test_path="test_data2.csv", train_data=dataset_train, train_path='train_data2.csv')

    def test_train_test_different_dimension(self, tmp_data_manager, data):
        dataset_test = data.copy()
        dataset_train = data.copy()
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat1', test_data=dataset_test, test_path="test_data.csv", train_data=dataset_train, train_path='train_data.csv')  # This works
        dataset_train.drop(columns=['attribute1'], inplace=True)
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat2', test_data=dataset_test, test_path="test_data2.csv", train_data=dataset_train, train_path='train_data2.csv')

    def test_test_directory_does_not_exist(self, tmp_data_manager, data, tmp_path):
        directory = 'some_directory'
        assert not os.path.exists(tmp_path / directory)
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=data, test_path=directory + "/test_data.csv")
        assert os.path.exists(tmp_path / directory)
        assert os.path.exists(tmp_path / directory / 'test_data.csv')

    def test_test_directory_does_not_exists_recursive(self, tmp_data_manager, data, tmp_path):
        directory = 'some_directory/some_inner_directory'
        assert not os.path.exists(tmp_path / directory)
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=data, test_path=directory + "/test_data.csv")
        assert os.path.exists(tmp_path / directory)
        assert os.path.exists(tmp_path / directory / 'test_data.csv')

    def test_train_directory_does_not_exist(self, tmp_data_manager, data, tmp_path):
        directory = 'some_directory'
        assert not os.path.exists(tmp_path / directory)
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat1', test_data=data, test_path="test_data.csv", train_data=data, train_path=directory + "/train_data.csv")
        assert os.path.exists(tmp_path / directory)
        assert os.path.exists(tmp_path / directory / 'train_data.csv')

    def test_train_directory_does_not_exists_recursive(self, tmp_data_manager, data, tmp_path):
        directory = 'some_directory/some_inner_directory'
        assert not os.path.exists(tmp_path / directory)
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat1', test_data=data, test_path="test_data.csv", train_data=data, train_path=directory + "/train_data.csv")
        assert os.path.exists(tmp_path / directory)
        assert os.path.exists(tmp_path / directory / 'train_data.csv')

    def test_train_data_but_no_train_path(self, tmp_data_manager, data, tmp_path):
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat1', test_data=data, test_path="test_data.csv", train_data=data, train_path='train_data.csv')
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat2', test_data=data, test_path="test_data.csv", train_data=data)

    def test_train_path_but_no_train_data(self, tmp_data_manager, data, tmp_path):
        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat1', test_data=data, test_path="test_data.csv", train_data=data, train_path='train_data.csv')
        with pytest.raises(ValueError):
            self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat2', test_data=data, test_path="test_data.csv", train_path='train_data.csv')

    def test_dataset_index_assignment_univariate(self, tmp_data_manager, data):
        # univariate
        data.drop(columns=['attribute2'], inplace=True)
        raw_data = data['attribute1'].values

        self.add_data_to_data_manager(
            tmp_data_manager,
            collection_name="collection",
            dataset_name="data1",
            test_data=data.copy(),
            test_path="test_data.csv",
            dataset_type="synthetic",
            train_type="semi-supervised",
            train_is_normal=True,
            trend="kubic trend",
            stationarity="trend_stationary",
            train_data=data.copy(),
            train_path="train_data.csv",
            period_size=100,
            split_at=None
        )

        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('collection', 'data1') in tmp_data_manager.get()

        tmp_data_manager.clear()
        tmp_data_manager.select({'collection_name': 'collection', 'dataset_name': 'data1'})
        assert len(tmp_data_manager.get()) == 1

        metadata = tmp_data_manager.get_metadata(tmp_data_manager.get(0))
        assert metadata['test_path'] == 'test_data.csv'
        assert metadata['train_path'] == 'train_data.csv'
        assert metadata['dataset_type'] == 'synthetic'
        assert metadata['train_type'] == 'semi-supervised'
        assert metadata['train_is_normal']
        assert not metadata['datetime_index']
        assert metadata['trend'] == 'kubic trend'
        assert metadata['stationarity'] == 'trend_stationary'
        assert metadata['period_size'] == 100
        assert pd.isna(metadata['split_at'])
        assert metadata['input_type'] == 'univariate'
        assert metadata['length'] == 1000
        assert metadata['dimensions'] == 1
        assert metadata['contamination'] == 200/1000
        assert metadata['num_anomalies'] == 3
        assert metadata['min_anomaly_length'] == 50
        assert metadata['median_anomaly_length'] == 50
        assert metadata['max_anomaly_length'] == 100
        assert metadata['mean'] == pytest.approx(raw_data.mean())
        assert metadata['stddev'] == pytest.approx(raw_data.std(), rel=1e-2)

    def test_dataset_index_assignment_multivariate(self, tmp_data_manager, data):
        raw_data_1 = data['attribute1'].values
        raw_data_2 = data['attribute2'].values

        self.add_data_to_data_manager(
            tmp_data_manager,
            collection_name="collection",
            dataset_name="data1",
            test_data=data.copy(),
            test_path="test_data.csv",
            dataset_type="real",
            train_type="supervised",
            train_is_normal=True,
            trend="linear trend",
            stationarity="stationary",
            train_data=data.copy(),
            train_path="train_data.csv",
            period_size=100,
            split_at=None
        )

        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('collection', 'data1') in tmp_data_manager.get()

        tmp_data_manager.clear()
        tmp_data_manager.select({'collection_name': 'collection', 'dataset_name': 'data1'})
        assert len(tmp_data_manager.get()) == 1

        metadata = tmp_data_manager.get_metadata(tmp_data_manager.get(0))
        assert metadata['test_path'] == 'test_data.csv'
        assert metadata['train_path'] == 'train_data.csv'
        assert metadata['dataset_type'] == 'real'
        assert metadata['train_type'] == 'supervised'
        assert metadata['train_is_normal']
        assert not metadata['datetime_index']
        assert metadata['trend'] == 'linear trend'
        assert metadata['stationarity'] == 'stationary'
        assert metadata['period_size'] == 100
        assert pd.isna(metadata['split_at'])
        assert metadata['input_type'] == 'multivariate'
        assert metadata['length'] == 1000
        assert metadata['dimensions'] == 2
        assert metadata['contamination'] == 200/1000
        assert metadata['num_anomalies'] == 3
        assert metadata['min_anomaly_length'] == 50
        assert metadata['median_anomaly_length'] == 50
        assert metadata['max_anomaly_length'] == 100
        assert metadata['mean'] == pytest.approx((raw_data_1.mean() + raw_data_2.mean()) / 2)
        assert metadata['stddev'] == pytest.approx((raw_data_1.std() + raw_data_2.std()) / 2, rel=1e-2)

    def test_dataset_index_assignment_anomaly_at_end(self, tmp_data_manager, data):
        raw_data_1 = data['attribute1'].values
        raw_data_2 = data['attribute2'].values

        data['is_anomaly'][990:] = 1
        # data['is_anomaly'][-1] = 1

        self.add_data_to_data_manager(
            tmp_data_manager,
            collection_name="collection",
            dataset_name="data1",
            test_data=data.copy(),
            test_path="test_data.csv",
            dataset_type="real",
            train_type="supervised",
            train_is_normal=True,
            trend="linear trend",
            stationarity="stationary",
            train_data=data.copy(),
            train_path="train_data.csv",
            period_size=None,
            split_at=None
        )

        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('collection', 'data1') in tmp_data_manager.get()

        tmp_data_manager.clear()
        tmp_data_manager.select({'collection_name': 'collection', 'dataset_name': 'data1'})
        assert len(tmp_data_manager.get()) == 1

        metadata = tmp_data_manager.get_metadata(tmp_data_manager.get(0))
        assert metadata['test_path'] == 'test_data.csv'
        assert metadata['train_path'] == 'train_data.csv'
        assert metadata['dataset_type'] == 'real'
        assert metadata['train_type'] == 'supervised'
        assert metadata['train_is_normal']
        assert not metadata['datetime_index']
        assert metadata['trend'] == 'linear trend'
        assert metadata['stationarity'] == 'stationary'
        assert pd.isna(metadata['period_size'])
        assert pd.isna(metadata['split_at'])
        assert metadata['input_type'] == 'multivariate'
        assert metadata['length'] == 1000
        assert metadata['dimensions'] == 2
        assert metadata['contamination'] == pytest.approx(210/1000)
        assert metadata['num_anomalies'] == 4
        assert metadata['min_anomaly_length'] == 10
        assert metadata['median_anomaly_length'] == 50
        assert metadata['max_anomaly_length'] == 100
        assert metadata['mean'] == pytest.approx((raw_data_1.mean() + raw_data_2.mean()) / 2)
        assert metadata['stddev'] == pytest.approx((raw_data_1.std() + raw_data_2.std()) / 2, rel=1e-2)

    def test_dataset_index_assignment_no_train_data(self, tmp_data_manager, data):
        raw_data_1 = data['attribute1'].values
        raw_data_2 = data['attribute2'].values

        self.add_data_to_data_manager(
            tmp_data_manager,
            collection_name="collection",
            dataset_name="data1",
            test_data=data.copy(),
            test_path="test_data.csv",
            dataset_type="real",
            train_type="unsupervised",
            train_is_normal=False,
            trend="no trend",
            stationarity="difference stationary",
            period_size=100,
            split_at=None
        )

        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('collection', 'data1') in tmp_data_manager.get()

        tmp_data_manager.clear()
        tmp_data_manager.select({'collection_name': 'collection', 'dataset_name': 'data1'})
        assert len(tmp_data_manager.get()) == 1

        metadata = tmp_data_manager.get_metadata(tmp_data_manager.get(0))
        assert metadata['test_path'] == 'test_data.csv'
        assert pd.isna(metadata['train_path'])
        assert metadata['dataset_type'] == 'real'
        assert metadata['train_type'] == 'unsupervised'
        assert not metadata['train_is_normal']
        assert not metadata['datetime_index']
        assert metadata['trend'] == 'no trend'
        assert metadata['stationarity'] == 'difference stationary'
        assert metadata['period_size'] == 100
        assert pd.isna(metadata['split_at'])
        assert metadata['input_type'] == 'multivariate'
        assert metadata['length'] == 1000
        assert metadata['dimensions'] == 2
        assert metadata['contamination'] == 200 / 1000
        assert metadata['num_anomalies'] == 3
        assert metadata['min_anomaly_length'] == 50
        assert metadata['median_anomaly_length'] == 50
        assert metadata['max_anomaly_length'] == 100
        assert metadata['mean'] == pytest.approx((raw_data_1.mean() + raw_data_2.mean()) / 2)
        assert metadata['stddev'] == pytest.approx((raw_data_1.std() + raw_data_2.std()) / 2, rel=1e-2)

    def test_dataset_index_assignment_datetime_index(self, tmp_data_manager, data):
        data.index = ['second_' + str(i) for i in data.index]
        data.index.name = 'timestamp'

        raw_data_1 = data['attribute1'].values
        raw_data_2 = data['attribute2'].values

        self.add_data_to_data_manager(
            tmp_data_manager,
            collection_name="collection",
            dataset_name="data1",
            test_data=data.copy(),
            test_path="test_data.csv",
            dataset_type="real",
            train_type="unsupervised",
            train_is_normal=False,
            trend="no trend",
            stationarity="difference stationary",
            period_size=100,
            split_at=None
        )

        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('collection', 'data1') in tmp_data_manager.get()

        tmp_data_manager.clear()
        tmp_data_manager.select({'collection_name': 'collection', 'dataset_name': 'data1'})
        assert len(tmp_data_manager.get()) == 1

        metadata = tmp_data_manager.get_metadata(tmp_data_manager.get(0))
        assert metadata['test_path'] == 'test_data.csv'
        assert pd.isna(metadata['train_path'])
        assert metadata['dataset_type'] == 'real'
        assert metadata['train_type'] == 'unsupervised'
        assert not metadata['train_is_normal']
        assert metadata['datetime_index']
        assert metadata['trend'] == 'no trend'
        assert metadata['stationarity'] == 'difference stationary'
        assert metadata['period_size'] == 100
        assert pd.isna(metadata['split_at'])
        assert metadata['input_type'] == 'multivariate'
        assert metadata['length'] == 1000
        assert metadata['dimensions'] == 2
        assert metadata['contamination'] == 200 / 1000
        assert metadata['num_anomalies'] == 3
        assert metadata['min_anomaly_length'] == 50
        assert metadata['median_anomaly_length'] == 50
        assert metadata['max_anomaly_length'] == 100
        assert metadata['mean'] == pytest.approx((raw_data_1.mean() + raw_data_2.mean()) / 2)
        assert metadata['stddev'] == pytest.approx((raw_data_1.std() + raw_data_2.std()) / 2, rel=1e-2)

    def test_remove_test_data(self, tmp_data_manager, data, tmp_path):
        assert not os.path.exists(tmp_path / 'test_data.csv')

        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=data, test_path="test_data.csv")
        assert os.path.exists(tmp_path / 'test_data.csv')
        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('Col', 'dat1') in tmp_data_manager.get()

        tmp_data_manager.clear()
        tmp_data_manager.remove_dataset(('Col', 'dat1'))
        assert not os.path.exists(tmp_path / 'test_data.csv')
        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert not ('Col', 'dat1') in tmp_data_manager.get()

    def test_remove_test_and_train_data(self, tmp_data_manager, data, tmp_path):
        assert not os.path.exists(tmp_path / 'test_data.csv')
        assert not os.path.exists(tmp_path / 'train_data.csv')

        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col',  dataset_name='dat1', test_data=data.copy(), test_path="test_data.csv", train_data=data.copy(), train_path='train_data.csv')
        assert os.path.exists(tmp_path / 'test_data.csv')
        assert os.path.exists(tmp_path / 'train_data.csv')
        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('Col', 'dat1') in tmp_data_manager.get()

        tmp_data_manager.clear()
        tmp_data_manager.remove_dataset(('Col', 'dat1'))
        assert not os.path.exists(tmp_path / 'test_data.csv')
        assert not os.path.exists(tmp_path / 'train_data.csv')
        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert not ('Col', 'dat1') in tmp_data_manager.get()

    def test_remove_data_selected_index(self, tmp_data_manager, data, tmp_path):
        assert not os.path.exists(tmp_path / 'test_data.csv')

        self.add_data_to_data_manager(tmp_data_manager, collection_name='Col', dataset_name='dat1', test_data=data, test_path="test_data.csv")  # This works
        assert os.path.exists(tmp_path / 'test_data.csv')
        tmp_data_manager.select()
        tmp_data_manager.filter_available_datasets()
        assert ('Col', 'dat1') in tmp_data_manager.get()

        tmp_data_manager.select()  # Make sure it is selected
        tmp_data_manager.remove_dataset(('Col', 'dat1'))
        assert not os.path.exists(tmp_path / 'test_data.csv')

        tmp_data_manager.select({'collection_name': 'Col', 'dataset_name': 'dat1'})  # Try to select the data
        tmp_data_manager.filter_available_datasets()
        assert not ('Col', 'dat1') in tmp_data_manager.get()
