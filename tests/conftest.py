
import pytest
import numpy as np
from dtaianomaly.data_management.DataManager import DataManager


@pytest.fixture
def data_manager() -> DataManager:
    return DataManager('data', 'datasets.csv')


@pytest.fixture
def demo_time_series(data_manager) -> np.ndarray:
    data_manager.select({'collection_name': 'Demo', 'dataset_name': 'Demo1'})
    demo_time_series_index = data_manager.get(0)
    return data_manager.load_raw_data(demo_time_series_index)[0]
