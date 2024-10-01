
import numpy as np
import pytest
from dtaianomaly.data import LazyDataLoader, DataSet, from_directory


class DummyLoader(LazyDataLoader):

    def load(self) -> DataSet:
        return DataSet(x=np.array([]), y=np.array([]))

    def __str__(self) -> str:
        return 'DummyLoader'


class TestLazyDataLoader:

    def test_invalid_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader = DummyLoader(tmp_path / 'some' / 'invalid' / 'path')

    def test_valid_file(self, tmp_path):
        open(tmp_path / 'valid-file', 'a').close()  # Make the file
        loader = DummyLoader(tmp_path / 'valid-file')
        assert loader.path == str(tmp_path / 'valid-file')

    def test_valid_directory(self, tmp_path):
        loader = DummyLoader(tmp_path)
        assert loader.path == str(tmp_path)


class TestFromDirectory:

    def test_no_directory(self):
        with pytest.raises(FileNotFoundError):
            from_directory('some-invalid-path', DummyLoader)

    def test_file_given(self, tmp_path):
        open(tmp_path / 'a-file', 'a').close()  # Make the file
        with pytest.raises(FileNotFoundError):
            from_directory(tmp_path / 'a-file', DummyLoader)

    def test_valid(self, tmp_path):
        paths = [str(tmp_path / f'file-{i}') for i in range(5)]
        for path in paths:
            open(path, 'a').close()  # Make the file
        data_loaders = from_directory(tmp_path, DummyLoader)

        assert len(data_loaders) == len(paths)
        for data_loader in data_loaders:
            assert data_loader.path in paths
