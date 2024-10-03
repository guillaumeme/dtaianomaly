import abc
import os
import numpy as np
from pathlib import Path
from typing import NamedTuple, List, Type, Union

from dtaianomaly.PrettyPrintable import PrettyPrintable


class DataSet(NamedTuple):
    """
    A class for time series anomaly detection data sets. These
    consist of the raw data itself and the ground truth labels.

    Parameters
    ----------
    x: array-like of shape (n_samples, n_features)
        The time series.
    y: array-like of shape (n_samples)
        The ground truth anomaly labels.
    """
    x: np.ndarray
    y: np.ndarray


class LazyDataLoader(PrettyPrintable):
    """
    A lazy dataloader for anomaly detection workflows

    This is a data loading utility to point towards a specific data set
    (with `path`) and to load it at a later point in time during 
    execution of a workflow.

    This way we limit memory usage and allow for virtually unlimited scaling
    of the number of data sets in a workflow.

    Parameters
    ----------
    path: str
        Path to the relevant data set.

    Raises
    ------
    FileNotFoundError
        If the given path does not point to an existing file or directory.
    """
    path: str

    def __init__(self, path: Union[str, Path]) -> None:
        if not (Path(path).is_file() or Path(path).is_dir()):
            raise FileNotFoundError(f'No such file or directory: {path}')
        self.path = str(path)

    @abc.abstractmethod
    def load(self) -> DataSet:
        """
        Load the dataset.

        Returns
        -------
        data_set: DataSet
            The loaded dataset.
        """


def from_directory(directory: Union[str, Path], dataloader: Type[LazyDataLoader]) -> List[LazyDataLoader]:
    """
    Construct a `LazyDataLoader` instance for every file in the given `directory`

    Parameters
    ----------
    directory: str or Path
        Path to the directory in question
    dataloader: LazyDataLoader **object**
        Class object of the data loader, called for constructing
        each data loader instance

    Returns
    -------
    data_loaders: List[LazyDataLoader]
        A list of the initialized data loaders, one for each data set in the
        given directory.

    Raises
    ------
    FileNotFoundError
        If `directory` cannot be found
    """
    if not Path(directory).is_dir():
        raise FileNotFoundError(f'No such directory: {directory}')

    all_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) or os.path.isdir(os.path.join(directory, f))
    ]
    return [dataloader(file) for file in all_files]
