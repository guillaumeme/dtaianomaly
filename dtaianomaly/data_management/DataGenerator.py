
import numpy as np
from typing import Optional


class DataGenerator:
    """
    An object to generate time series.

    Note
    ----
    The generated time series are not representative for real world data. Therefore,
    they should only be used for testing the implementation of algorithms or for
    sanity checks.
    """

    @staticmethod
    def random_time_series(length: int, dimension: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a uniform random time series of the given length with the given number of dimensions.

        Parameters
        ----------
        length : int
            The length of the time series to generate.
        dimension : int, default=1
            The dimension of the time series to generate, i.e., the number of attributes.
        seed : int, default=None
            The seed to use for the random number generator.

        Returns
        -------
        np.ndarray
            A uniform random time series with values from the half-open interval [0.0, 1.0) of
            size (length, dimensions).
        """
        return np.random.default_rng(seed).random(size=(length, dimension))
