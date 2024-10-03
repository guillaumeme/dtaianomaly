import pytest
import numpy as np

from dtaianomaly.preprocessing import SamplingRateUnderSampler, NbSamplesUnderSampler


class TestSamplingRateUnderSampler:

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            SamplingRateUnderSampler(0)

    def test(self):
        x = np.array([4, 9, 2, 5, 4, 7, 4, 5])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        preprocessor = SamplingRateUnderSampler(2)
        x_, y_ = preprocessor.fit_transform(x, y)
        assert np.array_equal(x_, np.array([4, 2, 4, 4]))
        assert np.array_equal(y_, np.array([0, 0, 0, 0]))

    def test_univariate(self, univariate_time_series):
        x_, _ = SamplingRateUnderSampler(42).fit_transform(univariate_time_series)
        assert x_.shape[0] == np.ceil(univariate_time_series.shape[0] / 42)
        assert x_[0] == univariate_time_series[0]
        assert x_[1] == univariate_time_series[42]
        assert x_[2] == univariate_time_series[84]

    def test_multivariate(self, multivariate_time_series):
        x_, _ = SamplingRateUnderSampler(42).fit_transform(multivariate_time_series)
        assert x_.shape[0] == np.ceil(multivariate_time_series.shape[0] / 42)
        assert x_.shape[1] == multivariate_time_series.shape[1]
        assert np.array_equal(x_[0], multivariate_time_series[0])
        assert np.array_equal(x_[1], multivariate_time_series[42])
        assert np.array_equal(x_[2], multivariate_time_series[84])

    def test_too_large_sampling_rate(self, univariate_time_series):
        with pytest.raises(ValueError):
            SamplingRateUnderSampler(univariate_time_series.shape[0]).fit_transform(univariate_time_series)

    def test_str(self):
        assert str(SamplingRateUnderSampler(42)) == 'SamplingRateUnderSampler(sampling_rate=42)'


class TestNbSamplesUnderSampler:

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            NbSamplesUnderSampler(1)

    def test(self):
        x = np.array([4, 9, 2, 5, 4, 7, 4, 5, 8])
        y = np.array([0, 1, 0, 0, 0, 1, 0, 1, 1])
        preprocessor = NbSamplesUnderSampler(5)
        x_, y_ = preprocessor.fit_transform(x, y)
        assert np.array_equal(x_, np.array([4, 2, 4, 4, 8]))
        assert np.array_equal(y_, np.array([0, 0, 0, 0, 1]))

    def test_univariate(self, univariate_time_series):
        x_, _ = NbSamplesUnderSampler(42).fit_transform(univariate_time_series)
        assert x_.shape[0] == 42
        assert x_[0] == univariate_time_series[0]
        assert x_[-1] == univariate_time_series[-1]

    def test_multivariate(self, multivariate_time_series):
        x_, _ = NbSamplesUnderSampler(42).fit_transform(multivariate_time_series)
        assert x_.shape[0] == 42
        assert x_.shape[1] == multivariate_time_series.shape[1]
        assert np.array_equal(x_[0], multivariate_time_series[0])
        assert np.array_equal(x_[-1], multivariate_time_series[-1])

    def test_nb_samples_equal_to_length(self, univariate_time_series):
        x_, _ = NbSamplesUnderSampler(univariate_time_series.shape[0]).fit_transform(univariate_time_series)
        assert np.array_equal(x_, univariate_time_series)

    def test_nb_samples_longer_than_length(self, univariate_time_series):
        x_, _ = NbSamplesUnderSampler(univariate_time_series.shape[0] + 1).fit_transform(univariate_time_series)
        assert np.array_equal(x_, univariate_time_series)

    def test_only_two_samples(self, univariate_time_series):
        x_, _ = NbSamplesUnderSampler(2).fit_transform(univariate_time_series)
        assert x_.shape[0] == 2
        assert x_[0] == univariate_time_series[0]
        assert x_[1] == univariate_time_series[-1]

    def test_str(self):
        assert str(NbSamplesUnderSampler(42)) == 'NbSamplesUnderSampler(nb_samples=42)'
