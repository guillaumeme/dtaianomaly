
import numpy as np
import matplotlib.pyplot as plt
from dtaianomaly.visualization import plot_time_series_colored_by_score


class TestVisualization:

    def test_univariate(self, univariate_time_series):
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        fig = plot_time_series_colored_by_score(univariate_time_series, y)
        assert len(fig.get_axes()[0].lines) == univariate_time_series.shape[0] - 1

    def test_multivariate(self, multivariate_time_series):
        y = np.random.choice([0, 1], size=multivariate_time_series.shape[0], replace=True)
        fig = plot_time_series_colored_by_score(multivariate_time_series, y)
        assert len(fig.get_axes()[0].lines) == (multivariate_time_series.shape[0] - 1) * multivariate_time_series.shape[1]

    def test_given_axis(self, univariate_time_series):
        fig = plt.figure()
        plt.axes()
        [ax] = fig.axes
        y = np.random.choice([0, 1], size=univariate_time_series.shape[0], replace=True)
        plot_time_series_colored_by_score(univariate_time_series, y, ax=ax)
        assert len(ax.lines) == univariate_time_series.shape[0] - 1
