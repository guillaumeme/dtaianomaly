
import numpy as np
import pandas as pd

from visualization import plot_data, plot_anomaly_scores
from anomaly_detection import PYODAnomalyDetector

# Create a synthetic dataset consisting of two sine waves with different frequencies
trend_data = np.concatenate([
    np.sin(np.linspace(0, 25 * np.pi, 1400) + 2) + 10,
    np.sin(np.linspace(0, 9 * np.pi, 1000)) + 9.5
])

# Add Gaussian noise
np.random.seed(42)
trend_data += np.random.normal(0, 0.1, trend_data.shape)

# Convert to dataframe
trend_data_df = pd.DataFrame({'value': trend_data, 'is_anomaly': np.zeros(len(trend_data))})

# # Add an anomaly
start_anomaly = 920
end_anomaly = 965
trend_data[start_anomaly:end_anomaly] -= 0.5
trend_data_df.loc[start_anomaly:end_anomaly, 'value'] -= 0.5
trend_data_df.loc[start_anomaly:end_anomaly, 'is_anomaly'] = 1

# Plot the data
fig = plot_data(trend_data_df, show_ground_truth='inline')
fig.show()


# A function to analyze the two normalization strategies: 'min_max' and 'unify'
def analyze_normalization(anomaly_detector_name: str, w: int) -> None:
    anomaly_detector = PYODAnomalyDetector.load({
        "pyod_model": anomaly_detector_name,
        "windowing": {
            "window_size": w
        }
    })
    anomaly_detector.fit(trend_data)
    decision_scores = anomaly_detector.decision_function(trend_data)
    predicted_proba_min_max = anomaly_detector.predict_proba(trend_data, normalization='min_max')
    predicted_proba_unify = anomaly_detector.predict_proba(trend_data, normalization='unify')
    fig = plot_anomaly_scores(
        trend_data_df,
        anomaly_scores={'decision scores': decision_scores, 'min max': predicted_proba_min_max, 'unify': predicted_proba_unify},
        show_ground_truth='inline'
    )
    fig.suptitle(f"{anomaly_detector_name} with window size {window_size}")
    fig.tight_layout()
    fig.show()


for window_size in [16, 32, 64, 128]:
    analyze_normalization('IForest', window_size)
    analyze_normalization('LOF', window_size)
    analyze_normalization('KNN', window_size)
