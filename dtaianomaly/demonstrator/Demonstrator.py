import streamlit as st
import pandas as pd
import numpy as np
import inspect
import matplotlib.pyplot as plt
import traceback
import time
from typing import List, Dict, Any, Union
from scipy import stats
import seaborn as sns
import plotly.express as px  # Import Plotly Express for bar charts

from dtaianomaly import data, anomaly_detection, preprocessing, evaluation, thresholding
from dtaianomaly.pipeline import EvaluationPipeline
from dtaianomaly.workflow.utils import convert_to_proba_metrics
from dtaianomaly.visualization import (
    plot_anomaly_scores, plot_demarcated_anomalies, plot_time_series_colored_by_score,
    plot_time_series_anomalies, plot_with_zoom, format_time_steps
)
from dtaianomaly.utils import is_valid_array_like, is_univariate, get_dimension

# Explicitly import metrics for robustness
from dtaianomaly.evaluation import AreaUnderROC, Precision

# Streamlit app title and description
st.title("dtaianomaly Demonstrator")
st.markdown(
    """
    Een no-code demonstrator voor de **dtaianomaly** bibliotheek, waarmee je interactief anomaliedetectietechnieken 
    voor tijdreeksdata kunt verkennen. Deze tool ondersteunt zowel kwalitatieve evaluatie (visualisaties) als 
    kwantitatieve evaluatie (benchmarking en prestatieconclusies) met vergelijking van meerdere detectoren.
    """
)

# --- Helper Functions for Dynamic Loading and Options ---

def get_available_options(module, base_class, include_functions=False):
    """Dynamically retrieves available options from a dtaianomaly module."""
    options = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class and
                name not in ['BaseDetector', 'PyODAnomalyDetector', 'Preprocessor', 'Metric', 'ProbaMetric',
                             'BinaryMetric', 'ThresholdMetric', 'BestThresholdMetric', 'Thresholding',
                             'LazyDataLoader']):
            options.append(name)
        elif include_functions and inspect.isfunction(obj) and 'time_series' in name:
            options.append(name)
    return options

def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a dataset from dtaianomaly.data or dtaianomaly.data.synthetic."""
    if hasattr(data, dataset_name):
        dataset_function = getattr(data, dataset_name)
    else:
        dataset_function = getattr(data.synthetic, dataset_name)
    x, y = dataset_function()
    if not is_valid_array_like(x) or not is_valid_array_like(y) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Invalid dataset: must be array-like with binary labels (0 or 1).")
    return x, y

def load_component(module, component_name: str, **kwargs):
    """Loads a component (e.g., preprocessor, detector) from a dtaianomaly module."""
    component_class = getattr(module, component_name)
    return component_class(**kwargs)

def generate_hyperparam_inputs(detector_name: str, prefix: str) -> Dict[str, Any]:
    """Generates input fields for detector hyperparameters with a unique prefix."""
    detector_class = getattr(anomaly_detection, detector_name)
    signature = inspect.signature(detector_class.__init__)
    hyperparams = {}
    for param_name, param_obj in signature.parameters.items():
        if param_name in ['self', 'kwargs']:
            continue
        default_value = param_obj.default if param_obj.default != inspect.Parameter.empty else (
            10 if param_obj.annotation == int else 1.0 if param_obj.annotation == float else
            False if param_obj.annotation == bool else "" if param_obj.annotation == str else None)

        if param_name == "window_size":
            window_size_option = st.selectbox(
                f"{prefix} Window Size Option", ["Auto (fft)", "Manual"], key=f"{prefix}_window_size_option", index=0
            )
            if window_size_option == "Manual":
                hyperparams['window_size'] = int(st.number_input(
                    f"{prefix} Manual Window Size", min_value=1, value=20 if default_value is None else default_value,
                    key=f"{prefix}_window_size_manual"
                ))
            else:
                hyperparams['window_size'] = 'fft'
        else:
            if isinstance(default_value, bool):
                hyperparams[param_name] = st.checkbox(f"{prefix} {param_name}", value=default_value,
                                                      key=f"{prefix}_{param_name}")
            elif isinstance(default_value, int):
                hyperparams[param_name] = st.number_input(f"{prefix} {param_name}", value=default_value, step=1,
                                                          format="%d", key=f"{prefix}_{param_name}")
            elif isinstance(default_value, float):
                hyperparams[param_name] = st.number_input(f"{prefix} {param_name}", value=default_value, step=0.1,
                                                          format="%.2f", key=f"{prefix}_{param_name}")
            elif isinstance(default_value, str):
                hyperparams[param_name] = st.text_input(f"{prefix} {param_name}", value=default_value,
                                                        key=f"{prefix}_{param_name}")
            else:
                st.write(f"Parameter {param_name} has an unsupported type. Skipped.")
    return hyperparams

# --- Sidebar for User Selections ---
def configure_sidebar():
    """Configures the Streamlit sidebar for user selections."""
    with st.sidebar:
        st.header("Configuratie")

        # Dataset Selection
        st.subheader("1. Dataset")
        dataset_options = get_available_options(data, data.LazyDataLoader, True) + get_available_options(data.synthetic,
                                                                                                         object, True)
        st.session_state.selected_dataset_name = st.selectbox(
            "Selecteer Dataset", dataset_options, key="dataset_select",
            index=dataset_options.index(
                'demonstration_time_series') if 'demonstration_time_series' in dataset_options else 0
        )

        # Preprocessing Selection
        st.subheader("2. Preprocessing")
        preprocess_options = ["Geen"] + get_available_options(preprocessing, preprocessing.Preprocessor)
        st.session_state.selected_preprocess_name = st.selectbox("Selecteer Preprocessing", preprocess_options,
                                                                 key="preprocess_select", index=0)

        # Preprocessor Hyperparameters
        st.subheader("3. Preprocessor Hyperparameters")
        st.session_state.preprocess_hyperparams = {}
        if st.session_state.selected_preprocess_name != "Geen":
            preprocessor_class = getattr(preprocessing, st.session_state.selected_preprocess_name)
            signature = inspect.signature(preprocessor_class.__init__)
            for param_name, param_obj in signature.parameters.items():
                if param_name in ['self', 'kwargs']:
                    continue
                default_value = param_obj.default if param_obj.default != inspect.Parameter.empty else (
                    1 if param_obj.annotation == int else 0.1 if param_obj.annotation == float else
                    False if param_obj.annotation == bool else "" if param_obj.annotation == str else None)

                if param_obj.annotation == int and param_name in ["window_size", "order", "sampling_rate",
                                                                  "nb_samples"]:
                    st.session_state.preprocess_hyperparams[param_name] = int(
                        st.number_input(f"{param_name}", value=default_value, step=1, format="%d",
                                        key=f"preprocess_{param_name}"))
                elif param_obj.annotation == float and param_name == "alpha":
                    st.session_state.preprocess_hyperparams[param_name] = float(
                        st.number_input(f"{param_name}", value=default_value, step=0.1, format="%.2f",
                                        key=f"preprocess_{param_name}"))
                elif param_obj.annotation == tuple and param_name == "quantile_range":
                    user_input = st.text_input(f"{param_name}", value=str(default_value),
                                               key=f"preprocess_{param_name}")
                    try:
                        st.session_state.preprocess_hyperparams[param_name] = eval(user_input)
                    except:
                        st.error(f"Invalid input for {param_name}. Using default value.")
                        st.session_state.preprocess_hyperparams[param_name] = default_value
                elif param_obj.annotation == bool:
                    st.session_state.preprocess_hyperparams[param_name] = st.checkbox(f"{param_name}",
                                                                                      value=default_value,
                                                                                      key=f"preprocess_{param_name}")
                elif param_obj.annotation == str:
                    st.session_state.preprocess_hyperparams[param_name] = st.text_input(f"{param_name}",
                                                                                        value=default_value,
                                                                                        key=f"preprocess_{param_name}")
                else:
                    st.write(f"Parameter {param_name} has an unsupported type. Skipped.")

        # Anomaly Detectors Selection
        st.subheader("4. Anomaliedetectoren")
        num_detectors = st.number_input("Aantal Detectoren", min_value=1, value=2, step=1, key="num_detectors")
        detector_options = get_available_options(anomaly_detection, anomaly_detection.BaseDetector)
        st.session_state.selected_detectors = []
        st.session_state.detector_hyperparams = {}
        for i in range(num_detectors):
            detector_key = f"detector_{i+1}"
            selected_detector = st.selectbox(f"Selecteer Detector {i+1}", detector_options, key=detector_key)
            st.session_state.selected_detectors.append(selected_detector)
            st.subheader(f"Detector {i+1} Hyperparameters")
            hyperparams = generate_hyperparam_inputs(selected_detector, prefix=detector_key)
            st.session_state.detector_hyperparams[detector_key] = hyperparams

        # Evaluation Metrics Selection
        st.subheader("5. Evaluatiemetrics")
        metric_options = get_available_options(evaluation, evaluation.Metric)
        st.session_state.selected_metrics = st.multiselect("Selecteer Metrics", metric_options, default=["AreaUnderROC"],
                                                           key="metrics_select")

        # Thresholding Selection
        st.subheader("6. Thresholding (voor Binaire Metrics)")
        threshold_options = get_available_options(thresholding, thresholding.Thresholding)
        valid_default_thresholds = [opt for opt in ["FixedThreshold"] if opt in threshold_options] or [
            threshold_options[0]] if threshold_options else []
        st.session_state.selected_thresholds = st.multiselect("Selecteer Thresholds", threshold_options,
                                                              default=valid_default_thresholds, key="thresholds_select")

        # Threshold Hyperparameters
        st.subheader("7. Threshold Hyperparameters")
        st.session_state.threshold_hyperparams = {}
        for threshold_name in st.session_state.selected_thresholds:
            st.session_state.threshold_hyperparams[threshold_name] = {}
            threshold_class = getattr(thresholding, threshold_name)
            signature = inspect.signature(threshold_class.__init__)
            for param_name, param_obj in signature.parameters.items():
                if param_name in ['self', 'kwargs']:
                    continue
                default_value = param_obj.default if param_obj.default != inspect.Parameter.empty else (
                    0.5 if param_obj.annotation == float else 1 if param_obj.annotation == int else None)
                if isinstance(default_value, float):
                    st.session_state.threshold_hyperparams[threshold_name][param_name] = st.number_input(
                        f"{threshold_name} - {param_name}", value=default_value, step=0.1, format="%.2f",
                        key=f"threshold_{threshold_name}_{param_name}"
                    )
                elif isinstance(default_value, int):
                    st.session_state.threshold_hyperparams[threshold_name][param_name] = st.number_input(
                        f"{threshold_name} - {param_name}", value=default_value, step=1, format="%d",
                        key=f"threshold_{threshold_name}_{param_name}"
                    )
                else:
                    st.write(f"Parameter {param_name} for {threshold_name} has an unsupported type. Skipped.")

        # Visualization Options
        st.subheader("8. Visualisatie-opties")
        visualization_options = ["Anomaliescores", "Afgebakende Anomalieën", "Tijdreeks Gekleurd door Score",
                                 "Tijdreeks met Anomalieën", "Zoomweergave"]
        st.session_state.selected_visualizations = st.multiselect("Selecteer Visualisaties", visualization_options,
                                                                  default=["Anomaliescores"],
                                                                  key="visualizations_select")

        # Zoom Range (for Zoomweergave)
        if "Zoomweergave" in st.session_state.selected_visualizations:
            st.subheader("9. Zoom Bereik")
            st.session_state.zoom_start = st.number_input("Start Index", min_value=0, value=0, step=1, key="zoom_start")
            st.session_state.zoom_end = st.number_input("Eind Index", min_value=1, value=100, step=1, key="zoom_end")

# --- Main Content: Run Pipeline and Display Results ---
def run_pipeline():
    """Executes the anomaly detection pipeline and displays qualitative and quantitative results."""
    st.header("Resultaten")

    # Dataset Laden
    try:
        x, y = load_dataset(st.session_state.selected_dataset_name)
        st.write("Dataset succesvol geladen.")
        st.write(f"Tijdreeks dimensionaliteit: {get_dimension(x)}")
        st.write(f"Univariate tijdreeks: {is_univariate(x)}")
    except Exception as e:
        st.error(f"Fout bij laden dataset: {e}")
        return

    # Preprocessor Laden
    try:
        preprocessor = load_component(preprocessing, st.session_state.selected_preprocess_name,
                                      **st.session_state.preprocess_hyperparams) if st.session_state.selected_preprocess_name != "Geen" else None
        st.write("Preprocessor succesvol geconfigureerd.")
    except Exception as e:
        st.error(f"Fout bij configureren preprocessor: {e}")
        return

    # Preprocessing Toepassen
    x_processed, y_processed = preprocessor.fit_transform(x, y) if preprocessor else (x, y)

    # Thresholds en Metrics Laden
    try:
        thresholds = [load_component(thresholding, name, **st.session_state.threshold_hyperparams.get(name, {})) for
                      name in st.session_state.selected_thresholds]
        metrics = [load_component(evaluation, name) for name in st.session_state.selected_metrics]
        st.write("Thresholds en metrics succesvol geconfigureerd.")
    except Exception as e:
        st.error(f"Fout bij configureren thresholds of metrics: {e}")
        return

    # Alle Detectoren Verwerken
    results = {}
    for i, detector_name in enumerate(st.session_state.selected_detectors):
        detector_key = f"detector_{i+1}"
        hyperparams = st.session_state.detector_hyperparams[detector_key]
        detector_label = f"Detector {i+1} ({detector_name})"
        try:
            detector = load_component(anomaly_detection, detector_name, **hyperparams)
            start_time = time.time()
            detector.fit(x_processed, y_processed)
            fit_time = time.time() - start_time
            start_time = time.time()
            anomaly_scores = (detector.predict_proba(x_processed) if hasattr(detector, 'predict_proba')
                              else detector.decision_function(x_processed))
            predict_time = time.time() - start_time
            thresholded_predictions = anomaly_scores.copy()
            for threshold_obj in thresholds:
                if hasattr(threshold_obj, 'threshold'):
                    threshold_value = threshold_obj.threshold(anomaly_scores)
                    thresholded_predictions = (anomaly_scores >= threshold_value).astype(int)
            metric_results = {}
            for metric in metrics:
                metric_name = metric.__class__.__name__
                metric_results[metric_name] = (metric.compute(y_processed, thresholded_predictions)
                                               if isinstance(metric, evaluation.BinaryMetric)
                                               else metric.compute(y_processed, anomaly_scores))
            results[detector_label] = {
                'anomaly_scores': anomaly_scores,
                'thresholded_predictions': thresholded_predictions,
                'metrics': metric_results,
                'fit_time': fit_time,
                'predict_time': predict_time
            }
            st.write(f"{detector_label} succesvol verwerkt. Fit-tijd: {fit_time:.2f} seconden, Predict-tijd: {predict_time:.2f} seconden.")
        except Exception as e:
            st.error(f"Fout bij {detector_label}: {e}")
            traceback.print_exc()

    # Kwalitatieve Evaluatie: Visualisaties
    st.subheader("Kwalitatieve Evaluatie: Visualisaties")
    try:
        time_steps = format_time_steps(None, x_processed.shape[0])
        for detector_label in results.keys():
            st.write(f"{detector_label}")
            for viz in st.session_state.selected_visualizations:
                st.write(f"{viz}")
                fig = generate_visualization(viz, x_processed, y_processed,
                                             results[detector_label]['anomaly_scores'],
                                             results[detector_label]['thresholded_predictions'],
                                             time_steps)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
    except Exception as e:
        st.error(f"Fout bij genereren visualisaties: {e}")

    # Kwantitatieve Evaluatie: Metrics Vergelijking
    st.subheader("Kwantitatieve Evaluatie: Metrics Vergelijking")
    if results:
        metric_names = list(next(iter(results.values()))['metrics'].keys())
        data = {'Metric': metric_names + ['Fit-tijd (s)', 'Predict-tijd (s)']}
        for detector_label in results.keys():
            metric_values = [results[detector_label]['metrics'].get(m, float('nan')) for m in metric_names]
            fit_time = results[detector_label]['fit_time']
            predict_time = results[detector_label]['predict_time']
            data[detector_label] = metric_values + [fit_time, predict_time]
        df = pd.DataFrame(data)
        st.table(df)

        # Toon alleen de metrische waarden, niet de tijden
        metrics_df = df.iloc[:len(metric_names), :].copy()

        # Plotly Bar Chart voor Metrics Vergelijking
        st.subheader("Visuele Vergelijking van Metrics")
        melted_df = metrics_df.melt(id_vars='Metric', var_name='Detector', value_name='Score')
        fig = px.bar(
            melted_df,
            x='Metric',
            y='Score',
            color='Detector',
            barmode='group',
            title='Vergelijking van Detector Prestaties',
            labels={'Metric': 'Evaluatiemetrics', 'Score': 'Score'}
        )
        fig.update_layout(
            xaxis_title='Evaluatiemetrics',
            yaxis_title='Score',
            legend_title='Anomaliedetectoren',
            font=dict(size=12),
            title_font=dict(size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plotly Bar Chart voor Verwerkingstijden
        st.subheader("Vergelijking van Verwerkingstijden")
        time_data = {'Detector': [], 'Type': [], 'Tijd (s)': []}
        for detector_label in results.keys():
            time_data['Detector'].append(detector_label)
            time_data['Type'].append('Trainingstijd')
            time_data['Tijd (s)'].append(results[detector_label]['fit_time'])
            time_data['Detector'].append(detector_label)
            time_data['Type'].append('Voorspellingstijd')
            time_data['Tijd (s)'].append(results[detector_label]['predict_time'])
        time_df = pd.DataFrame(time_data)
        time_fig = px.bar(
            time_df,
            x='Detector',
            y='Tijd (s)',
            color='Type',
            barmode='group',
            title='Vergelijking van Verwerkingstijden',
            labels={'Detector': 'Anomaliedetectoren', 'Tijd (s)': 'Tijd (seconden)', 'Type': 'Tijdstype'}
        )
        time_fig.update_layout(
            xaxis_title='Anomaliedetectoren',
            yaxis_title='Tijd (seconden)',
            legend_title='Tijdstype',
            font=dict(size=12),
            title_font=dict(size=14),
        )
        st.plotly_chart(time_fig, use_container_width=True)

        # Kwantitatieve Analyse: Prestaties en Hulpbronnengebruik
        st.subheader("Kwantitatieve Analyse en Conclusies")
        avg_scores = {label: np.mean(list(results[label]['metrics'].values())) for label in results}
        best_detector = max(avg_scores, key=avg_scores.get)
        st.markdown(f"**Hoogste Prestaties**: {best_detector} met gemiddelde metric-score van {avg_scores[best_detector]:.3f}")
        fit_times = {label: results[label]['fit_time'] for label in results}
        fastest_detector = min(fit_times, key=fit_times.get)
        st.markdown(f"**Minste Fit-tijd**: {fastest_detector} met fit-tijd van {fit_times[fastest_detector]:.2f} seconden")
        predict_times = {label: results[label]['predict_time'] for label in results}
        fastest_predict_detector = min(predict_times, key=predict_times.get)
        st.markdown(f"**Minste Predict-tijd**: {fastest_predict_detector} met predict-tijd van {predict_times[fastest_predict_detector]:.2f} seconden")
        st.markdown("**Impact van Data-eigenschappen**:")
        dim, is_uni = get_dimension(x), is_univariate(x)
        st.write(f"- Dimensionaliteit: {dim}, Univariate: {is_uni}")

def generate_visualization(viz: str, x: np.ndarray, y: np.ndarray, anomaly_scores: np.ndarray,
                           thresholded_predictions: np.ndarray, time_steps: np.ndarray) -> plt.Figure:
    """Generates the requested visualization."""
    if viz == "Anomaliescores":
        return plot_anomaly_scores(x, y, y_pred=anomaly_scores, time_steps=time_steps, figsize=(10, 6))
    elif viz == "Afgebakende Anomalieën":
        return plot_demarcated_anomalies(x, y, time_steps=time_steps, figsize=(10, 6))
    elif viz == "Tijdreeks Gekleurd door Score":
        return plot_time_series_colored_by_score(x, anomaly_scores, time_steps=time_steps, figsize=(10, 6))
    elif viz == "Tijdreeks met Anomalieën":
        return plot_time_series_anomalies(x, y, y_pred=thresholded_predictions, time_steps=time_steps, figsize=(10, 6))
    elif viz == "Zoomweergave":
        if st.session_state.zoom_end <= st.session_state.zoom_start or st.session_state.zoom_end > x.shape[0]:
            st.error("Ongeldig zoombereik.")
            return None
        return plot_with_zoom(x, y, start_zoom=st.session_state.zoom_start, end_zoom=st.session_state.zoom_end,
                              time_steps=time_steps, y_pred=thresholded_predictions,
                              method_to_plot=plot_time_series_anomalies, figsize=(10, 6))
    return None

# --- App Uitvoeren ---
def main():
    """Main function to run the Streamlit app."""
    configure_sidebar()
    if st.button("Pipeline Uitvoeren"):
        run_pipeline()

if __name__ == "__main__":
    main()