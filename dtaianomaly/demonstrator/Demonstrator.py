import streamlit as st
import pandas as pd
import numpy as np
import inspect
import matplotlib.pyplot as plt
import traceback
import time
import logging
import os
import sys
from typing import List, Dict, Any, Union
from scipy import stats
import seaborn as sns
import plotly.express as px
import io  # Added for BytesIO

from dtaianomaly import data, anomaly_detection, preprocessing, evaluation, thresholding
from dtaianomaly.pipeline import EvaluationPipeline
from dtaianomaly.workflow.utils import convert_to_proba_metrics
from dtaianomaly.visualization import (
    plot_anomaly_scores, plot_demarcated_anomalies, plot_time_series_colored_by_score,
    plot_time_series_anomalies, plot_with_zoom, format_time_steps
)
from dtaianomaly.utils import is_valid_array_like, is_univariate, get_dimension
from dtaianomaly.evaluation import AreaUnderROC, Precision

# Configure logging
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Create logger
logger = logging.getLogger('streamlit_app')
if not logger.handlers:  # Only add handlers if they don't exist (to avoid duplicates)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'app.log'))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

logger.info("Starting Streamlit application")

# --- Initialiseren van session state variabelen ---
logger.debug("Initializing session state variables")
if 'experience_level' not in st.session_state:
    st.session_state.experience_level = "Beginner"
if 'uploaded_data_valid' not in st.session_state:
    st.session_state.uploaded_data_valid = False
if 'selected_dataset_name' not in st.session_state:
    st.session_state.selected_dataset_name = None
if 'detector_tabs' not in st.session_state:
    st.session_state.detector_tabs = [{'id': 0, 'mode': 'Beginner', 'detector': None}]
if 'next_tab_id' not in st.session_state:
    st.session_state.next_tab_id = 1
if 'current_detector_tab' not in st.session_state:
    st.session_state.current_detector_tab = 0
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'detector_hyperparams' not in st.session_state:
    st.session_state.detector_hyperparams = {}
if 'export_data' not in st.session_state:
    st.session_state.export_data = {
        'time_steps': None,
        'original_x': None,
        'original_y': None,
        'processed_x': None,
        'processed_y': None,
        'anomaly_scores': {},
        'thresholded_predictions': {},
        'metrics': {},
        'fit_times': {},
        'predict_times': {}
    }
if 'threshold_hyperparams' not in st.session_state:
    st.session_state.threshold_hyperparams = {}

# --- Helper functies ---

def get_available_options(module, base_class, include_functions=False):
    """Haalt dynamisch beschikbare opties op uit een dtaianomaly module."""
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
    """Laadt een dataset uit dtaianomaly.data of dtaianomaly.data.synthetic."""
    if hasattr(data, dataset_name):
        dataset_function = getattr(data, dataset_name)
    else:
        dataset_function = getattr(data.synthetic, dataset_name)
    x, y = dataset_function()
    if not is_valid_array_like(x) or not is_valid_array_like(y) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Ongeldige dataset: moet array-achtig zijn met binaire labels (0 of 1).")
    return x, y


def load_component(module, component_name: str, **kwargs):
    """Laadt dynamisch een component van dtaianomaly op basis van naam en parameters."""
    if component_name is None or component_name == "Geen":
        return None
    try:
        component_class = getattr(module, component_name)
        try:
            return component_class(**kwargs)
        except TypeError as e:
            st.error(f"Fout bij initialiseren van {component_name}: Verkeerde parameters: {e}")
            # Show the expected parameters
            sig = inspect.signature(component_class.__init__)
            param_str = ", ".join([f"{p}" for p in sig.parameters if p != 'self'])
            st.info(f"Verwachte parameters voor {component_name}: {param_str}")
            return None
        except Exception as e:
            st.error(f"Fout bij initialiseren van {component_name}: {e}")
            return None
    except AttributeError:
        st.error(f"Component {component_name} niet gevonden in de module.")
        return None
    except Exception as e:
        st.error(f"Onverwachte fout bij laden van {component_name}: {e}")
        return None


def generate_hyperparam_inputs(detector_name: str, prefix: str) -> Dict[str, Any]:
    """Genereert invoervelden voor detector hyperparameters met een unieke prefix."""
    try:
        detector_class = getattr(anomaly_detection, detector_name)
        signature = inspect.signature(detector_class.__init__)
        hyperparams = {}
        
        for param_name, param_obj in signature.parameters.items():
            if param_name in ['self', 'kwargs', 'args']:
                continue
                
            # Determine default value based on parameter annotation or default
            default_value = param_obj.default
            if default_value == inspect.Parameter.empty:
                # Set appropriate defaults based on type annotation
                param_type = param_obj.annotation
                if param_type == int or param_type == inspect.Parameter.empty:
                    default_value = 10
                elif param_type == float:
                    default_value = 1.0
                elif param_type == bool:
                    default_value = False
                elif param_type == str:
                    default_value = ""
                else:
                    # For unknown types, try to use a sensible default
                    default_value = None
            
            # Special handling for window_size parameter
            if param_name == "window_size":
                window_size_option = st.selectbox(
                    f"{param_name} optie", ["Auto (fft)", "Handmatig"], 
                    key=f"{prefix}_window_size_option", 
                    index=0
                )
                if window_size_option == "Handmatig":
                    hyperparams['window_size'] = int(st.number_input(
                        f"Handmatige {param_name}", 
                        min_value=1, 
                        value=20 if default_value is None else int(default_value),
                        key=f"{prefix}_window_size_manual"
                    ))
                else:
                    hyperparams['window_size'] = 'fft'
            else:
                # Handle different parameter types
                if isinstance(default_value, bool):
                    hyperparams[param_name] = st.checkbox(
                        f"{param_name}", 
                        value=default_value,
                        key=f"{prefix}_{param_name}"
                    )
                elif isinstance(default_value, int):
                    hyperparams[param_name] = st.number_input(
                        f"{param_name}", 
                        value=default_value, 
                        step=1,
                        format="%d", 
                        key=f"{prefix}_{param_name}"
                    )
                elif isinstance(default_value, float):
                    hyperparams[param_name] = st.number_input(
                        f"{param_name}", 
                        value=default_value, 
                        step=0.1,
                        format="%.2f", 
                        key=f"{prefix}_{param_name}"
                    )
                elif isinstance(default_value, str):
                    hyperparams[param_name] = st.text_input(
                        f"{param_name}", 
                        value=default_value,
                        key=f"{prefix}_{param_name}"
                    )
                elif default_value is None:
                    # For parameters with no clear type, provide options
                    param_type_option = st.selectbox(
                        f"{param_name} type", 
                        ["None", "String", "Integer", "Float", "Boolean"],
                        key=f"{prefix}_{param_name}_type"
                    )
                    
                    if param_type_option == "String":
                        hyperparams[param_name] = st.text_input(
                            f"{param_name} (string)", 
                            value="",
                            key=f"{prefix}_{param_name}_value"
                        )
                    elif param_type_option == "Integer":
                        hyperparams[param_name] = st.number_input(
                            f"{param_name} (integer)", 
                            value=0,
                            step=1, 
                            key=f"{prefix}_{param_name}_value"
                        )
                    elif param_type_option == "Float":
                        hyperparams[param_name] = st.number_input(
                            f"{param_name} (float)", 
                            value=0.0,
                            step=0.1, 
                            format="%.2f",
                            key=f"{prefix}_{param_name}_value"
                        )
                    elif param_type_option == "Boolean":
                        hyperparams[param_name] = st.checkbox(
                            f"{param_name} (boolean)", 
                            value=False,
                            key=f"{prefix}_{param_name}_value"
                        )
                    else:  # None
                        hyperparams[param_name] = None
                else:
                    # For complex types or containers (list, dict, etc.), use string input
                    st.warning(f"Parameter {param_name} heeft een complex type. Voer in als string.")
                    hyperparams[param_name] = st.text_input(
                        f"{param_name} (complex type)", 
                        value=str(default_value),
                        key=f"{prefix}_{param_name}"
                    )
        
        return hyperparams
    except Exception as e:
        st.error(f"Fout bij genereren hyperparameters voor {detector_name}: {e}")
        return {}


def get_default_hyperparams(component_class):
    """Haalt de standaard hyperparameters op voor een gegeven klasse."""
    signature = inspect.signature(component_class.__init__)
    hyperparams = {}
    for param_name, param_obj in signature.parameters.items():
        if param_name in ['self', 'kwargs']:
            continue
        if param_obj.default != inspect.Parameter.empty:
            hyperparams[param_name] = param_obj.default
        else:
            if param_obj.annotation == float:
                hyperparams[param_name] = 0.5
            elif param_obj.annotation == int:
                hyperparams[param_name] = 1
            elif param_obj.annotation == bool:
                hyperparams[param_name] = False
            else:
                hyperparams[param_name] = None
    return hyperparams


# --- Functie om geüploade data te valideren ---
def validate_uploaded_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Valideert de geüploade dataset en retourneert x, y als numpy arrays.
    Retourneert ook een foutmelding als de data ongeldig is.
    """
    required_columns = ['Time Step', 'Value', 'Label']
    if not all(col in df.columns for col in required_columns):
        return None, None, "De dataset moet de kolommen 'Time Step', 'Value' en 'Label' bevatten."

    if not pd.api.types.is_numeric_dtype(df['Time Step']):
        return None, None, "De 'Time Step' kolom moet numeriek zijn."

    if not pd.api.types.is_numeric_dtype(df['Value']):
        return None, None, "De 'Value' kolom moet numeriek zijn."

    if not pd.api.types.is_numeric_dtype(df['Label']):
        return None, None, "De 'Label' kolom moet numeriek zijn."

    if not set(df['Label']).issubset({0, 1}):
        return None, None, "De 'Label' kolom moet binaire waarden (0 of 1) bevatten."

    x = df[['Time Step', 'Value']].to_numpy()
    y = df['Label'].to_numpy()

    if not is_valid_array_like(x) or not is_valid_array_like(y):
        return None, None, "De dataset is ongeldig: moet array-achtig zijn."

    return x, y, ""


def configure_sidebar():
    """Configureert de Streamlit sidebar met data, evaluatiemetrics en visualisatie-opties."""
    with st.sidebar:
        st.header("Configuratie")

        # Optie om eigen data te uploaden of ingebouwde dataset te kiezen
        st.subheader("1. Dataset")
        upload_option = st.radio(
            "Kies een dataset bron:",
            ["Gebruik ingebouwde dataset", "Upload eigen dataset"],
            key="upload_option"
        )

        if upload_option == "Upload eigen dataset":
            uploaded_file = st.file_uploader(
                "Sleep je CSV of Excel bestand hierheen of klik om te uploaden",
                type=["csv", "xlsx", "xls"],
                key="file_uploader"
            )
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    x, y, error = validate_uploaded_data(df)
                    if error:
                        st.error(error)
                        st.session_state.uploaded_data_valid = False
                    else:
                        st.session_state.uploaded_data = (x, y)
                        st.session_state.uploaded_data_valid = True
                        st.success("Geüploade dataset is geldig en klaar voor gebruik.")
                except Exception as e:
                    st.error(f"Fout bij het laden van de geüploade dataset: {e}")
                    st.session_state.uploaded_data_valid = False
            else:
                st.session_state.uploaded_data_valid = False
        else:
            dataset_options = get_available_options(data, data.LazyDataLoader, True) + get_available_options(
                data.synthetic, object, True)
            default_dataset_index = dataset_options.index(
                'demonstration_time_series') if 'demonstration_time_series' in dataset_options else 0
            st.session_state.selected_dataset_name = st.selectbox(
                "Selecteer Dataset",
                dataset_options,
                key="dataset_select",
                index=default_dataset_index,
                help="Kies een dataset om te analyseren."
            )
            st.session_state.uploaded_data_valid = False
        
        # Preprocessing instelling (verborgen voor gebruikers, standaard op MinMaxScaler)
        st.session_state.selected_preprocess_name = "MinMaxScaler"
        st.session_state.preprocess_hyperparams = {}
        
        # Evaluatiemetrics Selectie
        st.subheader("2. Evaluatiemetrics")
        metric_options = get_available_options(evaluation, evaluation.Metric)
        st.session_state.selected_metrics = st.multiselect(
            "Selecteer Metrics", 
            metric_options,
            default=["AreaUnderROC", "Precision"],
            key="metrics_select"
        )
        
        # Thresholding Selectie
        st.subheader("3. Thresholding")
        threshold_options = get_available_options(thresholding, thresholding.Thresholding)
        if not threshold_options:
            st.error("Geen thresholding-klassen gevonden. Controleer de installatie.")
            st.stop()
            
        valid_default_thresholds = [opt for opt in ["FixedThreshold"] if opt in threshold_options] or [threshold_options[0]]
        st.session_state.selected_thresholds = st.multiselect(
            "Selecteer Thresholds", 
            threshold_options,
            default=valid_default_thresholds,
            key="thresholds_select"
        )
        
        # Threshold Hyperparameters
        for threshold_name in st.session_state.selected_thresholds:
            threshold_class = getattr(thresholding, threshold_name)
            st.session_state.threshold_hyperparams[threshold_name] = get_default_hyperparams(threshold_class)
            with st.expander(f"{threshold_name} Instellingen"):
                for param_name in st.session_state.threshold_hyperparams[threshold_name]:
                    default_value = st.session_state.threshold_hyperparams[threshold_name][param_name]
                    if isinstance(default_value, float):
                        st.session_state.threshold_hyperparams[threshold_name][param_name] = st.number_input(
                            f"{param_name}", value=default_value, step=0.1, format="%.2f",
                            key=f"threshold_{threshold_name}_{param_name}"
                        )
                    elif isinstance(default_value, int):
                        st.session_state.threshold_hyperparams[threshold_name][param_name] = st.number_input(
                            f"{param_name}", value=default_value, step=1,
                            key=f"threshold_{threshold_name}_{param_name}"
                        )
                    elif isinstance(default_value, bool):
                        st.session_state.threshold_hyperparams[threshold_name][param_name] = st.checkbox(
                            f"{param_name}", value=default_value,
                            key=f"threshold_{threshold_name}_{param_name}"
                        )
                    elif isinstance(default_value, str):
                        st.session_state.threshold_hyperparams[threshold_name][param_name] = st.text_input(
                            f"{param_name}", value=default_value,
                            key=f"threshold_{threshold_name}_{param_name}"
                        )
        
        # Visualisatie-opties
        st.subheader("4. Visualisatie-opties")
        visualization_options = ["Anomaliescores", "Afgebakende Anomalieën", "Tijdreeks Gekleurd door Score",
                                 "Tijdreeks met Anomalieën", "Zoomweergave"]
        st.session_state.selected_visualizations = st.multiselect(
            "Selecteer Visualisaties",
            visualization_options,
            default=["Tijdreeks met Anomalieën"],
            key="visualizations_select"
        )

        # Zoom-instellingen indien gewenst
        if "Zoomweergave" in st.session_state.selected_visualizations:
            st.session_state.zoom_start = st.number_input("Zoom Start Index", min_value=0, value=0,
                                                         key="zoom_start")
            st.session_state.zoom_end = st.number_input("Zoom End Index", min_value=1, value=100,
                                                       key="zoom_end")

def add_detector_tab():
    """Voegt een nieuwe detector tab toe aan de session state."""
    new_tab = {
        'id': st.session_state.next_tab_id,
        'mode': 'Beginner',
        'detector': None
    }
    st.session_state.detector_tabs.append(new_tab)
    st.session_state.next_tab_id += 1
    st.session_state.current_detector_tab = len(st.session_state.detector_tabs) - 1
    
def remove_detector_tab(tab_id):
    """Verwijdert een detector tab uit de session state."""
    for i, tab in enumerate(st.session_state.detector_tabs):
        if tab['id'] == tab_id:
            st.session_state.detector_tabs.pop(i)
            # Als de huidige tab is verwijderd, ga naar de eerste tab
            if st.session_state.current_detector_tab >= len(st.session_state.detector_tabs):
                st.session_state.current_detector_tab = 0
            break

def configure_detector_tab(tab_index):
    """Configureert de detector instellingen voor een specifieke tab."""
    tab = st.session_state.detector_tabs[tab_index]
    
    # Beginner/Expert toggle
    col1, col2 = st.columns([8, 2])
    with col1:
        st.subheader(f"Detector {tab_index + 1}")
    with col2:
        mode = st.selectbox(
            "Modus",
            ["Beginner", "Expert"],
            index=0 if tab['mode'] == 'Beginner' else 1,
            key=f"mode_select_{tab['id']}"
        )
        tab['mode'] = mode
    
    # Detector Selectie
    detector_options = get_available_options(anomaly_detection, anomaly_detection.BaseDetector)
    default_index = detector_options.index('IsolationForest') if 'IsolationForest' in detector_options else 0
    if tab_index == 1 and 'LOF' in detector_options:
        default_index = detector_options.index('LOF')
    
    selected_detector = st.selectbox(
        "Selecteer Detector",
        detector_options,
        index=default_index,
        key=f"detector_select_{tab['id']}"
    )
    tab['detector'] = selected_detector
    
    # Check if the detector requires window_size
    try:
        detector_class = getattr(anomaly_detection, selected_detector)
        init_params = inspect.signature(detector_class.__init__).parameters
        requires_window_size = 'window_size' in init_params
    except (AttributeError, TypeError):
        requires_window_size = False
    
    # Hyperparameters instellen
    detector_key = f"detector_{tab_index + 1}"
    
    if tab['mode'] == 'Beginner':
        # Eenvoudige hyperparameters voor beginners
        st.session_state.detector_hyperparams[detector_key] = {"window_size": "fft"}
        if requires_window_size:
            st.info(f"Detector {selected_detector} gebruikt standaard window_size: 'fft'")
        st.write("Standaard instellingen worden gebruikt voor deze detector.")
    else:
        # Geavanceerde hyperparameters voor experts
        with st.expander("Geavanceerde Instellingen"):
            hyperparams = generate_hyperparam_inputs(selected_detector, prefix=f"expert_{detector_key}")
            
            # Ensure window_size is included if needed
            if requires_window_size and 'window_size' not in hyperparams:
                st.warning(f"Detector {selected_detector} vereist window_size maar deze is niet opgegeven. Gebruikt 'fft' als standaard.")
                hyperparams['window_size'] = 'fft'
                
            st.session_state.detector_hyperparams[detector_key] = hyperparams
    
    return tab['detector']

def run_detector(tab_index):
    """Voert de detectiepipeline uit voor een specifieke detector tab."""
    logger.info(f"Starting detector execution for tab {tab_index}")
    tab = st.session_state.detector_tabs[tab_index]
    detector_key = f"detector_{tab_index + 1}"
    selected_detector = tab['detector']
    
    if not selected_detector:
        logger.warning("No detector selected")
        st.error("Selecteer eerst een detector.")
        return
    
    # Voer pipeline uit voor deze detector
    detector_config = {
        "name": selected_detector,
        "key": detector_key,
        "hyperparams": st.session_state.detector_hyperparams.get(detector_key, {})
    }
    logger.debug(f"Detector configuration: {detector_config}")
    
    # Datasets en preprocessing
    try:
        if st.session_state.uploaded_data_valid:
            logger.debug("Using uploaded data")
            x, y = st.session_state.uploaded_data
            st.write("Geüploade dataset succesvol geladen.")
        else:
            logger.debug(f"Loading dataset: {st.session_state.selected_dataset_name}")
            x, y = load_dataset(st.session_state.selected_dataset_name)
            st.write("Ingebouwde dataset succesvol geladen.")
        
        # Log info about input data
        logger.debug(f"Input data type: {type(x)}, Shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
        st.info(f"Input data type: {type(x)}, Shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
        
        # Set export data
        st.session_state.export_data['time_steps'] = format_time_steps(None, x.shape[0])
        st.session_state.export_data['original_x'] = x
        st.session_state.export_data['original_y'] = y
        
        # Process data using original code's approach
        if st.session_state.selected_preprocess_name != "Geen":
            logger.debug(f"Applying preprocessor: {st.session_state.selected_preprocess_name}")
            preprocessor = load_component(preprocessing, st.session_state.selected_preprocess_name,
                                       **st.session_state.preprocess_hyperparams)
            if preprocessor:
                # Handle both single output and tuple output from preprocessor
                preprocess_result = preprocessor.fit_transform(x)
                if isinstance(preprocess_result, tuple):
                    logger.debug("Preprocessor returned a tuple, extracting first element")
                    st.info("Preprocessor returned a tuple, extracting first element as processed_x")
                    processed_x, processed_y = preprocess_result
                else:
                    logger.debug("Preprocessor returned a single object")
                    processed_x = preprocess_result
                    processed_y = y
                st.session_state.export_data['processed_x'] = processed_x
                st.session_state.export_data['processed_y'] = processed_y
            else:
                logger.debug("No preprocessor initialized, using original data")
                processed_x, processed_y = x, y
        else:
            logger.debug("No preprocessing selected, using original data")
            processed_x, processed_y = x, y
            
        # Convert to numpy array with proper handling for inhomogeneous shapes
        if not isinstance(processed_x, np.ndarray):
            try:
                # Try to directly convert to float64 array
                processed_x = np.array(processed_x, dtype=np.float64)
            except ValueError as e:
                # If we get an inhomogeneous shape error, log information about the data
                st.warning(f"Data heeft ongelijke dimensies: {e}")
                
                # Log info about the structure
                if hasattr(processed_x, '__len__'):
                    st.info(f"Data heeft {len(processed_x)} samples")
                    
                # Try to pad the data to make it homogeneous
                if isinstance(processed_x, list):
                    # Find the maximum length
                    if all(isinstance(item, (list, tuple)) for item in processed_x):
                        max_len = max(len(item) for item in processed_x)
                        st.info(f"Padding data naar uniforme lengte van {max_len}")
                        # Pad with zeros
                        processed_x = [list(item) + [0.0] * (max_len - len(item)) for item in processed_x]
                        processed_x = np.array(processed_x, dtype=np.float64)
                    else:
                        # If it's not a list of lists, try to reshape as needed
                        st.info("Proberen om data om te zetten naar 2D array")
                        processed_x = np.array(processed_x, dtype=np.object)
                        
                        # For univariate data, reshape to a column vector (n_samples, 1)
                        if processed_x.ndim == 1:
                            processed_x = processed_x.reshape(-1, 1)
        
        # If processed_x is still a tuple after conversion attempts, extract the first element
        if isinstance(processed_x, tuple):
            st.info("processed_x is nog steeds een tuple, eerste element gebruiken")
            processed_x = processed_x[0]
            if isinstance(processed_x, tuple):  # Handle nested tuples if necessary
                processed_x = processed_x[0]
        
        # Ensure the data is in the right format (2D array)
        if isinstance(processed_x, np.ndarray):
            if processed_x.ndim == 1:
                # Convert 1D array to 2D column vector
                processed_x = processed_x.reshape(-1, 1)
            elif processed_x.ndim > 2:
                # Flatten higher dimensions to 2D
                n_samples = processed_x.shape[0]
                processed_x = processed_x.reshape(n_samples, -1)
        
        # Convert to float64 array if not already
        if isinstance(processed_x, np.ndarray) and processed_x.dtype != np.float64:
            try:
                processed_x = processed_x.astype(np.float64)
            except (ValueError, TypeError):
                st.warning("Kon data niet converteren naar float64. Probeer object array.")
                processed_x = processed_x.astype(object)
        
        # Handle nans or infinities if present
        if hasattr(processed_x, 'dtype') and np.issubdtype(processed_x.dtype, np.number):
            if np.isnan(processed_x).any() or np.isinf(processed_x).any():
                # Replace NaN with zeros and infinities with large values
                processed_x = np.nan_to_num(processed_x)
        
        # Check if data is empty
        if hasattr(processed_x, 'size') and processed_x.size == 0:
            st.error("De dataset is leeg na preprocessing.")
            return False
        
        # Check for constant values across samples
        if isinstance(processed_x, np.ndarray) and processed_x.size > 0:
            if np.all(processed_x == processed_x[0]):
                st.error("De dataset heeft geen variantie na preprocessing.")
                return False
            
        # Log info about the processed data
        logger.debug(f"Processed data type: {type(processed_x)}, Shape: {processed_x.shape if hasattr(processed_x, 'shape') else 'unknown'}")
        st.info(f"Verwerkte data type: {type(processed_x)}, Shape: {processed_x.shape if hasattr(processed_x, 'shape') else 'unknown'}")
        
        # Log a sample of the processed data
        if isinstance(processed_x, np.ndarray) and processed_x.size > 0:
            logger.debug(f"Processed data sample (first 5 elements): {processed_x.flatten()[:5]}")
            logger.debug(f"Processed data min: {np.min(processed_x)}, max: {np.max(processed_x)}")
        
        # Load detector 
        logger.debug(f"Loading detector: {detector_config['name']}")
        detector = load_component(anomaly_detection, detector_config["name"], 
                                **detector_config["hyperparams"])
        
        # Fit detector and time it
        logger.debug("Fitting detector")
        start_time = time.time()
        detector.fit(processed_x)
        fit_time = time.time() - start_time
        logger.debug(f"Detector fit time: {fit_time:.4f} seconds")
        
        # Predict and time it - using original code approach
        logger.debug("Predicting anomaly scores")
        start_time = time.time()
        if hasattr(detector, 'predict_proba') and callable(getattr(detector, 'predict_proba')):
            logger.debug("Using predict_proba method")
            anomaly_scores = detector.predict_proba(processed_x)
        elif hasattr(detector, 'decision_function') and callable(getattr(detector, 'decision_function')):
            logger.debug("Using decision_function method")
            anomaly_scores = detector.decision_function(processed_x)
        else:
            logger.debug("Using predict method")
            anomaly_scores = detector.predict(processed_x)
        predict_time = time.time() - start_time
        logger.debug(f"Prediction time: {predict_time:.4f} seconds")
        
        # Ensure anomaly_scores is a numpy array with the right shape
        if not isinstance(anomaly_scores, np.ndarray):
            logger.debug(f"Converting anomaly scores from {type(anomaly_scores)} to numpy array")
            anomaly_scores = np.array(anomaly_scores)
        
        # Reshape if needed to match expected shape (n_samples,)
        if anomaly_scores.ndim > 1:
            logger.debug(f"Reshaping anomaly scores from shape {anomaly_scores.shape} to 1D array")
            anomaly_scores = anomaly_scores.reshape(-1)
            
        # Log anomaly scores info
        logger.debug(f"Anomaly scores type: {type(anomaly_scores)}, shape: {anomaly_scores.shape}")
        logger.debug(f"Anomaly scores sample (first 5): {anomaly_scores[:5]}")
        logger.debug(f"Anomaly scores min: {np.min(anomaly_scores)}, max: {np.max(anomaly_scores)}")
            
        # Normalize anomaly scores to 0-1 range if needed
        if not np.all((anomaly_scores >= 0) & (anomaly_scores <= 1)):
            logger.debug("Normalizing anomaly scores to 0-1 range")
            min_score = np.min(anomaly_scores)
            max_score = np.max(anomaly_scores)
            if max_score > min_score:  # Avoid division by zero
                anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score)
                logger.debug(f"Normalized anomaly scores - min: {np.min(anomaly_scores)}, max: {np.max(anomaly_scores)}")
        
        # Load thresholds and apply - improved approach for thresholding
        thresholded_predictions = {}
        all_metrics = {}
        
        # Clear existing results to avoid confusion
        if detector_key in st.session_state.results:
            logger.debug(f"Clearing previous results for {detector_key}")
            del st.session_state.results[detector_key]
        
        for threshold_name in st.session_state.selected_thresholds:
            logger.info(f"Processing threshold: {threshold_name}")
            try:
                # Special handling for ContaminationRate to avoid format string error
                if threshold_name == 'ContaminationRate':
                    logger.info("Using special handling for ContaminationRate")
                    
                    # Ensure processed_y is not None - use original y if it is
                    if processed_y is None:
                        logger.warning("processed_y is None, using original y for metrics calculation")
                        processed_y = y
                    
                    # Use direct computation to avoid the format string error
                    threshold_params = st.session_state.threshold_hyperparams.get(threshold_name, {})
                    contamination_rate = threshold_params.get('contamination_rate', 0.5)  # Default to 0.5 if not specified
                    logger.debug(f"Using contamination rate: {contamination_rate}")
                    
                    # Sort anomaly scores and find the threshold value
                    sorted_scores = np.sort(anomaly_scores)
                    threshold_index = int(len(sorted_scores) * (1 - contamination_rate))
                    threshold_value = sorted_scores[threshold_index] if threshold_index < len(sorted_scores) else sorted_scores[-1]
                    logger.debug(f"Calculated threshold value for ContaminationRate: {threshold_value}")
                    
                    # Apply threshold
                    y_pred = (anomaly_scores >= threshold_value).astype(int)
                    logger.debug(f"Created binary predictions using contamination rate threshold")
                    st.info(f"Using threshold value {threshold_value:.4f} calculated from contamination rate {contamination_rate}")
                    
                    # Store predictions
                    thresholded_predictions[threshold_name] = y_pred
                    logger.debug(f"Stored thresholded predictions for {threshold_name}")
                    
                    # Log predictions summary
                    if isinstance(y_pred, np.ndarray):
                        anomaly_count = np.sum(y_pred)
                        logger.debug(f"Predictions summary - anomalies: {anomaly_count}, normal: {len(y_pred) - anomaly_count}")
                        logger.debug(f"Anomaly percentage: {anomaly_count/len(y_pred)*100:.2f}%")
                    
                    # Calculate metrics for ContaminationRate using scikit-learn directly
                    # This is more reliable than the custom metrics
                    metrics_dict = {}
                    
                    try:
                        # Import metrics from scikit-learn
                        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
                        
                        # Make sure processed_y is valid array-like
                        if not isinstance(processed_y, np.ndarray):
                            processed_y = np.array(processed_y)
                        
                        # Calculate all possible metrics to ensure they're all captured
                        # Add these to metrics_dict regardless of what's selected, 
                        # filtering can happen at display time
                        try:
                            score = roc_auc_score(processed_y, anomaly_scores)
                            metrics_dict['AreaUnderROC'] = score
                            logger.debug(f"AreaUnderROC score: {score}")
                        except Exception as e:
                            logger.error(f"Error calculating AreaUnderROC: {e}")
                        
                        try:
                            score = precision_score(processed_y, y_pred)
                            metrics_dict['Precision'] = score
                            logger.debug(f"Precision score: {score}")
                        except Exception as e:
                            logger.error(f"Error calculating Precision: {e}")
                            
                        try:
                            score = recall_score(processed_y, y_pred)
                            metrics_dict['Recall'] = score
                            logger.debug(f"Recall score: {score}")
                        except Exception as e:
                            logger.error(f"Error calculating Recall: {e}")
                            
                        try:
                            score = f1_score(processed_y, y_pred)
                            metrics_dict['F1Score'] = score
                            logger.debug(f"F1Score score: {score}")
                        except Exception as e:
                            logger.error(f"Error calculating F1Score: {e}")
                            
                        try:
                            score = accuracy_score(processed_y, y_pred)
                            metrics_dict['Accuracy'] = score
                            logger.debug(f"Accuracy score: {score}")
                        except Exception as e:
                            logger.error(f"Error calculating Accuracy: {e}")
                        
                        # Add the metrics to all_metrics
                        all_metrics[threshold_name] = metrics_dict
                        logger.debug(f"Stored metrics for {threshold_name}: {metrics_dict}")
                        
                    except Exception as metrics_err:
                        logger.error(f"Error calculating metrics for ContaminationRate: {metrics_err}")
                        logger.exception(f"ContaminationRate metrics error traceback:")
                        all_metrics[threshold_name] = {}
                    
                    continue  # Skip the rest of the processing for ContaminationRate
                
                # Standard processing for other threshold classes
                logger.debug(f"Getting threshold class: {threshold_name}")
                threshold_class = getattr(thresholding, threshold_name)
                threshold_params = st.session_state.threshold_hyperparams.get(threshold_name, {})
                logger.debug(f"Threshold parameters: {threshold_params}")
                
                # Inspect the threshold class
                logger.debug(f"Threshold class methods: {[m for m in dir(threshold_class) if not m.startswith('_')]}")
                
                # Create the threshold object
                logger.debug(f"Creating threshold object with params: {threshold_params}")
                threshold_obj = threshold_class(**threshold_params)
                
                # Inspect the threshold instance
                logger.debug(f"Threshold instance methods: {[m for m in dir(threshold_obj) if not m.startswith('_')]}")
                
                # Get thresholded predictions
                if hasattr(threshold_obj, 'transform') and callable(getattr(threshold_obj, 'transform')):
                    # Use transform method if available
                    logger.debug("Using transform method")
                    
                    # Debug the transform method signature
                    transform_method = getattr(threshold_obj, 'transform')
                    transform_sig = inspect.signature(transform_method)
                    logger.debug(f"Transform method signature: {transform_sig}")
                    
                    # Log the anomaly scores before transform
                    logger.debug(f"Anomaly scores before transform - type: {type(anomaly_scores)}, shape: {anomaly_scores.shape}")
                    
                    # Call transform with detailed error handling
                    try:
                        y_pred = threshold_obj.transform(anomaly_scores)
                        logger.debug(f"Transform successful - y_pred type: {type(y_pred)}, shape: {y_pred.shape if hasattr(y_pred, 'shape') else 'unknown'}")
                    except Exception as transform_error:
                        logger.error(f"Error in transform method: {transform_error}")
                        logger.exception("Transform method traceback:")
                        raise
                elif hasattr(threshold_obj, 'threshold') and callable(getattr(threshold_obj, 'threshold')):
                    # Use threshold method if available
                    logger.debug("Using threshold method")
                    
                    # Debug the threshold method signature
                    threshold_method = getattr(threshold_obj, 'threshold')
                    threshold_sig = inspect.signature(threshold_method)
                    logger.debug(f"Threshold method signature: {threshold_sig}")
                    
                    # Call threshold method
                    try:
                        threshold_value = threshold_obj.threshold(anomaly_scores)
                        logger.debug(f"Calculated threshold value: {threshold_value}")
                        y_pred = (anomaly_scores >= threshold_value).astype(int)
                        logger.debug(f"Created binary predictions using threshold {threshold_value}")
                        st.info(f"Using threshold() method from {threshold_name}, calculated threshold: {threshold_value:.4f}")
                    except Exception as threshold_error:
                        logger.error(f"Error in threshold method: {threshold_error}")
                        logger.exception("Threshold method traceback:")
                        raise
                else:
                    # Fixed threshold fallback
                    fixed_threshold_used = True
                    threshold_value = 0.5
                    logger.warning(f"{threshold_name} has no transform or threshold method. Using fixed threshold at {threshold_value}")
                    st.warning(f"{threshold_name} has no transform or threshold method. Using a fixed threshold at {threshold_value}.")
                    y_pred = (anomaly_scores >= threshold_value).astype(int)
                    logger.debug(f"Created binary predictions using fixed threshold {threshold_value}")
                
                # Store thresholded predictions
                thresholded_predictions[threshold_name] = y_pred
                logger.debug(f"Stored thresholded predictions for {threshold_name}")
                
                # Log predictions summary
                if isinstance(y_pred, np.ndarray):
                    logger.debug(f"Predictions summary - positive: {np.sum(y_pred)}, negative: {len(y_pred) - np.sum(y_pred)}")
                
                # Calculate metrics with safer approach - CALCULATE ALL METRICS 
                # regardless of what's selected to ensure a complete set
                metrics_dict = {}
                try:
                    # Import metrics from scikit-learn for direct calculation
                    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
                    
                    # Make sure processed_y is valid array-like
                    if not isinstance(processed_y, np.ndarray):
                        processed_y = np.array(processed_y)
                    
                    # Calculate all possible metrics
                    try:
                        score = roc_auc_score(processed_y, anomaly_scores)
                        metrics_dict['AreaUnderROC'] = score
                        logger.debug(f"AreaUnderROC score: {score}")
                    except Exception as e:
                        logger.error(f"Error calculating AreaUnderROC: {e}")
                    
                    try:
                        score = precision_score(processed_y, y_pred)
                        metrics_dict['Precision'] = score
                        logger.debug(f"Precision score: {score}")
                    except Exception as e:
                        logger.error(f"Error calculating Precision: {e}")
                        
                    try:
                        score = recall_score(processed_y, y_pred)
                        metrics_dict['Recall'] = score
                        logger.debug(f"Recall score: {score}")
                    except Exception as e:
                        logger.error(f"Error calculating Recall: {e}")
                        
                    try:
                        score = f1_score(processed_y, y_pred)
                        metrics_dict['F1Score'] = score
                        logger.debug(f"F1Score score: {score}")
                    except Exception as e:
                        logger.error(f"Error calculating F1Score: {e}")
                        
                    try:
                        score = accuracy_score(processed_y, y_pred)
                        metrics_dict['Accuracy'] = score
                        logger.debug(f"Accuracy score: {score}")
                    except Exception as e:
                        logger.error(f"Error calculating Accuracy: {e}")
                    
                    # Now also try the module's metrics if available
                    for metric_name in st.session_state.selected_metrics:
                        logger.debug(f"Calculating metric: {metric_name}")
                        try:
                            metric_class = getattr(evaluation, metric_name)
                            metric_instance = metric_class()
                            
                            # Log metric instance info
                            logger.debug(f"Metric class: {metric_class.__name__}")
                            logger.debug(f"Is ProbaMetric: {isinstance(metric_instance, evaluation.ProbaMetric)}")
                            
                            # Try to calculate metrics in a safer way
                            if isinstance(metric_instance, evaluation.ProbaMetric):
                                # For probability-based metrics
                                logger.debug("Using probability-based metric computation")
                                score = metric_instance.compute(processed_y, anomaly_scores)
                            else:
                                # For binary metrics
                                logger.debug("Using binary metric computation")
                                logger.debug(f"Ground truth shape: {processed_y.shape}, predictions shape: {y_pred.shape}")
                                score = metric_instance.compute(processed_y, y_pred)
                                
                            # Only add if not already calculated or if it's different
                            if metric_name not in metrics_dict:
                                logger.debug(f"Metric {metric_name} score: {score}")
                                metrics_dict[metric_name] = score
                        except Exception as metric_error:
                            logger.error(f"Error calculating {metric_name}: {metric_error}")
                            logger.exception("Metric calculation traceback:")
                            # Don't override if already calculated via scikit-learn
                            if metric_name not in metrics_dict:
                                metrics_dict[metric_name] = None
                except Exception as metrics_error:
                    logger.error(f"General metrics error: {metrics_error}")
                    logger.exception("General metrics traceback:")
                    st.error(f"General metrics error: {metrics_error}")
                    
                all_metrics[threshold_name] = metrics_dict
                logger.debug(f"Stored metrics for {threshold_name}: {metrics_dict}")
                
            except Exception as e:
                logger.error(f"Error applying {threshold_name}: {e}")
                logger.exception(f"Thresholding error traceback:")
                st.error(f"Fout bij toepassen van {threshold_name}: {e}")
                continue
                
        # Store results
        st.session_state.results[detector_key] = {
            'detector_name': detector_config['name'],
            'anomaly_scores': anomaly_scores,
            'thresholded_predictions': thresholded_predictions,
            'metrics': all_metrics,
            'fit_time': fit_time,
            'predict_time': predict_time
        }
        logger.info(f"Stored results for detector {detector_key}")
        
        # Update export data
        st.session_state.export_data['anomaly_scores'][detector_key] = anomaly_scores
        st.session_state.export_data['thresholded_predictions'][detector_key] = thresholded_predictions
        st.session_state.export_data['metrics'][detector_key] = all_metrics
        st.session_state.export_data['fit_times'][detector_key] = fit_time
        st.session_state.export_data['predict_times'][detector_key] = predict_time
        
        logger.info(f"Detector {detector_key} execution completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error executing detector: {e}")
        logger.exception("Detector execution traceback:")
        st.error(f"Fout tijdens uitvoeren van detector: {e}")
        st.error(traceback.format_exc())
        return False

def display_detector_results(tab_index):
    """Toont de resultaten van de detectie voor een specifieke tab."""
    tab = st.session_state.detector_tabs[tab_index]
    detector_key = f"detector_{tab_index + 1}"
    
    if detector_key not in st.session_state.results:
        st.warning("Deze detector is nog niet uitgevoerd.")
        return
    
    results = st.session_state.results[detector_key]
    detector_name = results['detector_name']
    
    try:
        if st.session_state.uploaded_data_valid:
            x, y = st.session_state.uploaded_data
        else:
            if 'selected_dataset_name' not in st.session_state or not st.session_state.selected_dataset_name:
                st.error("Geen dataset geselecteerd.")
                return
            x, y = load_dataset(st.session_state.selected_dataset_name)
        
        time_steps = format_time_steps(None, x.shape[0])
        anomaly_scores = results['anomaly_scores']
        
        # Individual Detector Information
        st.subheader(f"Detector Informatie: {detector_name}")
        st.write(f"Fit-tijd: {results['fit_time']:.4f} seconden")
        st.write(f"Predict-tijd: {results['predict_time']:.4f} seconden")
        
        # Create a metrics section for this detector
        st.subheader("Prestatiemetrics")
        
        # Create metrics dataframe for display for just this detector
        metric_rows = []
        
        # Iterate over thresholds for this detector
        if 'thresholded_predictions' in results and results['thresholded_predictions']:
            thresholds = list(results['thresholded_predictions'].keys())
            
            for threshold_name in thresholds:
                y_pred = results['thresholded_predictions'][threshold_name]
                
                # If we have calculated metrics in the results, use them
                if 'metrics' in results and threshold_name in results['metrics']:
                    metric_dict = results['metrics'][threshold_name]
                    for metric_name, value in metric_dict.items():
                        if value is not None:
                            metric_rows.append({
                            'Threshold': threshold_name,
                            'Metric': metric_name,
                            'Waarde': value
                        })
                
                # If no metrics are in the results for the current threshold, calculate on the fly
                if not metric_rows or all(row['Threshold'] != threshold_name for row in metric_rows):
                    try:
                        # Use scikit-learn to calculate common metrics
                        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
                        
                        # Calculate metrics that were selected by the user
                        for metric_name in st.session_state.selected_metrics:
                            try:
                                if metric_name == 'AreaUnderROC':
                                    score = roc_auc_score(y, anomaly_scores)
                                elif metric_name == 'Precision':
                                    score = precision_score(y, y_pred)
                                elif metric_name == 'Recall':
                                    score = recall_score(y, y_pred)
                                elif metric_name == 'F1Score':
                                    score = f1_score(y, y_pred)
                                elif metric_name == 'Accuracy':
                                    score = accuracy_score(y, y_pred)
                                else:
                                    # Skip metrics we don't know how to calculate
                                    continue
                                
                                metric_rows.append({
                                    'Threshold': threshold_name,
                                    'Metric': metric_name,
                                    'Waarde': score
                                })
                            except Exception as e:
                                logger.error(f"Error calculating metric {metric_name}: {e}")
                    except Exception as e:
                        logger.error(f"Error calculating metrics: {e}")
        
        # Display metrics dataframe
        if metric_rows:
            metrics_df = pd.DataFrame(metric_rows)
            st.dataframe(metrics_df)
            
            # Show metrics using Streamlit's metric component for better visualization
            st.subheader("Metric Overzicht")
            
            # Create rows of metrics for visual display
            metrics_list = list(set([row['Metric'] for row in metric_rows]))
            thresholds_list = list(set([row['Threshold'] for row in metric_rows]))
            
            # Display metrics in columns
            for threshold_name in thresholds_list:
                st.write(f"**Threshold: {threshold_name}**")
                cols = st.columns(min(len(metrics_list), 3))  # Up to 3 metrics per row
                
                for i, metric_name in enumerate(metrics_list):
                    # Find the matching row
                    matching_rows = [row for row in metric_rows 
                                    if row['Metric'] == metric_name and row['Threshold'] == threshold_name]
                    
                    if matching_rows:
                        score = matching_rows[0]['Waarde']
                        # Format score as percentage if it's a float
                        display_value = f"{score:.2%}" if isinstance(score, float) else score
                        cols[i % len(cols)].metric(
                            label=metric_name,
                            value=display_value
                        )
        else:
            st.warning("Geen metrics berekend voor deze detector.")
        
        # Individual Visualizations section
        st.subheader("Visualisaties")
        if 'selected_visualizations' not in st.session_state or not st.session_state.selected_visualizations:
            st.info("Geen visualisaties geselecteerd.")
        else:
            # Use tabs for different visualization types
            viz_tabs = st.tabs(st.session_state.selected_visualizations)
            
            for i, viz in enumerate(st.session_state.selected_visualizations):
                with viz_tabs[i]:
                    # For visualizations that don't need thresholded predictions
                    if viz in ["Anomaliescores", "Tijdreeks Gekleurd door Score", "Afgebakende Anomalieën"]:
                        try:
                            fig = generate_visualization(viz, x, y, anomaly_scores)
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.warning(f"Kon visualisatie {viz} niet genereren.")
                        except Exception as viz_error:
                            st.error(f"Fout bij genereren van {viz}: {viz_error}")
                            st.error(traceback.format_exc())
                    else:
                        # For visualizations that need thresholded predictions
                        y_pred = None
                        
                        # Try to get thresholded predictions if available
                        if results['thresholded_predictions']:
                            # Get the first available threshold's predictions
                            threshold_name = list(results['thresholded_predictions'].keys())[0]
                            y_pred = results['thresholded_predictions'][threshold_name]
                        
                        try:
                            fig = generate_visualization(viz, x, y, anomaly_scores, y_pred, time_steps)
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.warning(f"Kon visualisatie {viz} niet genereren.")
                        except Exception as viz_error:
                            st.error(f"Fout bij genereren van {viz}: {viz_error}")
                            st.error(traceback.format_exc())
        
        # Export opties
        st.subheader("Exporteren")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Exporteer naar Excel", key=f"export_excel_{tab_index}"):
                try:
                    excel_file = export_to_excel()
                    st.download_button(
                        label="Download Excel bestand",
                        data=excel_file,
                        file_name=f"anomaly_detection_results_{detector_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_excel_{tab_index}"
                    )
                except Exception as export_error:
                    st.error(f"Fout bij exporteren naar Excel: {export_error}")
                    st.error(traceback.format_exc())
        
        with col2:
            if st.button("Exporteer naar CSV", key=f"export_csv_{tab_index}"):
                try:
                    csv_buffer = io.StringIO()
                    # Export anomaly scores
                    anomaly_scores_df = pd.DataFrame({
                        'Time Step': time_steps,
                        'Anomaly Score': anomaly_scores
                    })
                    anomaly_scores_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Download CSV bestand",
                        data=csv_data,
                        file_name=f"anomaly_detection_results_{detector_name}.csv",
                        mime="text/csv",
                        key=f"download_csv_{tab_index}"
                    )
                except Exception as export_error:
                    st.error(f"Fout bij exporteren naar CSV: {export_error}")
                    st.error(traceback.format_exc())
        
    except Exception as e:
        st.error(f"Fout bij weergeven resultaten: {e}")
        st.error(traceback.format_exc())

def run_pipeline():
    """Voert de anomaliedetectiepipeline uit voor alle actieve detectors."""
    success = False
    for i, _ in enumerate(st.session_state.detector_tabs):
        if run_detector(i):
            success = True
    
    return success

def generate_visualization(viz: str, x: np.ndarray, y: np.ndarray, anomaly_scores: np.ndarray,
                           thresholded_predictions: np.ndarray = None, time_steps: np.ndarray = None) -> plt.Figure:
    """Genereert de gevraagde visualisatie."""
    # Check inputs and provide defaults if needed
    if time_steps is None:
        time_steps = format_time_steps(None, x.shape[0])
        
    # Some visualizations don't need thresholded predictions
    if viz == "Anomaliescores":
        return plot_anomaly_scores(x, y, y_pred=anomaly_scores, time_steps=time_steps, figsize=(10, 6))
    elif viz == "Tijdreeks Gekleurd door Score":
        return plot_time_series_colored_by_score(x, anomaly_scores, time_steps=time_steps, figsize=(10, 6))
    elif viz == "Afgebakende Anomalieën":
        return plot_demarcated_anomalies(x, y, time_steps=time_steps, figsize=(10, 6))
        
    # Visualizations that require thresholded predictions
    if thresholded_predictions is None:
        # If no thresholded predictions available, create a simple one with a 0.5 threshold
        thresholded_predictions = (anomaly_scores >= 0.5).astype(int)
    
    if viz == "Tijdreeks met Anomalieën":
        return plot_time_series_anomalies(x, y, y_pred=thresholded_predictions, time_steps=time_steps, figsize=(10, 6))
    elif viz == "Zoomweergave":
        # Default zoom range if not specified
        zoom_start = 0
        zoom_end = min(100, x.shape[0])
        
        if hasattr(st.session_state, 'zoom_start') and hasattr(st.session_state, 'zoom_end'):
            zoom_start = st.session_state.zoom_start
            zoom_end = st.session_state.zoom_end
            
        if zoom_end <= zoom_start or zoom_end > x.shape[0]:
            st.error("Ongeldig zoombereik.")
            return None
            
        return plot_with_zoom(x, y, start_zoom=zoom_start, end_zoom=zoom_end,
                          time_steps=time_steps, y_pred=thresholded_predictions,
                          method_to_plot=plot_time_series_anomalies, figsize=(10, 6))
    
    return None


def export_to_excel():
    """Genereert een Excel bestand met de verzamelde data."""
    export_data = st.session_state.export_data
    time_steps = export_data['time_steps']
    original_x = export_data['original_x']
    original_y = export_data['original_y']
    processed_x = export_data['processed_x']
    processed_y = export_data['processed_y']
    anomaly_scores = export_data['anomaly_scores']
    thresholded_predictions = export_data['thresholded_predictions']
    metrics = export_data['metrics']
    fit_times = export_data['fit_times']
    predict_times = export_data['predict_times']

    # Create DataFrame for original dataset
    if is_univariate(original_x):
        original_df = pd.DataFrame({
            'Time Step': time_steps,
            'Value': original_x.flatten(),
            'Label': original_y
        })
    else:
        columns = ['Time Step'] + [f'Dimension {i + 1}' for i in range(original_x.shape[1])] + ['Label']
        original_df = pd.DataFrame(np.column_stack([time_steps, original_x, original_y]), columns=columns)

    # Create DataFrame for processed dataset if applicable
    if processed_x is not None:
        if is_univariate(processed_x):
            processed_df = pd.DataFrame({
                'Time Step': time_steps,
                'Value': processed_x.flatten(),
                'Label': processed_y
            })
        else:
            columns = ['Time Step'] + [f'Dimension {i + 1}' for i in range(processed_x.shape[1])] + ['Label']
            processed_df = pd.DataFrame(np.column_stack([time_steps, processed_x, processed_y]), columns=columns)
    else:
        processed_df = None

    # Anomaly scores DataFrame
    anomaly_scores_df = pd.DataFrame({'Time Step': time_steps})
    for detector_label, scores in anomaly_scores.items():
        anomaly_scores_df[detector_label] = scores

    # Thresholded predictions DataFrame
    thresholded_predictions_df = pd.DataFrame({'Time Step': time_steps})
    for detector_label, preds in thresholded_predictions.items():
        thresholded_predictions_df[detector_label] = preds

    # Metrics DataFrame
    metric_names = list(next(iter(metrics.values())).keys()) if metrics else []
    metrics_df = pd.DataFrame(index=metrics.keys(), columns=metric_names + ['Fit Time (s)', 'Predict Time (s)'])
    for detector_label in metrics:
        for metric_name in metric_names:
            metrics_df.at[detector_label, metric_name] = metrics[detector_label].get(metric_name, float('nan'))
        metrics_df.at[detector_label, 'Fit Time (s)'] = fit_times[detector_label]
        metrics_df.at[detector_label, 'Predict Time (s)'] = predict_times[detector_label]

    # Create Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        original_df.to_excel(writer, sheet_name='Original Dataset', index=False)
        if processed_df is not None:
            processed_df.to_excel(writer, sheet_name='Processed Dataset', index=False)
        anomaly_scores_df.to_excel(writer, sheet_name='Anomaly Scores', index=False)
        thresholded_predictions_df.to_excel(writer, sheet_name='Thresholded Predictions', index=False)
        metrics_df.to_excel(writer, sheet_name='Evaluation Metrics', index=True)
    output.seek(0)
    return output


# --- Hoofdapp uitvoering ---
def main():
    """Hoofdfunctie om de Streamlit app te draaien."""
    # App titel en beschrijving
    st.title("dtaianomaly Demonstrator")
    st.markdown(
        """
        Een no-code demonstrator voor de **dtaianomaly** bibliotheek, waarmee je interactief anomaliedetectietechnieken 
        voor tijdreeksdata kunt verkennen. Deze tool ondersteunt zowel kwalitatieve evaluatie (visualisaties) als 
        kwantitatieve evaluatie (benchmarking en prestatieconclusies) met vergelijking van meerdere detectoren.
        """
    )
    
    # Verborgen ervaring niveau selectie - alleen de sessie state variabele instellen
    # We verbergen de UI-element maar behouden de functionaliteit
    if 'experience_level' not in st.session_state:
        st.session_state.experience_level = "Beginner"
    
    # Main layout with sidebar and main content
    sidebar_col, main_col = st.columns([1, 3])
    
    # Left sidebar (fixed configuration)
    with sidebar_col:
        # Configure the sidebar with datasets, metrics, thresholds, and visualizations
        configure_sidebar()
    
    # Main content area 
    with main_col:
        # Create detector tabs section
        st.subheader("Anomaliedetectoren")
        
        # Add/Remove detector tabs
        tab_col1, tab_col2 = st.columns([9, 1])
        with tab_col2:
            if st.button("➕", help="Voeg detector toe"):
                add_detector_tab()
                st.rerun()
        
        # Only proceed if there are detector tabs
        if not st.session_state.detector_tabs:
            st.warning("Geen detectoren beschikbaar. Voeg een detector toe met de + knop.")
            return
            
        # Create tabs for detectors
        tab_names = [f"Detector {i+1}" for i in range(len(st.session_state.detector_tabs))]
        tabs = st.tabs(tab_names)
        
        # Initialize detector_results to store which detectors have been run
        if 'detector_results' not in st.session_state:
            st.session_state.detector_results = set()
        
        # Process each detector tab
        for i, tab in enumerate(tabs):
            with tab:
                # Show remove button except for first tab
                if i > 0:
                    if st.button("❌", key=f"remove_tab_{i}", help="Verwijder deze detector"):
                        remove_detector_tab(st.session_state.detector_tabs[i]['id'])
                        if f"detector_{i+1}" in st.session_state.detector_results:
                            st.session_state.detector_results.remove(f"detector_{i+1}")
                        st.rerun()
                
                # Configure detector - each tab has its own independent detector
                detector_config_container = st.container()
                with detector_config_container:
                    selected_detector = configure_detector_tab(i)
                
                # Run button for individual detector
                detector_run_container = st.container()
                with detector_run_container:
                    if st.button("Uitvoeren", key=f"run_detector_{i}"):
                        if st.session_state.uploaded_data_valid or 'selected_dataset_name' in st.session_state:
                            with st.spinner(f"Detector {i+1} uitvoeren..."):
                                if run_detector(i):
                                    st.success(f"Detector {i+1} succesvol uitgevoerd!")
                                    st.session_state.detector_results.add(f"detector_{i+1}")
                                    st.rerun()
                        else:
                            st.error("Selecteer een geldige dataset of upload een correct bestand voordat je de detector uitvoert.")
                
                # Display detector results if available
                detector_result_container = st.container()
                with detector_result_container:
                    detector_key = f"detector_{i+1}"
                    if detector_key in st.session_state.results:
                        display_detector_results(i)
                    else:
                        st.info(f"Detector {i+1} is nog niet uitgevoerd. Klik op 'Uitvoeren' om de detector te starten.")
        
        # Global run button
        global_run_container = st.container()
        with global_run_container:
            if st.button("Alle Detectoren Uitvoeren", key="run_all_detectors"):
                if st.session_state.uploaded_data_valid or 'selected_dataset_name' in st.session_state:
                    with st.spinner("Alle detectoren uitvoeren..."):
                        success = run_pipeline()
                        if success:
                            # Add all detectors to the results set
                            for i in range(len(st.session_state.detector_tabs)):
                                st.session_state.detector_results.add(f"detector_{i+1}")
                            st.success("Alle detectoren succesvol uitgevoerd!")
                            st.rerun()
                else:
                    st.error("Selecteer een geldige dataset of upload een correct bestand voordat je de pipeline uitvoert.")
        
        # GLOBAL COMPARISON SECTION - Outside of the tabs, always visible when there are multiple detectors
        if len([k for k in st.session_state.results.keys() if k.startswith('detector_')]) >= 2:
            st.markdown("---")  # Divider
            comparison_container = st.container()
            with comparison_container:
                st.header("Vergelijkend Overzicht Detectoren")
                
                # Create comparison tabs
                compare_tabs = st.tabs(["Metrics Vergelijking", "Tijdsvergelijking", "Prestatie-analyse"])
                
                # Metrics Vergelijking tab
                with compare_tabs[0]:
                    st.write("**Kwantitatieve Evaluatie: Metrics Vergelijking**")
                    
                    # Collect metrics data from all detectors
                    metric_names = set()
                    detector_metrics = {}
                    
                    for detector_key, detector_result in st.session_state.results.items():
                        if not detector_key.startswith('detector_'):
                            continue
                            
                        detector_label = f"Detector {detector_key[-1]} ({detector_result['detector_name']})"
                        detector_metrics[detector_label] = {
                            'fit_time': detector_result['fit_time'],
                            'predict_time': detector_result['predict_time']
                        }
                        
                        # Collect metrics from each threshold
                        if 'metrics' in detector_result and detector_result['metrics']:
                            for threshold_name, metrics in detector_result['metrics'].items():
                                for metric_name, value in metrics.items():
                                    if value is not None:
                                        metric_names.add(metric_name)
                                        detector_metrics[detector_label][metric_name] = value
                    
                    # Create DataFrame for display
                    if metric_names:
                        metric_list = list(metric_names) + ['fit_time', 'predict_time']
                        data = {'Metric': metric_list}
                        
                        for detector_label, metrics in detector_metrics.items():
                            data[detector_label] = [
                                metrics.get(m, float('nan')) for m in metric_list
                            ]
                        
                        df = pd.DataFrame(data)
                        st.dataframe(df)
                        
                        # Create bar chart for metrics comparison
                        st.write("**Visuele Vergelijking van Metrics**")
                        metrics_df = df[df['Metric'].isin(list(metric_names))]
                        
                        try:
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
                        except Exception as chart_error:
                            st.error(f"Fout bij genereren van vergelijkingsgrafiek: {chart_error}")
                    else:
                        st.warning("Geen metrics beschikbaar voor vergelijking.")
                
                # Tijdsvergelijking tab
                with compare_tabs[1]:
                    st.write("**Vergelijking van Verwerkingstijden**")
                    
                    try:
                        # Create comparison bar chart for timing
                        time_data = {'Detector': [], 'Type': [], 'Tijd (s)': []}
                        
                        for detector_key, detector_result in st.session_state.results.items():
                            if not detector_key.startswith('detector_'):
                                continue
                                
                            detector_label = f"Detector {detector_key[-1]} ({detector_result['detector_name']})"
                            
                            time_data['Detector'].append(detector_label)
                            time_data['Type'].append('Trainingstijd')
                            time_data['Tijd (s)'].append(detector_result['fit_time'])
                            
                            time_data['Detector'].append(detector_label)
                            time_data['Type'].append('Voorspellingstijd')
                            time_data['Tijd (s)'].append(detector_result['predict_time'])
                        
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
                    except Exception as time_error:
                        st.error(f"Fout bij genereren van tijdsvergelijking: {time_error}")
                
                # Prestatie-analyse tab
                with compare_tabs[2]:
                    st.write("**Kwantitatieve Analyse en Conclusies**")
                    
                    try:
                        # Calculate avg scores and find best detector
                        avg_scores = {}
                        for detector_key, detector_result in st.session_state.results.items():
                            if not detector_key.startswith('detector_'):
                                continue
                                
                            detector_label = f"Detector {detector_key[-1]} ({detector_result['detector_name']})"
                            
                            # Collect all metric values
                            all_values = []
                            if 'metrics' in detector_result and detector_result['metrics']:
                                for threshold_metrics in detector_result['metrics'].values():
                                    all_values.extend([v for v in threshold_metrics.values() if v is not None])
                            
                            if all_values:
                                avg_scores[detector_label] = np.mean(all_values)
                        
                        if avg_scores:
                            best_detector = max(avg_scores, key=avg_scores.get)
                            st.markdown(f"**Hoogste Prestaties**: {best_detector} met gemiddelde metric-score van {avg_scores[best_detector]:.3f}")
                        
                        # Time comparisons
                        fit_times = {}
                        predict_times = {}
                        
                        for detector_key, detector_result in st.session_state.results.items():
                            if not detector_key.startswith('detector_'):
                                continue
                                
                            detector_label = f"Detector {detector_key[-1]} ({detector_result['detector_name']})"
                            fit_times[detector_label] = detector_result['fit_time']
                            predict_times[detector_label] = detector_result['predict_time']
                        
                        if fit_times:
                            fastest_detector = min(fit_times, key=fit_times.get)
                            st.markdown(f"**Minste Fit-tijd**: {fastest_detector} met fit-tijd van {fit_times[fastest_detector]:.2f} seconden")
                        
                        if predict_times:
                            fastest_predict_detector = min(predict_times, key=predict_times.get)
                            st.markdown(f"**Minste Predict-tijd**: {fastest_predict_detector} met predict-tijd van {predict_times[fastest_predict_detector]:.2f} seconden")
                        
                        # Data properties
                        if st.session_state.uploaded_data_valid:
                            x, y = st.session_state.uploaded_data
                        elif 'selected_dataset_name' in st.session_state:
                            x, y = load_dataset(st.session_state.selected_dataset_name)
                        else:
                            x, y = None, None
                            
                        if x is not None:
                            st.markdown("**Impact van Data-eigenschappen**:")
                            dim, is_uni = get_dimension(x), is_univariate(x)
                            st.write(f"- Dimensionaliteit: {dim}, Univariate: {is_uni}")
                        
                    except Exception as analysis_error:
                        st.error(f"Fout bij genereren van prestatie-analyse: {analysis_error}")


if __name__ == "__main__":
    main()
