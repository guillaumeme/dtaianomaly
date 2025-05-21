#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import inspect
import traceback
import time
import logging
import os
import sys
from typing import List, Dict, Any, Union
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io  # Added for BytesIO

# Add these imports to allow for direct execution
from streamlit.web import cli as stcli
from streamlit import runtime

from dtaianomaly import data, anomaly_detection, preprocessing, evaluation, thresholding
from dtaianomaly.pipeline import EvaluationPipeline
from dtaianomaly.workflow.utils import convert_to_proba_metrics
from dtaianomaly.utils import is_valid_array_like, is_univariate, get_dimension
from dtaianomaly.evaluation import AreaUnderROC, Precision

# Import custom detector
try:
    from dtaianomaly.demonstrator.custom_detector_demo import NbSigmaAnomalyDetector
    CUSTOM_DETECTOR_AVAILABLE = True
    print("Successfully imported NbSigmaAnomalyDetector")
except ImportError:
    CUSTOM_DETECTOR_AVAILABLE = False
    print("Could not import NbSigmaAnomalyDetector")

# Configure logging
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Create logger
logger = logging.getLogger('streamlit_app')
if not logger.handlers:  
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
    st.session_state.experience_level = "Expert"
if 'uploaded_data_valid' not in st.session_state:
    st.session_state.uploaded_data_valid = False
if 'selected_dataset_name' not in st.session_state:
    st.session_state.selected_dataset_name = None
if 'detector_tabs' not in st.session_state:
    st.session_state.detector_tabs = [{'id': 0, 'mode': 'Expert', 'detector': None}]
if 'next_tab_id' not in st.session_state:
    st.session_state.next_tab_id = 1
if 'current_detector_tab' not in st.session_state:
    st.session_state.current_detector_tab = 0
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'detector_hyperparams' not in st.session_state:
    st.session_state.detector_hyperparams = {}
if 'detector_preprocessors' not in st.session_state:
    st.session_state.detector_preprocessors = {}
if 'preprocessor_hyperparams' not in st.session_state:
    st.session_state.preprocessor_hyperparams = {}
if 'custom_visualizations' not in st.session_state:
    st.session_state.custom_visualizations = {}
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

# --- Helper functions ---

# Mapping from dtaianomaly modules to keys in the custom_components dictionary
# Adjust keys ('detectors', 'preprocessors', etc.) if needed to match your custom_components structure
CUSTOM_COMPONENT_KEY_MAP = {
    anomaly_detection: 'detectors',
    preprocessing: 'preprocessors',
    evaluation: 'metrics',
    thresholding: 'thresholds',
    data: 'data_loaders',
    # Remove reference to data.synthetic which doesn't exist
}

def get_available_options(module, base_class, include_functions=False):
    """Dynamically retrieves available options from a dtaianomaly module."""
    options = []
    
    # First get standard options from the module
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class and
                name not in ['BaseDetector', 'PyODAnomalyDetector', 'Preprocessor', 'Metric', 'ProbaMetric',
                             'BinaryMetric', 'ThresholdMetric', 'BestThresholdMetric', 'Thresholding',
                             'LazyDataLoader']):
            options.append(name)
        elif include_functions and inspect.isfunction(obj) and 'time_series' in name:
            options.append(name)
    
    # Check for custom components in session state
    import streamlit as st
    if 'custom_components' in st.session_state:
        # If we're looking for detectors, add any custom ones
        if module is anomaly_detection and base_class is anomaly_detection.BaseDetector:
            if 'detectors' in st.session_state.custom_components:
                for name in st.session_state.custom_components['detectors']:
                    if name not in options:
                        logger.info(f"Adding custom detector to options: {name}")
                        options.append(name)
    
    return options


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads a dataset from dtaianomaly.data or custom loaders."""
    # Add check for None dataset_name
    if dataset_name is None:
        logger.error("load_dataset called with None dataset_name")
        raise ValueError("No dataset selected.")
        
    if hasattr(data, dataset_name):
        dataset_function = getattr(data, dataset_name)
    else:
        # Handle potential missing attribute error more gracefully
        try:
            # Check custom components if available
            custom_components = getattr(st.session_state, 'custom_components', {})
            data_loaders = custom_components.get('data_loaders', {})
            if dataset_name in data_loaders:
                dataset_function = data_loaders[dataset_name]
            else:
                raise AttributeError(f"Dataset '{dataset_name}' not found in dtaianomaly.data or custom loaders.")
        except AttributeError as e:
            logger.error(f"Error finding dataset: {e}")
            raise AttributeError(f"Dataset '{dataset_name}' not found. Check available datasets.")
        except Exception as e:
            logger.error(f"Unexpected error checking custom data loaders: {e}")
            raise AttributeError(f"Dataset '{dataset_name}' not found.")
            
    x, y = dataset_function()
    if not is_valid_array_like(x) or not is_valid_array_like(y) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Invalid dataset: must be array-like with binary labels (0 or 1).")
    return x, y


def load_component(module, component_name: str, **kwargs):
    """Dynamically loads a component from dtaianomaly based on name and parameters."""
    if component_name is None or component_name == "None":
        return None
    try:
        # First check if it's a custom component
        if 'custom_components' in st.session_state:
            # Determine component type based on module
            component_type = None
            for m, key in CUSTOM_COMPONENT_KEY_MAP.items():
                if module is m:
                    component_type = key
                    break
                    
            # Check if the component exists in custom components
            if component_type and component_type in st.session_state.custom_components:
                if component_name in st.session_state.custom_components[component_type]:
                    logger.info(f"Loading custom component: {component_name}")
                    component_class = st.session_state.custom_components[component_type][component_name]
                    try:
                        return component_class(**kwargs)
                    except Exception as e:
                        logger.error(f"Error initializing custom component {component_name}: {e}")
                        st.error(f"Error initializing custom component {component_name}: {e}")
                        return None
        
        # If not found in custom components, try loading from module
        component_class = getattr(module, component_name)
        try:
            return component_class(**kwargs)
        except TypeError as e:
            st.error(f"Error initializing {component_name}: Wrong parameters: {e}")
            # Show the expected parameters
            sig = inspect.signature(component_class.__init__)
            param_str = ", ".join([f"{p}" for p in sig.parameters if p != 'self'])
            st.info(f"Expected parameters for {component_name}: {param_str}")
            return None
        except Exception as e:
            st.error(f"Error initializing {component_name}: {e}")
            return None
    except AttributeError:
        st.error(f"Component {component_name} not found in the module.")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading {component_name}: {e}")
        return None


def get_parameter_documentation(class_name, param_name):
    """
    Extract documentation for a specific parameter of a class.
    
    Args:
        class_name: Name of the class
        param_name: Name of the parameter
        
    Returns:
        str: Documentation for the parameter, or None if not found
    """
    try:
        # Check if it's a standard detector
        if hasattr(anomaly_detection, class_name):
            detector_class = getattr(anomaly_detection, class_name)
            doc = inspect.getdoc(detector_class)
            
            if not doc:
                return None
                
            # Look for parameter documentation in the docstring
            # Pattern: parameter_name: type
            #          Description over potentially multiple lines
            lines = doc.split('\n')
            param_found = False
            param_doc = []
            
            # First try Parameters section format
            if "Parameters" in doc:
                params_section = doc.split("Parameters")[1].split("---")[0].split("Attributes")[0]
                param_lines = params_section.strip().split('\n')
                for i, line in enumerate(param_lines):
                    if param_name in line and ":" in line:
                        param_found = True
                        # Extract description from subsequent indented lines
                        j = i + 1
                        while j < len(param_lines) and (param_lines[j].startswith(' ') or param_lines[j].startswith('\t')):
                            param_doc.append(param_lines[j].strip())
                            j += 1
                        break
            
            # If not found, try another common format
            if not param_found:
                for i, line in enumerate(lines):
                    if line.strip().startswith(param_name + ':') or line.strip().startswith(param_name + ' :'):
                        param_found = True
                        param_doc.append(line.split(':', 1)[1].strip())
                        # Check for multi-line descriptions (indented lines following parameter)
                        j = i + 1
                        while j < len(lines) and (lines[j].startswith(' ') or lines[j].startswith('\t')):
                            param_doc.append(lines[j].strip())
                            j += 1
                        break
            
            if param_found and param_doc:
                return ' '.join(param_doc)
        
        # Custom detectors - similar approach could be added if needed
        
        return None
    except Exception as e:
        logger.error(f"Error getting parameter documentation: {e}")
        return None

def generate_hyperparam_inputs(detector_name: str, prefix: str) -> Dict[str, Any]:
    """Generates input fields for detector hyperparameters with a unique prefix."""
    try:
        detector_class = getattr(anomaly_detection, detector_name)
        signature = inspect.signature(detector_class.__init__)
        hyperparams = {}
        
        for param_name, param_obj in signature.parameters.items():
            if param_name in ['self', 'kwargs', 'args']:
                continue
                
            # Get parameter documentation for help text
            param_help = get_parameter_documentation(detector_name, param_name)
            if not param_help:
                param_help = f"Parameter {param_name}"
                
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
                    f"{param_name} option", ["Auto (fft)", "Manual"], 
                    key=f"{prefix}_window_size_option", 
                    index=0,
                    help=param_help
                )
                if window_size_option == "Manual":
                    hyperparams['window_size'] = int(st.number_input(
                        f"Manual {param_name}", 
                        min_value=1, 
                        value=20 if default_value is None else int(default_value),
                        key=f"{prefix}_window_size_manual",
                        help=param_help
                    ))
                else:
                    hyperparams['window_size'] = 'fft'
            else:
                # Handle different parameter types
                if isinstance(default_value, bool):
                    hyperparams[param_name] = st.checkbox(
                        f"{param_name}", 
                        value=default_value,
                        key=f"{prefix}_{param_name}",
                        help=param_help
                    )
                elif isinstance(default_value, int):
                    hyperparams[param_name] = st.number_input(
                        f"{param_name}", 
                        value=default_value, 
                        step=1,
                        format="%d", 
                        key=f"{prefix}_{param_name}",
                        help=param_help
                    )
                elif isinstance(default_value, float):
                    hyperparams[param_name] = st.number_input(
                        f"{param_name}", 
                        value=default_value, 
                        step=0.1,
                        format="%.2f", 
                        key=f"{prefix}_{param_name}",
                        help=param_help
                    )
                elif isinstance(default_value, str):
                    hyperparams[param_name] = st.text_input(
                        f"{param_name}", 
                        value=default_value,
                        key=f"{prefix}_{param_name}",
                        help=param_help
                    )
                elif default_value is None:
                    # For parameters with no clear type, provide options
                    param_type_option = st.selectbox(
                        f"{param_name} type", 
                        ["None", "String", "Integer", "Float", "Boolean"],
                        key=f"{prefix}_{param_name}_type",
                        help=param_help
                    )
                    
                    if param_type_option == "String":
                        hyperparams[param_name] = st.text_input(
                            f"{param_name} (string)", 
                            value="",
                            key=f"{prefix}_{param_name}_value",
                            help=param_help
                        )
                    elif param_type_option == "Integer":
                        hyperparams[param_name] = st.number_input(
                            f"{param_name} (integer)", 
                            value=0,
                            step=1, 
                            key=f"{prefix}_{param_name}_value",
                            help=param_help
                        )
                    elif param_type_option == "Float":
                        hyperparams[param_name] = st.number_input(
                            f"{param_name} (float)", 
                            value=0.0,
                            step=0.1, 
                            format="%.2f",
                            key=f"{prefix}_{param_name}_value",
                            help=param_help
                        )
                    elif param_type_option == "Boolean":
                        hyperparams[param_name] = st.checkbox(
                            f"{param_name} (boolean)", 
                            value=False,
                            key=f"{prefix}_{param_name}_value",
                            help=param_help
                        )
                    else:  # None
                        hyperparams[param_name] = None
                else:
                    # For complex types or containers (list, dict, etc.), use string input
                    st.warning(f"Parameter {param_name} has a complex type. Enter as string.")
                    hyperparams[param_name] = st.text_input(
                        f"{param_name} (complex type)", 
                        value=str(default_value),
                        key=f"{prefix}_{param_name}",
                        help=param_help
                    )
        
        return hyperparams
    except Exception as e:
        st.error(f"Error generating hyperparameters for {detector_name}: {e}")
        return {}


def get_default_hyperparams(component_class):
    """Gets the default hyperparameters for a given class."""
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


# --- Function to validate uploaded data ---
def validate_uploaded_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Validates the uploaded dataset and returns x, y as numpy arrays.
    Also returns an error message if the data is invalid.
    """
    required_columns = ['Time Step', 'Value', 'Label']
    if not all(col in df.columns for col in required_columns):
        return None, None, "The dataset must contain the columns 'Time Step', 'Value', and 'Label'."

    if not pd.api.types.is_numeric_dtype(df['Time Step']):
        return None, None, "The 'Time Step' column must be numeric."

    if not pd.api.types.is_numeric_dtype(df['Value']):
        return None, None, "The 'Value' column must be numeric."

    if not pd.api.types.is_numeric_dtype(df['Label']):
        return None, None, "The 'Label' column must be numeric."

    if not set(df['Label']).issubset({0, 1}):
        return None, None, "The 'Label' column must contain binary values (0 or 1)."

    x = df[['Time Step', 'Value']].to_numpy()
    y = df['Label'].to_numpy()

    if not is_valid_array_like(x) or not is_valid_array_like(y):
        return None, None, "The dataset is invalid: must be array-like."

    return x, y, ""


def configure_sidebar():
    """Configures the Streamlit sidebar with data, evaluation metrics, and visualization options."""
    st.sidebar.header("Configuration")

    # --- 1. Dataset ---
    st.sidebar.subheader("1. Dataset")
    upload_option = st.sidebar.radio(
        "Choose a dataset source:",
        ["Use built-in dataset", "Upload custom dataset"],
        key="upload_option"
    )

    if upload_option == "Upload custom dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or Excel (columns: Time Step*, Value(s), Label)", # Clarify expected columns
            type=["csv", "xlsx", "xls"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # --- Validation ---
                x, y, error = validate_uploaded_data(df)
                if error:
                    st.sidebar.error(f"Invalid Upload: {error}")
                    st.session_state.uploaded_data_valid = False
                    st.session_state.uploaded_data = None # Clear previous valid data
                else:
                    st.session_state.uploaded_data = (x, y)
                    st.session_state.uploaded_data_valid = True
                    st.session_state.selected_dataset_name = f"Uploaded: {uploaded_file.name}" # Use filename as identifier
                    st.sidebar.success(f"Uploaded '{uploaded_file.name}' is valid ({x.shape[0]} samples, {x.shape[1]} feature(s)).")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                logger.error(f"Error loading uploaded file: {traceback.format_exc()}")
                st.session_state.uploaded_data_valid = False
                st.session_state.uploaded_data = None
        # If no file is uploaded but option is selected, ensure state reflects this
        elif not st.session_state.get('uploaded_data_valid', False):
             st.session_state.uploaded_data_valid = False
             st.session_state.uploaded_data = None
             # Clear selected dataset name if switching from built-in
             # if st.session_state.selected_dataset_name and not st.session_state.selected_dataset_name.startswith("Uploaded:"):
             #      st.session_state.selected_dataset_name = None


    else: # Use built-in dataset
        # Only include demonstration_time_series
        dataset_options = ["demonstration_time_series"]
        
        st.session_state.selected_dataset_name = st.sidebar.selectbox(
            "Select Dataset",
            dataset_options,
            key="dataset_select",
            index=0,
            help="Choose a dataset to analyze."
        )
        st.session_state.uploaded_data_valid = False
    
    # Removed global preprocessor setting - will be configured per detector
    
    # Evaluation Metrics Selection
    st.sidebar.subheader("2. Evaluation Metrics")
    metric_options = get_available_options(evaluation, evaluation.Metric)
    st.session_state.selected_metrics = st.sidebar.multiselect(
        "Select Metrics", 
        metric_options,
        default=["AreaUnderROC", "Precision"],
        key="metrics_select"
    )
    
    # Thresholding Selection
    st.sidebar.subheader("3. Thresholding")
    threshold_options = get_available_options(thresholding, thresholding.Thresholding)
    if not threshold_options:
        st.error("No thresholding classes found. Check the installation.")
        st.stop()
        
    # Use radio buttons for single selection instead of multiselect
    # Get documentation for each thresholding method to use as help text
    threshold_help_texts = {}
    for threshold_name in threshold_options:
        threshold_help_texts[threshold_name] = get_threshold_documentation(threshold_name)
    
    selected_threshold = st.sidebar.radio(
        "Select Threshold", 
        threshold_options,
        index=0,
        key="threshold_select",
        help="Choose a thresholding method to determine anomaly boundaries."
    )
    st.session_state.selected_thresholds = [selected_threshold]
    
    # Threshold Hyperparameters
    for threshold_name in st.session_state.selected_thresholds:
        # Create threshold params if they don't exist
        if threshold_name not in st.session_state.threshold_hyperparams:
            # Special handling for FixedCutoff
            if threshold_name == 'FixedCutoff':
                st.session_state.threshold_hyperparams[threshold_name] = init_fixed_cutoff_params()
            else:
                # Standard handling for other thresholds
                threshold_class = getattr(thresholding, threshold_name)
                st.session_state.threshold_hyperparams[threshold_name] = get_default_hyperparams(threshold_class)
        
        # Generate inputs for each parameter with help tooltips
        for param_name in st.session_state.threshold_hyperparams[threshold_name]:
            # Get parameter documentation for help text
            param_help = get_threshold_parameter_documentation(threshold_name, param_name)
            if not param_help:
                # If no specific documentation, use general threshold information
                param_help = threshold_help_texts.get(threshold_name, f"Parameter for {threshold_name}")
                
            default_value = st.session_state.threshold_hyperparams[threshold_name][param_name]
            if isinstance(default_value, float):
                st.session_state.threshold_hyperparams[threshold_name][param_name] = st.sidebar.number_input(
                    f"{param_name}", value=default_value, step=0.1, format="%.2f",
                    key=f"threshold_{threshold_name}_{param_name}",
                    help=param_help
                )
            elif isinstance(default_value, int):
                st.session_state.threshold_hyperparams[threshold_name][param_name] = st.sidebar.number_input(
                    f"{param_name}", value=default_value, step=1,
                    key=f"threshold_{threshold_name}_{param_name}",
                    help=param_help
                )
            elif isinstance(default_value, bool):
                st.session_state.threshold_hyperparams[threshold_name][param_name] = st.sidebar.checkbox(
                    f"{param_name}", value=default_value,
                    key=f"threshold_{threshold_name}_{param_name}",
                    help=param_help
                )
            elif isinstance(default_value, str):
                st.session_state.threshold_hyperparams[threshold_name][param_name] = st.sidebar.text_input(
                    f"{param_name}", value=default_value,
                    key=f"threshold_{threshold_name}_{param_name}",
                    help=param_help
                )
    
    # Visualization Options section removed

def add_detector_tab():
    """Adds a new detector tab to the session state."""
    new_tab = {
        'id': st.session_state.next_tab_id,
        'mode': 'Expert',
        'detector': None
    }
    st.session_state.detector_tabs.append(new_tab)
    st.session_state.next_tab_id += 1
    st.session_state.current_detector_tab = len(st.session_state.detector_tabs) - 1
    
def remove_detector_tab(tab_id):
    """Removes a detector tab from the session state."""
    for i, tab in enumerate(st.session_state.detector_tabs):
        if tab['id'] == tab_id:
            st.session_state.detector_tabs.pop(i)
            # If the current tab was removed, go to the first tab
            if st.session_state.current_detector_tab >= len(st.session_state.detector_tabs):
                st.session_state.current_detector_tab = 0
            break

def get_detector_documentation(detector_name):
    """
    Extract documentation from detector class.
    
    Args:
        detector_name: Name of the detector class
        
    Returns:
        str: The main description part of the documentation string for the detector
    """
    try:
        # First check if it's a custom detector
        if 'custom_components' in st.session_state and 'detectors' in st.session_state.custom_components:
            if detector_name in st.session_state.custom_components['detectors']:
                detector_class = st.session_state.custom_components['detectors'][detector_name]
                doc = inspect.getdoc(detector_class)
                if doc:
                    # Extract just the main description (everything before Parameters, Notes or ---)
                    main_description = doc.split("Parameters")[0].split("---")[0].split("Notes")[0].strip()
                    return main_description
        
        # Check standard detectors
        if hasattr(anomaly_detection, detector_name):
            detector_class = getattr(anomaly_detection, detector_name)
            doc = inspect.getdoc(detector_class)
            if doc:
                # Extract just the main description (everything before Parameters, Notes or ---)
                main_description = doc.split("Parameters")[0].split("---")[0].split("Notes")[0].strip()
                return main_description
        
        return f"No documentation available for {detector_name}."
    except Exception as e:
        logger.error(f"Error getting documentation for {detector_name}: {e}")
        return f"Error retrieving documentation for {detector_name}."

def configure_detector_tab(tab_index):
    """Configures the detector settings for a specific tab."""
    tab = st.session_state.detector_tabs[tab_index]
    detector_key = f"detector_{tab_index + 1}"
    
    # Detector Selection
    detector_options = get_available_options(anomaly_detection, anomaly_detection.BaseDetector)
    
    # Log the available detector options for debugging
    logger.info(f"Available detector options: {detector_options}")
    
    # Explicitly add the custom detector if available
    if CUSTOM_DETECTOR_AVAILABLE and 'NbSigmaAnomalyDetector' not in detector_options:
        detector_options.append('NbSigmaAnomalyDetector')
        logger.info("Explicitly added NbSigmaAnomalyDetector to options")
    
    # Check for custom detectors directly in session state
    if 'custom_components' in st.session_state and 'detectors' in st.session_state.custom_components:
        custom_detectors = list(st.session_state.custom_components['detectors'].keys())
        logger.info(f"Found custom detectors: {custom_detectors}")
        # Add any custom detectors that aren't already in the options
        for detector in custom_detectors:
            if detector not in detector_options:
                detector_options.append(detector)
                logger.info(f"Added custom detector to options: {detector}")
    
    # Determine default detector index
    default_index = 0
    if 'NbSigmaAnomalyDetector' in detector_options:
        logger.info("Found NbSigmaAnomalyDetector in options, setting as default")
        default_index = detector_options.index('NbSigmaAnomalyDetector')
    
    selected_detector = st.selectbox(
        "Select Detector",
        detector_options,
        index=default_index,
        key=f"detector_select_{tab['id']}"
    )
    tab['detector'] = selected_detector
    
    # Display detector documentation
    detector_doc = get_detector_documentation(selected_detector)
    # Format the documentation before using it in the f-string
    formatted_doc = detector_doc.replace('\n', '<br>')
    
    with st.expander("Detector Documentation", expanded=True):
        # Add styling to make documentation more readable
        st.markdown(f"""
        <div style="background-color:#f8f9fa; padding:15px; border-radius:5px; border-left:5px solid #4CAF50;">
        <h4 style="color:#4CAF50;">{selected_detector}</h4>
        <div style="color:#333; font-size:0.95em;">
        {formatted_doc}
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if the detector requires window_size
    try:
        detector_class = getattr(anomaly_detection, selected_detector)
        init_params = inspect.signature(detector_class.__init__).parameters
        requires_window_size = 'window_size' in init_params
    except (AttributeError, TypeError):
        requires_window_size = False
    
    # Preprocessing Pipeline Section
    st.subheader("Preprocessing Pipeline")
    
    # Initialize preprocessor list for this detector if not exists
    if detector_key not in st.session_state.detector_preprocessors:
        st.session_state.detector_preprocessors[detector_key] = []
        
    # Initialize preprocessor hyperparams for this detector if not exists
    if detector_key not in st.session_state.preprocessor_hyperparams:
        st.session_state.preprocessor_hyperparams[detector_key] = {}
    
    # Get available preprocessors
    preprocessor_options = ["None"] + get_available_options(preprocessing, preprocessing.Preprocessor)
    
    # Display current preprocessors in the pipeline
    if not st.session_state.detector_preprocessors[detector_key]:
        st.info("No preprocessing steps configured. Add preprocessors below.")
        
    pipeline_updated = False
    
    # Show existing preprocessors
    for idx, preproc_name in enumerate(st.session_state.detector_preprocessors[detector_key]):
        with st.container():
            cols = st.columns([1, 6, 1])
            with cols[0]:
                st.write(f"{idx+1}.")
            with cols[1]:
                st.write(f"**{preproc_name}**")
            with cols[2]:
                if st.button("Remove", key=f"remove_preproc_{detector_key}_{idx}"):
                    st.session_state.detector_preprocessors[detector_key].pop(idx)
                    # Also remove hyperparams for this preprocessor
                    preproc_param_key = f"{detector_key}_preproc_{idx}"
                    if preproc_param_key in st.session_state.preprocessor_hyperparams[detector_key]:
                        del st.session_state.preprocessor_hyperparams[detector_key][preproc_param_key]
                    pipeline_updated = True
                    st.rerun()
            
            # Display hyperparameters for this preprocessor
            preproc_param_key = f"{detector_key}_preproc_{idx}"
            with st.expander(f"{preproc_name} Settings", expanded=False):
                # Initialize hyperparams if not exists
                if preproc_param_key not in st.session_state.preprocessor_hyperparams[detector_key]:
                    preproc_class = getattr(preprocessing, preproc_name)
                    st.session_state.preprocessor_hyperparams[detector_key][preproc_param_key] = get_default_hyperparams(preproc_class)
                        
                # Display hyperparameters
                hyperparams = st.session_state.preprocessor_hyperparams[detector_key][preproc_param_key]
                for param_name, default_value in hyperparams.items():
                    if isinstance(default_value, float):
                        hyperparams[param_name] = st.number_input(
                            f"{param_name}", value=default_value, step=0.1, format="%.2f",
                            key=f"{preproc_param_key}_{param_name}"
                        )
                    elif isinstance(default_value, int):
                        hyperparams[param_name] = st.number_input(
                            f"{param_name}", value=default_value, step=1,
                            key=f"{preproc_param_key}_{param_name}"
                        )
                    elif isinstance(default_value, bool):
                        hyperparams[param_name] = st.checkbox(
                            f"{param_name}", value=default_value,
                            key=f"{preproc_param_key}_{param_name}"
                        )
                    elif isinstance(default_value, str):
                        hyperparams[param_name] = st.text_input(
                            f"{param_name}", value=default_value,
                            key=f"{preproc_param_key}_{param_name}"
                        )
    
    # Add new preprocessor
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            new_preprocessor = st.selectbox(
                "Add Preprocessor", 
                preprocessor_options,
                key=f"add_preproc_{detector_key}"
            )
        with cols[1]:
            if st.button("Add", key=f"add_preproc_btn_{detector_key}"):
                if new_preprocessor != "None":
                    st.session_state.detector_preprocessors[detector_key].append(new_preprocessor)
                    # Initialize hyperparams for this new preprocessor
                    new_idx = len(st.session_state.detector_preprocessors[detector_key]) - 1
                    preproc_param_key = f"{detector_key}_preproc_{new_idx}"
                    preproc_class = getattr(preprocessing, new_preprocessor)
                    st.session_state.preprocessor_hyperparams[detector_key][preproc_param_key] = get_default_hyperparams(preproc_class)
                    pipeline_updated = True
                    st.rerun()
    
    # Detector Hyperparameters
    st.subheader("Detector Configuration")
    
    # Advanced hyperparameters for experts
    with st.expander("Advanced Settings"):
        hyperparams = generate_hyperparam_inputs(selected_detector, prefix=f"expert_{detector_key}")
        
        # Ensure window_size is included if needed
        if requires_window_size and 'window_size' not in hyperparams:
            st.warning(f"Detector {selected_detector} requires window_size but it wasn't specified. Using 'fft' as default.")
            hyperparams['window_size'] = 'fft'
                
        st.session_state.detector_hyperparams[detector_key] = hyperparams
    
    if pipeline_updated:
        st.experimental_rerun()
        
    return tab['detector']

def run_detector(tab_index):
    """Executes the detection pipeline for a specific detector tab."""
    logger.info(f"Starting detector execution for tab {tab_index}")
    tab = st.session_state.detector_tabs[tab_index]
    detector_key = f"detector_{tab_index + 1}"
    selected_detector = tab['detector']
    
    if not selected_detector:
        logger.warning("No detector selected")
        st.error("Please select a detector first.")
        return
    
    # Run pipeline for this detector
    detector_config = {
        "name": selected_detector,
        "key": detector_key,
        "hyperparams": st.session_state.detector_hyperparams.get(detector_key, {})
    }
    logger.debug(f"Detector configuration: {detector_config}")
    
    # Datasets and preprocessing
    try:
        if st.session_state.uploaded_data_valid:
            logger.debug("Using uploaded data")
            x, y = st.session_state.uploaded_data
            st.write("Successfully loaded uploaded dataset.")
        else:
            logger.debug(f"Loading dataset: {st.session_state.selected_dataset_name}")
            x, y = load_dataset(st.session_state.selected_dataset_name)
            st.write("Successfully loaded built-in dataset.")
        
        # Log info about input data
        logger.debug(f"Input data type: {type(x)}, Shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
        st.info(f"Input data type: {type(x)}, Shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
        
        # Set export data
        st.session_state.export_data['time_steps'] = format_time_steps(None, x.shape[0])
        st.session_state.export_data['original_x'] = x
        st.session_state.export_data['original_y'] = y
        
        # Process data using detector-specific preprocessing pipeline
        processed_x, processed_y = x, y
        
        # Apply preprocessing pipeline for this detector
        if detector_key in st.session_state.detector_preprocessors and st.session_state.detector_preprocessors[detector_key]:
            preprocessor_pipeline = st.session_state.detector_preprocessors[detector_key]
            st.write(f"Applying {len(preprocessor_pipeline)} preprocessing steps:")
            
            # Process data using the pipeline
            for idx, preprocessor_name in enumerate(preprocessor_pipeline):
                logger.debug(f"Applying preprocessor {idx+1}/{len(preprocessor_pipeline)}: {preprocessor_name}")
                st.write(f"Step {idx+1}: {preprocessor_name}")
                
                # Get preprocessor hyperparams
                preproc_param_key = f"{detector_key}_preproc_{idx}"
                preproc_hyperparams = st.session_state.preprocessor_hyperparams.get(detector_key, {}).get(preproc_param_key, {})
                
                # Load and apply preprocessor
                preprocessor = load_component(preprocessing, preprocessor_name, **preproc_hyperparams)
                if preprocessor:
                    # Handle both single output and tuple output from preprocessor
                    preprocess_result = preprocessor.fit_transform(processed_x)
                    if isinstance(preprocess_result, tuple):
                        logger.debug(f"Preprocessor {preprocessor_name} returned a tuple, extracting first element")
                        processed_x, processed_y = preprocess_result
                    else:
                        logger.debug(f"Preprocessor {preprocessor_name} returned a single object")
                        processed_x = preprocess_result
                else:
                    logger.warning(f"Failed to initialize preprocessor {preprocessor_name}")
                    st.warning(f"Failed to initialize preprocessor {preprocessor_name}, skipping this step")
        else:
            logger.debug("No preprocessing pipeline defined for this detector, using original data")
            st.write("No preprocessing steps defined. Using original data.")
            
        st.session_state.export_data['processed_x'] = processed_x
        st.session_state.export_data['processed_y'] = processed_y
            
        # Convert to numpy array with proper handling for inhomogeneous shapes
        if not isinstance(processed_x, np.ndarray):
            try:
                # Try to directly convert to float64 array
                processed_x = np.array(processed_x, dtype=np.float64)
            except ValueError as e:
                # If we get an inhomogeneous shape error, log information about the data
                st.warning(f"Data has uneven dimensions: {e}")
                
                # Log info about the structure
                if hasattr(processed_x, '__len__'):
                    st.info(f"Data has {len(processed_x)} samples")
                    
                # Try to pad the data to make it homogeneous
                if isinstance(processed_x, list):
                    # Find the maximum length
                    if all(isinstance(item, (list, tuple)) for item in processed_x):
                        max_len = max(len(item) for item in processed_x)
                        st.info(f"Padding data to uniform length of {max_len}")
                        # Pad with zeros
                        processed_x = [list(item) + [0.0] * (max_len - len(item)) for item in processed_x]
                        processed_x = np.array(processed_x, dtype=np.float64)
                    else:
                        # If it's not a list of lists, try to reshape as needed
                        st.info("Trying to convert data to 2D array")
                        processed_x = np.array(processed_x, dtype=np.object)
                        
                        # For univariate data, reshape to a column vector (n_samples, 1)
                        if processed_x.ndim == 1:
                            processed_x = processed_x.reshape(-1, 1)
        
        # If processed_x is still a tuple after conversion attempts, extract the first element
        if isinstance(processed_x, tuple):
            st.info("processed_x is still a tuple, using first element")
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
                st.warning("Could not convert data to float64. Trying object array.")
                processed_x = processed_x.astype(object)
        
        # Handle nans or infinities if present
        if hasattr(processed_x, 'dtype') and np.issubdtype(processed_x.dtype, np.number):
            if np.isnan(processed_x).any() or np.isinf(processed_x).any():
                # Replace NaN with zeros and infinities with large values
                processed_x = np.nan_to_num(processed_x)
        
        # Check if data is empty
        if hasattr(processed_x, 'size') and processed_x.size == 0:
            st.error("The dataset is empty after preprocessing.")
            return False
        
        # Check for constant values across samples
        if isinstance(processed_x, np.ndarray) and processed_x.size > 0:
            if np.all(processed_x == processed_x[0]):
                st.error("The dataset has no variance after preprocessing.")
                return False
            
        # Log info about the processed data
        logger.debug(f"Processed data type: {type(processed_x)}, Shape: {processed_x.shape if hasattr(processed_x, 'shape') else 'unknown'}")
        st.info(f"Processed data type: {type(processed_x)}, Shape: {processed_x.shape if hasattr(processed_x, 'shape') else 'unknown'}")
        
        # Log a sample of the processed data
        if isinstance(processed_x, np.ndarray) and processed_x.size > 0:
            logger.debug(f"Processed data sample (first 5 elements): {processed_x.flatten()[:5]}")
            logger.debug(f"Processed data min: {np.min(processed_x)}, max: {np.max(processed_x)}")
            
        # Try to load detector - first check if it's our custom detector
        detector = None
        if selected_detector == 'NbSigmaAnomalyDetector' and CUSTOM_DETECTOR_AVAILABLE:
            logger.info("Creating NbSigmaAnomalyDetector instance directly")
            try:
                # Initialize directly with hyperparameters
                detector = NbSigmaAnomalyDetector(**detector_config["hyperparams"])
                logger.info(f"Successfully created NbSigmaAnomalyDetector with params: {detector_config['hyperparams']}")
                st.success("Successfully initialized NbSigmaAnomalyDetector!")
            except Exception as e:
                logger.error(f"Error initializing NbSigmaAnomalyDetector: {e}")
                st.error(f"Error initializing NbSigmaAnomalyDetector: {e}")
        
        # If detector is still None, try custom components
        if detector is None and 'custom_components' in st.session_state and 'detectors' in st.session_state.custom_components:
            if selected_detector in st.session_state.custom_components['detectors']:
                logger.info(f"Loading custom detector from session state: {selected_detector}")
                detector_class = st.session_state.custom_components['detectors'][selected_detector]
                try:
                    detector = detector_class(**detector_config["hyperparams"])
                    logger.info(f"Successfully initialized custom detector: {selected_detector}")
                    st.success(f"Successfully initialized custom detector: {selected_detector}")
                except Exception as e:
                    logger.error(f"Error initializing custom detector {selected_detector}: {e}")
                    st.error(f"Error initializing custom detector {selected_detector}: {e}")
        
        # If detector is still None, try standard loading  
        if detector is None:
            logger.debug(f"Loading standard detector: {detector_config['name']}")
            detector = load_component(anomaly_detection, detector_config["name"], 
                                    **detector_config["hyperparams"])
        
        # Display what detector was loaded
        st.write(f"Using detector: {selected_detector}")
        
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
                
                # Special handling for FixedCutoff to apply the cutoff value directly
                elif threshold_name == 'FixedCutoff':
                    logger.info("Using special handling for FixedCutoff")
                    
                    # Ensure processed_y is not None - use original y if it is
                    if processed_y is None:
                        logger.warning("processed_y is None, using original y for metrics calculation")
                        processed_y = y
                    
                    # Get the cutoff value from parameters
                    threshold_params = st.session_state.threshold_hyperparams.get(threshold_name, {})
                    # Log the actual parameters to debug
                    logger.debug(f"FixedCutoff parameters: {threshold_params}")
                    
                    # Check if the parameter exists and is the correct type
                    if 'cutoff' not in threshold_params:
                        logger.warning("cutoff parameter not found in threshold_params, using default value 0.5")
                        cutoff = 0.5
                    else:
                        cutoff = float(threshold_params['cutoff'])
                    
                    logger.debug(f"Using fixed cutoff value: {cutoff}")
                    
                    # Apply threshold
                    y_pred = (anomaly_scores >= cutoff).astype(int)
                    logger.debug(f"Created binary predictions using fixed cutoff threshold")
                    st.info(f"Using fixed cutoff value: {cutoff}")
                    
                    # Store predictions
                    thresholded_predictions[threshold_name] = y_pred
                    logger.debug(f"Stored thresholded predictions for {threshold_name}")
                    
                    # Log predictions summary
                    if isinstance(y_pred, np.ndarray):
                        anomaly_count = np.sum(y_pred)
                        logger.debug(f"Predictions summary - anomalies: {anomaly_count}, normal: {len(y_pred) - anomaly_count}")
                        logger.debug(f"Anomaly percentage: {anomaly_count/len(y_pred)*100:.2f}%")
                    
                    # Calculate metrics for FixedCutoff
                    metrics_dict = {}
                    
                    try:
                        # Import metrics from scikit-learn
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
                        
                        # Add the metrics to all_metrics
                        all_metrics[threshold_name] = metrics_dict
                        logger.debug(f"Stored metrics for {threshold_name}: {metrics_dict}")
                        
                    except Exception as metrics_err:
                        logger.error(f"Error calculating metrics for FixedCutoff: {metrics_err}")
                        logger.exception(f"FixedCutoff metrics error traceback:")
                        all_metrics[threshold_name] = {}
                    
                    continue  # Skip the rest of the processing for FixedCutoff
                
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
                st.error(f"Error applying {threshold_name}: {e}")
                continue
                
        # Store results - now including the detector object
        st.session_state.results[detector_key] = {
            'detector_name': detector_config['name'],
            'detector': detector,  # Store the detector object for visualizations
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
        st.error(f"Error executing detector: {e}")
        st.error(traceback.format_exc())
        return False

def display_detector_results(tab_index):
    """Displays the detection results for a specific tab."""
    tab = st.session_state.detector_tabs[tab_index]
    detector_key = f"detector_{tab_index + 1}"
    
    if detector_key not in st.session_state.results:
        st.warning("This detector has not been executed yet.")
        return
    
    results = st.session_state.results[detector_key]
    detector_name = results['detector_name']
    
    try:
        if st.session_state.uploaded_data_valid:
            x, y = st.session_state.uploaded_data
        else:
            if 'selected_dataset_name' not in st.session_state or not st.session_state.selected_dataset_name:
                st.error("No dataset selected.")
                return
            x, y = load_dataset(st.session_state.selected_dataset_name)
        
        time_steps = format_time_steps(None, x.shape[0])
        anomaly_scores = results['anomaly_scores']
        
        # Get processed_x from export_data if available
        processed_x = st.session_state.export_data.get('processed_x', x)
        
        # Individual Detector Information
        st.subheader(f"Detector Information: {detector_name}")
        st.write(f"Fit time: {results['fit_time']:.4f} seconds")
        st.write(f"Predict time: {results['predict_time']:.4f} seconds")
        
        # Create a metrics section for this detector
        st.subheader("Performance Metrics")
        
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
                            'Value': value
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
                                    'Value': score
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
            st.subheader("Metrics Overview")
            
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
                        score = matching_rows[0]['Value']
                        # Format score as percentage if it's a float
                        display_value = f"{score:.2%}" if isinstance(score, float) else score
                        cols[i % len(cols)].metric(
                            label=metric_name,
                            value=display_value
                        )
        else:
            st.warning("No metrics calculated for this detector.")
        
        # Standard Visualizations section
        st.subheader("Standard Visualizations")
        
        # Always show both visualizations without requiring selection
        st.write("**Time Series with Anomalies**")
        try:
            # Get thresholded predictions if available
            y_pred = None
            
            # Try to get thresholded predictions if available
            if results['thresholded_predictions']:
                # Get the first available threshold's predictions
                threshold_name = list(results['thresholded_predictions'].keys())[0]
                y_pred = results['thresholded_predictions'][threshold_name]
            
            # Use Plotly for interactive visualizations
            fig = generate_plotly_visualization("Time Series with Anomalies", x, y, anomaly_scores, y_pred, time_steps)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate Time Series with Anomalies visualization.")
        except Exception as viz_error:
            st.error(f"Error generating Time Series with Anomalies visualization: {viz_error}")
            st.error(traceback.format_exc())
        
        st.write("**Anomaly Scores**")
        try:
            # Use Plotly for interactive visualizations
            fig = generate_plotly_visualization("Anomaly Scores", x, y, anomaly_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate Anomaly Scores visualization.")
        except Exception as viz_error:
            st.error(f"Error generating Anomaly Scores visualization: {viz_error}")
            st.error(traceback.format_exc())
        
        # Get detector from results data if available
        detector = results.get('detector', None)
        
        # Add Detector-specific visualizations section
        detector_figures = {}
        if detector:
            detector_figures = generate_detector_specific_visualization(
                detector_name, detector, x, processed_x, time_steps
            )
        
        # Add custom visualizations
        custom_figures = get_custom_visualizations(
            detector_name, detector, x, processed_x, time_steps
        )
        
        # Combine detector-specific and custom visualizations
        all_special_figures = {**detector_figures, **custom_figures}
        
        # Display detector-specific and custom visualizations if available
        if all_special_figures:
            st.subheader(f"{detector_name} Specific Visualizations")
            
            # Create tabs for each visualization
            special_viz_names = list(all_special_figures.keys())
            special_viz_tabs = st.tabs(special_viz_names)
            
            for i, viz_name in enumerate(special_viz_names):
                with special_viz_tabs[i]:
                    fig = all_special_figures[viz_name]
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Could not generate {viz_name}.")
        
        # Export options
        st.subheader("Export")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to Excel", key=f"export_excel_{tab_index}"):
                try:
                    excel_file = export_to_excel()
                    st.download_button(
                        label="Download Excel file",
                        data=excel_file,
                        file_name=f"anomaly_detection_results_{detector_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_excel_{tab_index}"
                    )
                except Exception as export_error:
                    st.error(f"Error exporting to Excel: {export_error}")
                    st.error(traceback.format_exc())
        
        with col2:
            if st.button("Export to CSV", key=f"export_csv_{tab_index}"):
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
                        label="Download CSV file",
                        data=csv_data,
                        file_name=f"anomaly_detection_results_{detector_name}.csv",
                        mime="text/csv",
                        key=f"download_csv_{tab_index}"
                    )
                except Exception as export_error:
                    st.error(f"Error exporting to CSV: {export_error}")
                    st.error(traceback.format_exc())
        
    except Exception as e:
        st.error(f"Error displaying results: {e}")
        st.error(traceback.format_exc())

def run_pipeline():
    """Executes the anomaly detection pipeline for all active detectors."""
    success = False
    for i, _ in enumerate(st.session_state.detector_tabs):
        if run_detector(i):
            success = True
    
    return success

def generate_plotly_visualization(viz: str, x: np.ndarray, y: np.ndarray, anomaly_scores: np.ndarray,
                              thresholded_predictions: np.ndarray = None, time_steps: np.ndarray = None):
    """Generates interactive Plotly visualizations for anomaly detection."""
    if time_steps is None:
        time_steps = format_time_steps(None, x.shape[0])
        
    # Convert x to the right format for plotting
    if is_univariate(x):
        plot_x = x.flatten()
    else:
        # For multivariate data, use the first dimension
        plot_x = x[:, 0].flatten()
    
    # Basic figure settings
    fig = None
    
    if viz == "Anomaly Scores":
        # Create a plot with two subplots (stacked vertically)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1,
                            subplot_titles=("Time Series", "Anomaly Scores"))
        
        # Add the original time series
        fig.add_trace(
            go.Scatter(x=time_steps, y=plot_x, mode='lines', name='Time Series'),
            row=1, col=1
        )
        
        # Mark ground truth anomalies on the time series
        anomaly_indices = np.where(y == 1)[0]
        if len(anomaly_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_steps[anomaly_indices], 
                    y=plot_x[anomaly_indices],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='True Anomalies'
                ),
                row=1, col=1
            )
        
        # Add anomaly scores
        fig.add_trace(
            go.Scatter(x=time_steps, y=anomaly_scores, mode='lines', 
                      name='Anomaly Score', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600, 
            width=900,
            title_text="Time Series and Anomaly Scores",
            hovermode="x unified"
        )
        
    elif viz == "Time Series with Anomalies":
        # Create a plot comparing detected vs true anomalies
        fig = make_subplots(rows=1, cols=1)
        
        # Add the original time series
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=plot_x,
                mode='lines',
                name='Time Series'
            )
        )
        
        # Add true anomalies
        true_anomaly_indices = np.where(y == 1)[0]
        if len(true_anomaly_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_steps[true_anomaly_indices],
                    y=plot_x[true_anomaly_indices],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='x'),
                    name='True Anomalies'
                )
            )
        
        # Add predicted anomalies if available
        if thresholded_predictions is not None:
            pred_anomaly_indices = np.where(thresholded_predictions == 1)[0]
            if len(pred_anomaly_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time_steps[pred_anomaly_indices],
                        y=plot_x[pred_anomaly_indices],
                        mode='markers',
                        marker=dict(color='blue', size=8, symbol='circle'),
                        name='Detected Anomalies'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="Time Series with True and Detected Anomalies",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            width=900,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    
    return fig

def export_to_excel():
    """Generates an Excel file with the collected data."""
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

# Add a utility function to replace format_time_steps if it's not available
def format_time_steps(t=None, length=None):
    """Format time steps for plotting.
    
    This is a utility function to create time steps for plotting,
    without depending on the original dtaianomaly visualization module.
    
    Args:
        t: Time steps to format. If None, create a range from 0 to length-1.
        length: Length of time series. Only used if t is None.
        
    Returns:
        Formatted time steps.
    """
    if t is None:
        if length is None:
            raise ValueError("Either t or length must be provided.")
        return np.arange(length)
    return t

# Add a function to generate detector-specific visualizations
def generate_detector_specific_visualization(detector_name, detector, x, processed_x, time_steps):
    """
    Generate detector-specific visualizations based on the detector type.
    
    Args:
        detector_name (str): Name of the detector
        detector (object): The fitted detector object
        x (np.ndarray): Original input data
        processed_x (np.ndarray): Processed input data used for model fitting
        time_steps (np.ndarray): Time steps for plotting
        
    Returns:
        dict: A dictionary of Plotly figures, keyed by visualization name
    """
    if not detector or not hasattr(detector, '__class__'):
        return {}
    
    figures = {}
    
    # Format x for plotting
    if is_univariate(x):
        plot_x = x.flatten()
    else:
        # For multivariate data, use the first dimension
        plot_x = x[:, 0].flatten() if x.shape[1] > 0 else x
    
    # Format processed_x for plotting
    if is_univariate(processed_x):
        plot_processed_x = processed_x.flatten()
    else:
        # For multivariate data, use the first dimension
        plot_processed_x = processed_x[:, 0].flatten() if processed_x.shape[1] > 0 else processed_x
    
    try:
        # KMeans-specific visualization
        if "KMeans" in detector_name:
            # Check if the detector has necessary attributes for visualization
            has_labels = hasattr(detector, 'labels_') or hasattr(detector, '_labels')
            has_centers = hasattr(detector, 'cluster_centers_') or hasattr(detector, '_cluster_centers')
            
            # Get labels from detector
            labels = None
            if hasattr(detector, 'labels_'):
                labels = detector.labels_
            elif hasattr(detector, '_labels'):
                labels = detector._labels
            
            # Create a simple scatter plot of the time series colored by cluster
            if has_labels and labels is not None:
                # Create figure for cluster visualization
                fig = go.Figure()
                
                # Add time series line for reference
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=plot_processed_x,
                        mode='lines',
                        name='Time Series',
                        line=dict(color='lightgrey', width=1)
                    )
                )
                
                # Color points by cluster
                for cluster_id in np.unique(labels):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    if len(cluster_indices) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_steps[cluster_indices],
                                y=plot_processed_x[cluster_indices],
                                mode='markers',
                                name=f'Cluster {cluster_id}',
                                marker=dict(size=8)
                            )
                        )
                
                # Update layout
                fig.update_layout(
                    title="K-Means Cluster Assignments",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode="closest",
                    height=500,
                    width=900
                )
                
                figures["Cluster Assignments"] = fig
            
            # Try to create a distance-to-centroid plot if distance information is available
            # Look for various attribute names that might contain distance information
            distance_attrs = [
                '_distance_to_closest_centroid', 
                'distances_', 
                '_distances',
                'decision_scores_'
            ]
            
            distances = None
            for attr in distance_attrs:
                if hasattr(detector, attr):
                    distances = getattr(detector, attr)
                    if distances is not None and len(distances) == len(time_steps):
                        break
            
            # If distances are available, create distance plot
            if distances is not None:
                fig_dist = go.Figure()
                
                # Add distance trace
                fig_dist.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=distances,
                        mode='lines',
                        name='Distance to Nearest Centroid',
                        line=dict(color='orange')
                    )
                )
                
                # Add threshold line if a threshold attribute exists
                threshold_attrs = ['threshold_', '_threshold']
                for attr in threshold_attrs:
                    if hasattr(detector, attr):
                        threshold = getattr(detector, attr)
                        if threshold is not None:
                            fig_dist.add_hline(
                                y=threshold,
                                line=dict(color='red', dash='dash'),
                                annotation_text="Threshold"
                            )
                
                # Update layout
                fig_dist.update_layout(
                    title="Distance to Nearest Centroid",
                    xaxis_title="Time",
                    yaxis_title="Distance",
                    hovermode="closest",
                    height=500,
                    width=900
                )
                
                figures["Centroid Distances"] = fig_dist
            
            # If we have centers and at least 2D data, create a scatter plot of centers and points
            centers = None
            if has_centers:
                if hasattr(detector, 'cluster_centers_'):
                    centers = detector.cluster_centers_
                elif hasattr(detector, '_cluster_centers'):
                    centers = detector._cluster_centers
            
            if centers is not None and not is_univariate(processed_x) and processed_x.shape[1] >= 2:
                fig_centers = go.Figure()
                
                # Add data points
                fig_centers.add_trace(
                    go.Scatter(
                        x=processed_x[:, 0],
                        y=processed_x[:, 1],
                        mode='markers',
                        name='Data Points',
                        marker=dict(
                            size=8,
                            color=labels if labels is not None else 'blue',
                            colorscale='Viridis' if labels is not None else None,
                            showscale=labels is not None
                        )
                    )
                )
                
                # Add cluster centers
                fig_centers.add_trace(
                    go.Scatter(
                        x=centers[:, 0],
                        y=centers[:, 1],
                        mode='markers',
                        name='Cluster Centers',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='star'
                        )
                    )
                )
                
                fig_centers.update_layout(
                    title="Cluster Centers and Data Points",
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    hovermode="closest",
                    height=600,
                    width=900
                )
                
                figures["Cluster Centers"] = fig_centers
        
        # IsolationForest-specific visualization
        elif "IsolationForest" in detector_name:
            # Try to get decision scores
            scores = None
            score_attrs = ['decision_scores_', 'decision_function_', 'score_samples_', 'anomaly_scores_']
            
            for attr in score_attrs:
                if hasattr(detector, attr):
                    scores_attr = getattr(detector, attr)
                    if callable(scores_attr):
                        try:
                            scores = scores_attr(processed_x)
                            break
                        except:
                            continue
                    elif isinstance(scores_attr, np.ndarray):
                        scores = scores_attr
                        break
            
            # If no scores found directly, try to compute them
            if scores is None:
                try:
                    if hasattr(detector, 'decision_function') and callable(getattr(detector, 'decision_function')):
                        scores = detector.decision_function(processed_x)
                    elif hasattr(detector, 'predict_proba') and callable(getattr(detector, 'predict_proba')):
                        scores = detector.predict_proba(processed_x)
                    elif hasattr(detector, 'score_samples') and callable(getattr(detector, 'score_samples')):
                        scores = detector.score_samples(processed_x)
                except:
                    pass
            
            # If we have scores, create a histogram
            if scores is not None:
                fig = go.Figure()
                
                # Add histogram of scores
                fig.add_trace(
                    go.Histogram(
                        x=scores,
                        name="Score Distribution",
                        opacity=0.7,
                        marker_color='blue',
                        nbinsx=30
                    )
                )
                
                # Add threshold line if available
                threshold_attrs = ['threshold_', '_threshold']
                for attr in threshold_attrs:
                    if hasattr(detector, attr):
                        threshold = getattr(detector, attr)
                        if threshold is not None:
                            fig.add_vline(
                                x=threshold,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Threshold",
                                annotation_position="top right"
                            )
                
                fig.update_layout(
                    title="Isolation Forest Score Distribution",
                    xaxis_title="Anomaly Score",
                    yaxis_title="Count",
                    hovermode="closest",
                    height=500,
                    width=900
                )
                
                figures["Score Distribution"] = fig
                
                # Create a time series with anomaly scores
                fig_scores = go.Figure()
                
                # Add time series
                fig_scores.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=plot_x,
                        mode='lines',
                        name='Time Series',
                        line=dict(color='lightgrey')
                    )
                )
                
                # Add anomaly scores as scatter points with color gradient
                fig_scores.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=plot_x,
                        mode='markers',
                        name='Anomaly Score',
                        marker=dict(
                            size=8,
                            color=scores,
                            colorscale='Viridis',
                            colorbar=dict(title="Anomaly Score"),
                            showscale=True
                        ),
                        text=[f"Time: {t}, Score: {s:.3f}" for t, s in zip(time_steps, scores)]
                    )
                )
                
                fig_scores.update_layout(
                    title="Time Series with Anomaly Scores",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode="closest", 
                    height=500,
                    width=900
                )
                
                figures["Time Series with Scores"] = fig_scores
        
        # MatrixProfile-specific visualization
        elif "MatrixProfile" in detector_name:
            # Look for matrix profile values
            matrix_profile = None
            profile_attrs = ['matrix_profile_', '_matrix_profile', 'profile_']
            
            for attr in profile_attrs:
                if hasattr(detector, attr):
                    profile = getattr(detector, attr)
                    if profile is not None and isinstance(profile, np.ndarray):
                        matrix_profile = profile
                        break
            
            # Create visualization for matrix profile
            if matrix_profile is not None:
                fig = go.Figure()
                
                # Add matrix profile values
                fig.add_trace(
                    go.Scatter(
                        x=time_steps[:len(matrix_profile)],
                        y=matrix_profile,
                        mode='lines',
                        name='Matrix Profile',
                        line=dict(color='blue')
                    )
                )
                
                # Add threshold if available
                threshold_attrs = ['threshold_', '_threshold']
                for attr in threshold_attrs:
                    if hasattr(detector, attr):
                        threshold = getattr(detector, attr)
                        if threshold is not None:
                            fig.add_hline(
                                y=threshold,
                                line=dict(color='red', dash='dash'),
                                annotation_text="Threshold"
                            )
                
                fig.update_layout(
                    title="Matrix Profile",
                    xaxis_title="Time",
                    yaxis_title="Distance",
                    hovermode="closest",
                    height=500,
                    width=900
                )
                
                figures["Matrix Profile"] = fig
                
                # Create a second visualization showing the original time series with matrix profile overlay
                fig_overlay = go.Figure()
                
                # Create a subplot with two y-axes
                fig_overlay = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add original time series on primary y-axis
                fig_overlay.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=plot_x,
                        mode='lines',
                        name='Time Series',
                        line=dict(color='blue')
                    ),
                    secondary_y=False
                )
                
                # Add matrix profile on secondary y-axis
                fig_overlay.add_trace(
                    go.Scatter(
                        x=time_steps[:len(matrix_profile)],
                        y=matrix_profile,
                        mode='lines',
                        name='Matrix Profile',
                        line=dict(color='red')
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig_overlay.update_layout(
                    title="Time Series and Matrix Profile",
                    hovermode="x unified",
                    height=500,
                    width=900
                )
                
                # Update axis labels
                fig_overlay.update_xaxes(title_text="Time")
                fig_overlay.update_yaxes(title_text="Value", secondary_y=False)
                fig_overlay.update_yaxes(title_text="Matrix Profile", secondary_y=True)
                
                figures["Time Series with Matrix Profile"] = fig_overlay
        
        # Generic 3D visualization for multivariate data (if 3+ dimensions)
        if not is_univariate(processed_x) and processed_x.shape[1] >= 3:
            # Create 3D scatter plot with first 3 dimensions
            fig_3d = go.Figure(data=[
                go.Scatter3d(
                    x=processed_x[:, 0],
                    y=processed_x[:, 1],
                    z=processed_x[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=0.7
                    )
                )
            ])
            
            fig_3d.update_layout(
                title="3D Data Visualization",
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    zaxis_title="Dimension 3"
                ),
                height=700,
                width=900
            )
            
            figures["3D Visualization"] = fig_3d
    
    except Exception as e:
        logger.error(f"Error generating detector-specific visualization: {e}")
        logger.exception(traceback.format_exc())
    
    return figures

# Add custom visualization registry
if 'custom_visualizations' not in st.session_state:
    st.session_state.custom_visualizations = {}

def register_custom_visualization(detector_name, visualization_name, visualization_func):
    """
    Register a custom visualization function for a specific detector.
    
    Args:
        detector_name (str): Name of the detector
        visualization_name (str): Name of the visualization
        visualization_func (callable): Function that returns a Plotly figure
            The function should accept (detector, x, processed_x, time_steps) as arguments
    """
    if detector_name not in st.session_state.custom_visualizations:
        st.session_state.custom_visualizations[detector_name] = {}
    
    st.session_state.custom_visualizations[detector_name][visualization_name] = visualization_func
    
def get_custom_visualizations(detector_name, detector, x, processed_x, time_steps):
    """
    Get custom visualizations for a specific detector.
    
    Args:
        detector_name (str): Name of the detector
        detector (object): The fitted detector object
        x (np.ndarray): Original input data
        processed_x (np.ndarray): Processed input data
        time_steps (np.ndarray): Time steps for plotting
        
    Returns:
        dict: A dictionary of Plotly figures, keyed by visualization name
    """
    figures = {}
    
    # Check if custom_visualizations exists in session_state
    custom_viz = getattr(st.session_state, 'custom_visualizations', {})
    
    if detector_name in custom_viz:
        for viz_name, viz_func in custom_viz[detector_name].items():
            try:
                fig = viz_func(detector, x, processed_x, time_steps)
                figures[viz_name] = fig
            except Exception as e:
                logger.error(f"Error generating custom visualization {viz_name}: {e}")
                logger.exception(traceback.format_exc())
    
    return figures

def get_threshold_documentation(threshold_name):
    """
    Extract documentation from threshold class.
    
    Args:
        threshold_name: Name of the threshold class
        
    Returns:
        str: The main description part of the documentation string for the threshold
    """
    try:
        if hasattr(thresholding, threshold_name):
            threshold_class = getattr(thresholding, threshold_name)
            doc = inspect.getdoc(threshold_class)
            if doc:
                # Extract just the main description (everything before Parameters or ---)
                main_description = doc.split("Parameters")[0].split("---")[0].strip()
                return main_description
        
        return f"No documentation available for {threshold_name}."
    except Exception as e:
        logger.error(f"Error getting documentation for {threshold_name}: {e}")
        return f"Error retrieving documentation for {threshold_name}."

def get_threshold_parameter_documentation(threshold_name, param_name):
    """
    Extract documentation for a specific parameter of a threshold class.
    
    Args:
        threshold_name: Name of the threshold class
        param_name: Name of the parameter
        
    Returns:
        str: Documentation for the parameter, or None if not found
    """
    try:
        if hasattr(thresholding, threshold_name):
            threshold_class = getattr(thresholding, threshold_name)
            doc = inspect.getdoc(threshold_class)
            
            if not doc:
                return None
                
            # Look for parameter documentation in the docstring
            # Pattern is similar to detector parameter documentation
            lines = doc.split('\n')
            param_found = False
            param_doc = []
            
            # First try Parameters section format
            if "Parameters" in doc:
                params_section = doc.split("Parameters")[1].split("---")[0].split("Attributes")[0]
                param_lines = params_section.strip().split('\n')
                for i, line in enumerate(param_lines):
                    if param_name in line and ":" in line:
                        param_found = True
                        # Extract description from subsequent indented lines
                        j = i + 1
                        while j < len(param_lines) and (param_lines[j].startswith(' ') or param_lines[j].startswith('\t')):
                            param_doc.append(param_lines[j].strip())
                            j += 1
                        break
            
            # If not found, try another common format
            if not param_found:
                for i, line in enumerate(lines):
                    if line.strip().startswith(param_name + ':') or line.strip().startswith(param_name + ' :'):
                        param_found = True
                        param_doc.append(line.split(':', 1)[1].strip())
                        # Check for multi-line descriptions (indented lines following parameter)
                        j = i + 1
                        while j < len(lines) and (lines[j].startswith(' ') or lines[j].startswith('\t')):
                            param_doc.append(lines[j].strip())
                            j += 1
                        break
            
            if param_found and param_doc:
                return ' '.join(param_doc)
        
        return None
    except Exception as e:
        logger.error(f"Error getting parameter documentation: {e}")
        return None

# Add a specific function to create default parameters for FixedCutoff
def init_fixed_cutoff_params():
    """Initialize default parameters for FixedCutoff threshold."""
    return {"cutoff": 0.5}
    
# --- Main application execution ---
def main():
    """Main function to run the Streamlit app."""
    # Check for custom components in session state
    import os
    import streamlit as st
    
    # Initialize custom_components in session state if not present
    if 'custom_components' not in st.session_state:
        logger.info("No custom components found in session state")
        st.session_state.custom_components = {}
    else:
        logger.info(f"Found custom components in session state: {list(st.session_state.custom_components.keys())}")
        # Check if we have custom detectors
        if 'detectors' in st.session_state.custom_components:
            custom_detectors = list(st.session_state.custom_components['detectors'].keys())
            logger.info(f"Custom detectors registered: {custom_detectors}")
    
    # App title and description
    st.title("dtaianomaly Demonstrator")
    st.markdown(
        """
        A no-code demonstrator for the **dtaianomaly** library, allowing you to interactively explore 
        anomaly detection techniques for time series data. This tool supports both qualitative evaluation 
        (visualizations) and quantitative evaluation (benchmarking and performance conclusions) with 
        comparison of multiple detectors.
        """
    )
    
    # Hidden experience level selection - only setting the session state variable
    # We hide the UI-element maar behouden de functionaliteit
    if 'experience_level' not in st.session_state:
        st.session_state.experience_level = "Expert"

    # Configure the sidebar with datasets, metrics, thresholds, and visualizations
    configure_sidebar()
    
    # Main content area
    # Create detector tabs section
    st.subheader("Anomaly Detectors")
    
    # Add/Remove detector tabs
    tab_col1, tab_col2 = st.columns([9, 1])
    with tab_col2:
        if st.button("➕", help="Add detector"):
            add_detector_tab()
            st.rerun()
    
    # Only proceed if there are detector tabs
    if not st.session_state.detector_tabs:
        st.warning("No detectors available. Add a detector using the + button.")
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
                if st.button("❌", key=f"remove_tab_{i}", help="Remove this detector"):
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
                if st.button("Run", key=f"run_detector_{i}"):
                    if st.session_state.uploaded_data_valid or 'selected_dataset_name' in st.session_state:
                        with st.spinner(f"Running Detector {i+1}..."):
                            if run_detector(i):
                                st.success(f"Detector {i+1} executed successfully!")
                                st.session_state.detector_results.add(f"detector_{i+1}")
                                st.rerun()
                    else:
                        st.error("Please select a valid dataset or upload a correct file before running the detector.")
            
            # Display detector results if available
            detector_result_container = st.container()
            with detector_result_container:
                detector_key = f"detector_{i+1}"
                if detector_key in st.session_state.results:
                    display_detector_results(i)
                else:
                    st.info(f"Detector {i+1} has not been run yet. Click 'Run' to start the detector.")
    
    # Global run button
    global_run_container = st.container()
    with global_run_container:
        if st.button("Run All Detectors", key="run_all_detectors"):
            if st.session_state.uploaded_data_valid or 'selected_dataset_name' in st.session_state:
                with st.spinner("Running all detectors..."):
                    success = run_pipeline()
                    if success:
                        # Add all detectors to the results set
                        for i in range(len(st.session_state.detector_tabs)):
                            st.session_state.detector_results.add(f"detector_{i+1}")
                        st.success("All detectors executed successfully!")
                        st.rerun()
            else:
                st.error("Please select a valid dataset or upload a correct file before running the pipeline.")
    
    # GLOBAL COMPARISON SECTION - Outside of the tabs, always visible when there are multiple detectors
    if len([k for k in st.session_state.results.keys() if k.startswith('detector_')]) >= 2:
        st.markdown("---")  # Divider
        comparison_container = st.container()
        with comparison_container:
            st.header("Comparative Overview of Detectors")
            
            # Create comparison tabs - remove Performance Analysis
            compare_tabs = st.tabs(["Metrics Comparison", "Time Comparison", "ROC Curve", "Anomaly Scores Overlay"])
            
            # Metrics Comparison tab
            with compare_tabs[0]:
                st.write("**Quantitative Evaluation: Metrics Comparison**")
                
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
                    
                    # Create interactive Plotly bar chart for metrics comparison
                    st.write("**Visual Comparison of Metrics**")
                    metrics_df = df[df['Metric'].isin(list(metric_names))]
                    
                    try:
                        melted_df = metrics_df.melt(id_vars='Metric', var_name='Detector', value_name='Score')
                        fig = px.bar(
                            melted_df,
                            x='Metric',
                            y='Score',
                            color='Detector',
                            barmode='group',
                            title='Comparison of Detector Performance',
                            labels={'Metric': 'Evaluation Metrics', 'Score': 'Score'},
                            hover_data=['Metric', 'Detector', 'Score'],
                            color_discrete_sequence=px.colors.qualitative.G10
                        )
                        fig.update_layout(
                            xaxis_title='Evaluation Metrics',
                            yaxis_title='Score',
                            legend_title='Anomaly Detectors',
                            font=dict(size=12),
                            title_font=dict(size=14),
                            hovermode='closest'
                        )
                        # Make the plot interactive with zoom capabilities
                        fig.update_layout(
                            xaxis=dict(rangeslider=dict(visible=False)),
                            clickmode='event+select'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as chart_error:
                        st.error(f"Error generating comparison chart: {chart_error}")
                else:
                    st.warning("No metrics available for comparison.")
            
            # Time Comparison tab
            with compare_tabs[1]:
                st.write("**Comparison of Processing Times**")
                
                try:
                    # Create comparison bar chart for timing with Plotly
                    time_data = {'Detector': [], 'Type': [], 'Time (s)': []}
                    
                    for detector_key, detector_result in st.session_state.results.items():
                        if not detector_key.startswith('detector_'):
                            continue
                            
                        detector_label = f"Detector {detector_key[-1]} ({detector_result['detector_name']})"
                        
                        time_data['Detector'].append(detector_label)
                        time_data['Type'].append('Training Time')
                        time_data['Time (s)'].append(detector_result['fit_time'])
                        
                        time_data['Detector'].append(detector_label)
                        time_data['Type'].append('Prediction Time')
                        time_data['Time (s)'].append(detector_result['predict_time'])
                    
                    time_df = pd.DataFrame(time_data)
                    time_fig = px.bar(
                        time_df,
                        x='Detector',
                        y='Time (s)',
                        color='Type',
                        barmode='group',
                        title='Comparison of Processing Times',
                        labels={'Detector': 'Anomaly Detectors', 'Time (s)': 'Time (seconds)', 'Type': 'Time Type'},
                        hover_data=['Detector', 'Type', 'Time (s)'],
                        color_discrete_sequence=['#00CC96', '#EF553B']
                    )
                    time_fig.update_layout(
                        xaxis_title='Anomaly Detectors',
                        yaxis_title='Time (seconds)',
                        legend_title='Time Type',
                        font=dict(size=12),
                        title_font=dict(size=14),
                        hovermode='closest'
                    )
                    # Add hover tooltips for better info
                    time_fig.update_traces(hovertemplate='<b>%{x}</b><br>%{y:.4f} seconds<extra></extra>')
                    
                    st.plotly_chart(time_fig, use_container_width=True)
                except Exception as time_error:
                    st.error(f"Error generating time comparison: {time_error}")
            
            # ROC Curve tab
            with compare_tabs[2]:
                st.write("**ROC Curve Comparison**")
                try:
                    # Get dataset 
                    if st.session_state.uploaded_data_valid:
                        _, y_true = st.session_state.uploaded_data
                    elif 'selected_dataset_name' in st.session_state:
                        _, y_true = load_dataset(st.session_state.selected_dataset_name)
                    else:
                        y_true = None
                        
                    if y_true is not None:
                        # Create ROC curve with sklearn
                        from sklearn.metrics import roc_curve, auc
                        
                        # Plot for each detector
                        fig = go.Figure()
                        for detector_key, detector_result in st.session_state.results.items():
                            if not detector_key.startswith('detector_'):
                                continue
                                
                            detector_name = detector_result['detector_name']
                            detector_label = f"Detector {detector_key[-1]} ({detector_name})"
                            anomaly_scores = detector_result['anomaly_scores']
                            
                            # Calculate ROC curve
                            fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
                            roc_auc = auc(fpr, tpr)
                            
                            # Add trace for this detector
                            fig.add_trace(
                                go.Scatter(
                                    x=fpr, 
                                    y=tpr,
                                    mode='lines',
                                    name=f'{detector_label} (AUC = {roc_auc:.3f})'
                                )
                            )
                        
                        # Add diagonal reference line
                        fig.add_trace(
                            go.Scatter(
                                x=[0, 1], 
                                y=[0, 1],
                                mode='lines',
                                name='Random Classifier',
                                line=dict(color='gray', dash='dash')
                            )
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title='ROC Curve Comparison',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            legend=dict(
                                x=1,
                                y=0,
                                xanchor='right',
                                yanchor='bottom'
                            ),
                            width=800,
                            height=600,
                            hovermode='closest'
                        )
                        
                        # Display the ROC curve
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Cannot generate ROC curve: No dataset available.")
                except Exception as roc_error:
                    st.error(f"Error generating ROC curve: {roc_error}")
                    st.error(traceback.format_exc())
            
            # Anomaly Scores Overlay tab
            with compare_tabs[3]:
                st.write("**Overlay of Anomaly Scores from All Detectors**")
                
                try:
                    # Get dataset for x-axis time steps
                    if st.session_state.uploaded_data_valid:
                        x, y_true = st.session_state.uploaded_data
                    elif 'selected_dataset_name' in st.session_state:
                        x, y_true = load_dataset(st.session_state.selected_dataset_name)
                    else:
                        x, y_true = None, None
                        
                    if x is not None:
                        # Create a subplot with 2 rows: time series on top, anomaly scores on bottom
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.1,
                                        subplot_titles=("Original Time Series with Anomalies", "Anomaly Scores Comparison"))
                        
                        # Get time steps
                        time_steps = format_time_steps(None, x.shape[0])
                        
                        # Convert to 1D for plotting if needed
                        if is_univariate(x):
                            plot_x = x.flatten()
                        else:
                            plot_x = x[:, 0].flatten()
                        
                        # Add original time series trace
                        fig.add_trace(
                            go.Scatter(
                                x=time_steps,
                                y=plot_x,
                                mode='lines',
                                name='Time Series',
                                line=dict(color='lightgrey', width=1.5)
                            ),
                            row=1, col=1
                        )
                        
                        # Add true anomalies on time series
                        anomaly_indices = np.where(y_true == 1)[0]
                        if len(anomaly_indices) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=time_steps[anomaly_indices],
                                    y=plot_x[anomaly_indices],
                                    mode='markers',
                                    marker=dict(color='red', size=8, symbol='x'),
                                    name='True Anomalies'
                                ),
                                row=1, col=1
                            )
                        
                        # Color palette for different detectors
                        colors = px.colors.qualitative.Plotly
                        
                        # Add anomaly score traces for each detector
                        for i, (detector_key, detector_result) in enumerate(st.session_state.results.items()):
                            if not detector_key.startswith('detector_'):
                                continue
                                
                            detector_name = detector_result['detector_name']
                            detector_label = f"Detector {detector_key[-1]} ({detector_name})"
                            anomaly_scores = detector_result['anomaly_scores']
                            
                            # Add detector's anomaly scores trace
                            fig.add_trace(
                                go.Scatter(
                                    x=time_steps, 
                                    y=anomaly_scores,
                                    mode='lines',
                                    name=detector_label,
                                    line=dict(color=colors[i % len(colors)], width=2)
                                ),
                                row=2, col=1
                            )
                        
                        # Update layout
                        fig.update_layout(
                            height=800,
                            width=900,
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        # Update y-axis labels
                        fig.update_yaxes(title_text="Value", row=1, col=1)
                        fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
                        
                        # Update x-axis label (only for bottom subplot)
                        fig.update_xaxes(title_text="Time Steps", row=2, col=1)
                        
                        # Display the figure
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Cannot generate anomaly score overlay: No dataset available.")
                except Exception as overlay_error:
                    st.error(f"Error generating anomaly scores overlay: {overlay_error}")
                    st.error(traceback.format_exc())


if __name__ == "__main__":
    # Check if the Streamlit runtime already exists
    if runtime.exists():
        main()
    else:
        # If it doesn't exist, run the Streamlit CLI with this file as argument
        script_path = os.path.abspath(__file__)
        sys.argv = ["streamlit", "run", script_path]
        sys.exit(stcli.main())
 
