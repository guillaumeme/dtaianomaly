"""
Demonstrator module for the dtaianomaly package.

This module provides a Streamlit-based interface for exploring and visualizing
anomaly detection algorithms on time series data.
"""

import os
import sys
import pathlib
import logging

from dtaianomaly.demonstrator.demonstrator import main, register_custom_visualization

def run_demonstrator():
    """Run the Streamlit-based demonstrator application."""
    script_path = pathlib.Path(__file__).parent / "demonstrator.py"
    
    try:
        from streamlit.web import cli as stcli
        from streamlit import runtime
        
        if runtime.exists():
            # Import and run the main function directly
            main()
        else:
            # Use the CLI to run the script
            sys.argv = ["streamlit", "run", str(script_path)]
            sys.exit(stcli.main())
    except ImportError as e:
        # Fallback to os.system if streamlit module can't be properly imported
        print(f"Warning: Could not import Streamlit properly: {e}")
        print(f"Falling back to os.system command...")
        try:
            os.system(f"streamlit run {script_path}")
        except Exception as os_error:
            print(f"Error using os.system fallback: {os_error}")
            print("Please ensure Streamlit is installed: pip install streamlit")

def run_with_detector(detector_class, detector_name=None):
    """
    Run the demonstrator with a custom detector.
    
    This is the simplest way to run the demonstrator with your own detector.
    Just pass your detector class and optionally a name for it.
    
    Args:
        detector_class: Your custom anomaly detector class
        detector_name: Name for your detector (defaults to the class name)
    
    Example:
        from dtaianomaly.demonstrator import run_with_detector
        from your_module import YourDetector
        
        run_with_detector(YourDetector)
    """
    # If no name provided, use the class name
    if detector_name is None:
        detector_name = detector_class.__name__
    
    # Import needed modules
    import streamlit as st
    from streamlit.web import cli as stcli
    from streamlit import runtime
    
    # Register the custom detector
    custom_components = {
        'detectors': {
            detector_name: detector_class
        }
    }
    
    # Set the custom components in session state
    st.session_state.custom_components = custom_components
    
    # Run the demonstrator
    script_path = pathlib.Path(__file__).parent / "demonstrator.py"
    
    if not runtime.exists():
        # Run with streamlit CLI
        sys.argv = ["streamlit", "run", str(script_path)]
        print(f"Running demonstrator with custom detector: {detector_name}")
        sys.exit(stcli.main())
    else:
        # Run main directly
        print(f"Running demonstrator with custom detector: {detector_name}")
        from dtaianomaly.demonstrator.demonstrator import main
        main()

def run_custom_detector_demo():
    """
    Run the demonstrator with the custom NbSigmaAnomalyDetector.
    
    This function imports the custom detector from custom_detector_demo.py
    and runs the demonstrator with it as a custom component.
    """
    # Import the custom detector and components
    from dtaianomaly.demonstrator.custom_detector_demo import NbSigmaAnomalyDetector, custom_components
    import streamlit as st
    
    # Set custom components in session state
    st.session_state.custom_components = custom_components
    
    # Run the demonstrator
    from streamlit.web import cli as stcli
    from streamlit import runtime
    import sys
    import os
    
    # If we're in a script, run via CLI
    if not runtime.exists():
        script_path = os.path.abspath(__file__).replace('__init__.py', 'demonstrator.py')
        sys.argv = ["streamlit", "run", script_path]
        sys.exit(stcli.main())
    # If runtime exists, run main directly
    else:
        from dtaianomaly.demonstrator.demonstrator import main
        main()

# Example of registering a custom visualization
def example_custom_visualization():
    """
    Example showing how to register a custom visualization.
    
    This shows how to create and register a custom visualization for a specific detector.
    """
    import numpy as np
    import plotly.graph_objects as go
    from dtaianomaly.utils import is_univariate
    
    def custom_visualization_function(detector, x, processed_x, time_steps):
        """Example custom visualization function."""
        # Format data for plotting
        if is_univariate(processed_x):
            plot_x = processed_x.flatten()
            # Generate dummy y values for 2D visualization
            plot_y = np.sin(time_steps / 10.0)
        else:
            # For multivariate data, use first two dimensions
            plot_x = processed_x[:, 0] if processed_x.shape[1] > 0 else processed_x
            plot_y = processed_x[:, 1] if processed_x.shape[1] > 1 else np.zeros_like(plot_x)
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                mode='markers',
                marker=dict(
                    size=8,
                    colorscale='Viridis',
                    showscale=True
                ),
                hoverinfo='text'
            )
        )
        
        fig.update_layout(
            title="Custom Visualization Example",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode="closest",
            height=600,
            width=900
        )
        
        return fig
    
    # Register the visualization with a specific detector
    register_custom_visualization("IsolationForest", "Custom Visualization", custom_visualization_function)
    
    print("Custom visualization registered successfully!")

# Export main function and the run_demonstrator function
__all__ = ['main', 'run_demonstrator', 'register_custom_visualization', 'example_custom_visualization', 'run_custom_detector_demo', 'run_with_detector'] 
