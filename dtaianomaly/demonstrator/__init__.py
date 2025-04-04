"""
Demonstrator module for the dtaianomaly package.

This module provides a Streamlit-based interface for exploring and visualizing
anomaly detection algorithms on time series data.
"""

import os
import sys
import pathlib

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
    except ImportError:
        # Fallback to os.system if streamlit module can't be properly imported
        os.system(f"streamlit run {script_path}")

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
__all__ = ['main', 'run_demonstrator', 'register_custom_visualization', 'example_custom_visualization'] 
