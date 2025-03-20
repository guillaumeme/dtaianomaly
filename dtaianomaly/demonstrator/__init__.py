"""
Demonstrator module for the dtaianomaly package.

This module provides a Streamlit-based interface for exploring and visualizing
anomaly detection algorithms on time series data.
"""

import os
import sys
import pathlib

from dtaianomaly.demonstrator.Demonstrator import main

def run_demonstrator():
    """Run the Streamlit-based demonstrator application."""
    script_path = pathlib.Path(__file__).parent / "Demonstrator.py"
    
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

# Export main function and the run_demonstrator function
__all__ = ['main', 'run_demonstrator'] 