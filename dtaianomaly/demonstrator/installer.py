import subprocess
import sys

def install_package(package):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

if __name__ == "__main__":
    # Comprehensive list of libraries for the Streamlit app and dtaianomaly
    packages = [
        "streamlit",       # Web app framework
        "pandas",          # Data manipulation
        "numpy",           # Numerical computations
        "matplotlib",      # Plotting
        "scipy",           # Statistical functions
        "seaborn",         # Data visualization
        "plotly",          # Interactive plots
        "stumpy",          # Dependency for dtaianomaly (KShapeAnomalyDetector)
        "pyod",            # Dependency for PyODAnomalyDetector
        "scikit-learn",    # Machine learning utilities (possible dtaianomaly dependency)
        "statsmodels",     # Time series and statistical modeling (potential dependency)
        "joblib"           # Model persistence (potential dependency)
    ]
    print("Starting installation of required libraries...")
    for package in packages:
        install_package(package)
    print("Installation complete. All dependencies should now be installed!")