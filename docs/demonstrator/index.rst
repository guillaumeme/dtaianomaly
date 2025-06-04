Demonstrator
============

Abstract
--------

Users can utilize the dtaianomaly demonstrator to evaluate anomaly detection algorithms through interactive web-based testing on time series data. Streamlit development of the tool enables users to perform comparisons between various anomaly detection methods along with preprocessing steps and thresholding methods without needing programming knowledge. The tool establishes a connection between theoretical knowledge of anomaly detection methods and their practical deployment.

Introduction
------------

Time series anomaly detection stands as an essential process for financial institutions as well as healthcare providers and industrial monitoring organizations. The assessment of different detection algorithms becomes difficult for users who lack programming experience. The dtaianomaly demonstrator functions as an interactive solution that enables users to:

1. Users can upload and display time series information through the system.
2. Apply various preprocessing techniques
3. Run and compare multiple anomaly detection algorithms
4. Users can assess detector performance by choosing different evaluation metrics.
5. Users can easily adjust algorithm parameters for their needs.
6. Export results for further analysis

Through its interface users can conduct educational exploration as well as practical tasks which helps them understand algorithm performance on specific data and select appropriate solutions for their needs. An introduction video is available in the :ref:`video-demonstration` section to help new users get started quickly.

**Why These Features Are Essential**

Each feature addresses a specific challenge in anomaly detection workflow:

- **Data Upload and Visualization**: Understanding your data is the first critical step in anomaly detection. Visual inspection helps identify patterns, seasonality, and potential issues before applying algorithms. This feature is necessary because anomaly detection performance heavily depends on data characteristics.

- **Preprocessing Techniques**: Raw time series data often contains noise, missing values, or requires normalization. Preprocessing is essential because it can significantly impact detection accuracy. Users need to experiment with different preprocessing approaches to find what works best for their specific use case.

- **Algorithm Comparison**: No single anomaly detection algorithm performs best across all scenarios. Different algorithms have different strengths - some excel at detecting point anomalies, others at detecting contextual anomalies. Users need the ability to compare multiple algorithms because choosing the right one can mean the difference between 60% and 95% detection accuracy.

- **Evaluation Metrics**: Different applications prioritize different aspects of performance. In fraud detection, high precision might be crucial to minimize false alarms, while in medical monitoring, high recall is essential to catch all anomalies. Multiple metrics are necessary because they provide different perspectives on algorithm performance.

- **Parameter Adjustment**: Algorithm performance is highly sensitive to parameter settings. Default parameters rarely work optimally for specific datasets. This feature is critical because proper parameter tuning can improve detection rates by 20-30% or more.

- **Result Export**: Real-world anomaly detection requires integration with existing workflows. Export functionality is essential for documentation, reporting, and further analysis using specialized tools.

Using the Demonstrator
----------------------

Installation
~~~~~~~~~~~~

**Prerequisites**

- Python 3.8–3.13
- pip

**Setup Environment**

For macOS/Linux:

.. code-block:: bash

   python3 -m venv dtaianomaly-demo-venv
   source dtaianomaly-demo-venv/bin/activate

For Windows (PowerShell):

.. code-block:: powershell

   python -m venv dtaianomaly-demo-venv
   .\dtaianomaly-demo-venv\Scripts\Activate.ps1

**Install Demonstrator**

Run in your shell (bash or PowerShell):

.. code-block:: bash

   pip install "dtaianomaly[demonstrator] @ git+https://github.com/guillaumeme/dtaianomaly.git"

Starting the Demonstrator
~~~~~~~~~~~~~~~~~~~~~~~~~

Launch the Streamlit app by first entering Python:

.. code-block:: bash

   python

Then in the Python console:

.. code-block:: python

   from dtaianomaly.demonstrator import run_demonstrator
   run_demonstrator()

Using the Interface
~~~~~~~~~~~~~~~~~~~

Once launched, the demonstrator opens in your default web browser with a user-friendly interface:

.. figure:: ../_static/images/demonstrator/main_interface.png
   :alt: dtaianomaly demonstrator main interface
   :align: center
   :width: 100%

   The main interface of the dtaianomaly demonstrator

1. **Dataset Selection**
   
   In the sidebar, choose between built-in datasets or upload your own CSV/Excel file. For custom uploads, ensure your data has columns for 'Time Step', 'Value', and 'Label' (0 for normal, 1 for anomalies).

   .. figure:: ../_static/images/demonstrator/dataset_selection.png
      :alt: Dataset selection options
      :align: center
      :width: 80%

      Dataset selection options in the sidebar

   **Why This Design Choice**: The sidebar location keeps the main area focused on results while making data selection always accessible. The requirement for specific column names ensures consistency and prevents errors during processing.

2. **Configure Evaluation Metrics**
   
   Select metrics like Area Under ROC, Precision, Recall, or F1-score to evaluate detector performance.

   **Why Multiple Metrics Matter**: Each metric captures different aspects of performance. For example, a detector might have high precision (few false positives) but low recall (misses many anomalies). Understanding these trade-offs is crucial for selecting the right algorithm for your specific application.

3. **Thresholding Method**
   
   Choose a method to convert continuous anomaly scores into binary anomaly labels.

   **Why Thresholding is Critical**: Most anomaly detectors output continuous scores indicating "anomalousness." Converting these to binary decisions (anomaly/normal) requires setting a threshold. Different thresholding methods (fixed cutoff, contamination rate, statistical) can dramatically affect the final results. The choice depends on your domain knowledge and tolerance for false positives/negatives.

4. **Detector Configuration**
   
   - Add detectors using the '+' button
   - For each detector, select the algorithm and configure its parameters
   - Set up preprocessing steps specific to each detector

   .. figure:: ../_static/images/demonstrator/detector_configuration.png
      :alt: Detector configuration panel
      :align: center
      :width: 80%

      Detector configuration panel

   **Why Per-Detector Configuration**: Different algorithms may require different preprocessing. For example, distance-based methods might need normalization, while tree-based methods might work better without it. This flexibility allows users to optimize each algorithm independently.

5. **Execute Detection**
   
   Run individual detectors or all detectors at once using the respective "Run" buttons.

   **Why Both Options**: Individual execution allows for quick testing and debugging, while batch execution enables efficient comparison of multiple approaches.

6. **Analyze Results**
   
   - View performance metrics for each detector
   - Explore visualizations of the time series with detected anomalies
   - Compare detector performance across multiple algorithms
   - Export results to Excel or CSV for further analysis

   .. figure:: ../_static/images/demonstrator/visualization_example.png
      :alt: Visualization of detection results
      :align: center
      :width: 80%

      Example visualization of time series with detected anomalies

   .. figure:: ../_static/images/demonstrator/comparison_view.png
      :alt: Detector comparison view
      :align: center
      :width: 80%

      Comparative view of multiple detectors

   **Why Comprehensive Visualization**: Visual analysis helps users understand not just whether an algorithm works, but how it works. Seeing where algorithms agree or disagree on anomalies provides insights into their behavior and helps build trust in the results.

.. _video-demonstration:

Video Demonstration
~~~~~~~~~~~~~~~~~~

Watch a video demonstration of the dtaianomaly demonstrator to quickly understand its functionality:

:download:`Download Demonstration Video <../_static/videos/demonstrator_demo.mp4>`

**Why Video Documentation**: Complex interactive tools are best understood through demonstration. A video can show the workflow and interactions more effectively than static documentation, reducing the learning curve for new users.

Extending the Demonstrator
~~~~~~~~~~~~~~~~~~~~~~~~~~

The demonstrator supports custom anomaly detectors and visualizations. Here's an example of adding a custom detector:

.. code-block:: python

   from dtaianomaly.anomaly_detection import BaseDetector, Supervision
   from dtaianomaly.demonstrator import run_with_detector
   
   class MyCustomDetector(BaseDetector):
       def __init__(self, param1=0.5):
           super().__init__(Supervision.UNSUPERVISED)
           self.param1 = param1
       
       def _fit(self, X, y=None, **kwargs):
           # Implement training logic
           return self
       
       def _decision_function(self, X):
           # Implement anomaly scoring logic
           return scores
   
   # Run the demonstrator with your custom detector
   run_with_detector(MyCustomDetector)

For custom visualizations:

.. code-block:: python

   from dtaianomaly.demonstrator import register_custom_visualization
   import plotly.graph_objects as go
   
   def my_visualization(detector, x, processed_x, time_steps):
       fig = go.Figure()
       # Create your custom visualization
       return fig
   
   register_custom_visualization("MyDetector", "Custom View", my_visualization)

**Why Extensibility Matters**: Real-world applications often require domain-specific algorithms or visualizations. The ability to extend the demonstrator ensures it remains useful beyond the built-in algorithms, allowing researchers and practitioners to test new approaches within the same framework.

User Feedback and Evaluation
----------------------------

As part of the development process, a user survey was conducted to evaluate the usability and effectiveness of the demonstrator. The survey collected responses from 13 participants with varying levels of experience in time series analysis and anomaly detection.

**Why User Studies Are Crucial**: Software tools, especially those designed for non-programmers, must be validated with real users. Academic assumptions about usability often differ from practical experience. User feedback ensures the tool actually solves the problems it was designed to address.

Survey Results
~~~~~~~~~~~~~~

.. figure:: ../_static/images/demonstrator/surveyresults.png
   :alt: Bar chart showing survey results
   :align: center
   :width: 90%

   Survey results: Average ratings for different aspects of the demonstrator

**Key Quantitative Ratings (Scale of 1-10):**

**Why These Metrics Were Chosen**: The survey focused on both technical aspects (functionality, performance) and user experience (ease of use, interface design). This dual focus ensures the tool is both powerful and accessible. High ratings in ease of use (8.5/10) validate the no-code approach, while strong functionality scores (8.7/10) confirm the tool meets technical requirements.

Improvements Based on Feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the survey results and user comments, several improvements were made to the demonstrator:

1. **Enhanced Documentation and Tooltips:** Added explanations for technical terminology, especially for thresholding parameters and evaluation metrics.

   **Why This Matters**: Users without deep technical knowledge struggled with terms like "contamination rate" or "AUC-ROC." Clear explanations directly in the interface reduce the need to search external resources.

2. **Streamlined Interface:** Improved the layout to make the workflow more intuitive and clearer where to start.

   **Why This Matters**: First-time users reported confusion about the sequence of steps. A clear visual workflow reduces cognitive load and prevents errors.

3. **Better Visualizations:** Reduced the number of default visualizations to focus on the most informative ones.

   **Why This Matters**: Too many visualizations overwhelmed users. Curating the most valuable views helps users focus on insights rather than navigating options.

4. **Error Handling:** Improved error messages and fixed bugs related to dataset handling.

   **Why This Matters**: Cryptic error messages frustrate users and prevent problem-solving. Clear, actionable error messages help users resolve issues independently.

5. **Detector Comparison:** Enhanced the comparative view to make it easier to understand detector performance differences.

   **Why This Matters**: The primary value of the tool is comparing algorithms. Improvements here directly support the core use case.

Conclusion and Future Work
--------------------------

The ``dtaianomaly`` demonstrator provides an accessible and flexible platform for exploring time series anomaly detection techniques. It successfully bridges the gap between theory and practice, allowing users without extensive programming knowledge to experiment with different algorithms and configurations.

**Why This Tool Is Needed**: The field of anomaly detection suffers from a disconnect between academic research and practical application. Many powerful algorithms remain unused because potential users lack the programming skills to implement them. By removing this barrier, the demonstrator democratizes access to advanced anomaly detection techniques.

The current implementation includes a variety of detection methods, preprocessing techniques, and visualization options. However, there are several directions for future enhancement:

1. **Batch Processing**: Adding support for batch evaluation across multiple datasets.

   **Why This Matters**: Production environments often require testing algorithms on numerous datasets. Batch processing would enable systematic evaluation and benchmarking.

2. **Online Learning**: Extending the platform to support streaming data and online learning scenarios.

   **Why This Matters**: Many real-world applications involve continuous data streams. Supporting online learning would make the tool applicable to monitoring systems and real-time detection scenarios.

3. **Cloud Deployment**: Creating a hosted version of the demonstrator accessible via web browser without local installation.

   **Why This Matters**: Installation remains a barrier for some users. A cloud version would provide instant access and enable collaboration by sharing results via URLs.

The demonstrator's modular architecture makes it well-suited for these extensions, and its open-source nature encourages community contributions to enhance its capabilities. **The ultimate goal is to make anomaly detection accessible to everyone who needs it, regardless of their technical background.**
