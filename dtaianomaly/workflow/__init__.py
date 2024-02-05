
"""
A workflow is defined as a large experiment, in which a (large) number of algorithms are
applied to detect anomalies in a (large) number of time series through configuration
files. This allows for a structured evaluation of an algorithm and increases reproducibility
of the results.

The functionality regarding workflows is contained within this module. The main function of
this module is the :py:meth:`~dtaianomaly.workflow.execute_algorithm`. It can be imported
and used as follows:

.. code-block:: python

   from dtaianomaly.workflow import execute_algorithm
   execute_algorithm(...)  # Fill in the parameters as desired

Internally, the output is configured through an :py:class:`~dtaianomaly.workflow.OutputConfiguration`.
To ensure that this object can easily be initialized from code to start a workflow, it is
also available as follows:

.. code-block:: python

   from dtaianomaly.workflow import OutputConfiguration

We refer to the `documentation <https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/large_scale_experiments.html>`_
for more information regarding the configuration files for a workflow. You can also check out
`this example <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/blob/main/notebooks/execute_workflow.ipynb>`_
of running a workflow in code.
"""

from .execute_algorithms import execute_algorithms
from .handle_output_configuration import OutputConfiguration
