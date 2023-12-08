Installation
============

Dependencies
------------

The core dependencies of ``dtaianomaly`` are `NumPy <https://numpy.org/>`_,
`Pandas <https://pandas.pydata.org/>`_, `matplotlib <https://matplotlib.org/>`_,
`scikit-learn <https://scikit-learn.org/stable/>`_ and `scipy <https://www.scipy.org/>`_.

.. _anomaly_detection_warning:
.. warning::
    Note that certain anomaly detectors may also depend on additional packages, which
    are not installed by default. This is to avoid installing unnecessary dependencies.
    Any additional dependencies may be found in a ``requirements.txt``-file alongside the
    source code of the anomaly detector or in the documentation itself.

From PyPi
---------

``dtaianomaly`` is available through `PyPI <https://pypi.org/project/dtaianomaly/>`_:

.. code-block:: bash

    pip install dtaianomaly

Running above command will install the latest, *released* version.


From GitLab
-----------

Alternatively, it is possible to install the latest, *unreleased* version directly
from `GitLab <https://gitlab.kuleuven.be/u0143709/dtaianomaly>`_:

.. code-block:: bash

    pip install git+https://gitlab.kuleuven.be/u0143709/dtaianomaly.git


From source
-----------

``dtaianomaly`` can also be installed directly from the source code. First, download
the source from `GitLab <https://gitlab.kuleuven.be/u0143709/dtaianomaly>`_. It is also
possible to download the source code for a specific release on `this webpage <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/releases>`_.
Unzip the files, and navigate to the root directory of the repository in the terminal.
Finally, ``dtaianomaly`` can be installed through the following command:

.. code-block:: bash

    pip install .
