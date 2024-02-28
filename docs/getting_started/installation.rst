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

From GitLab
-----------

The preferred manner to install ``dtaianomaly`` is directly from `GitLab <https://gitlab.kuleuven.be/u0143709/dtaianomaly>`_:

.. code-block:: bash

    pip install git+https://gitlab.kuleuven.be/u0143709/dtaianomaly.git

Note that this will install the latest, *unreleased* version. Therefore, we recommend specifying
a certain version as a tag. For example, you can install version ``X.Y.Z`` as follows:

.. code-block:: bash

    pip install git+https://gitlab.kuleuven.be/u0143709/dtaianomaly.git@X.Y.Z

The `release page <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/releases>`_ contains more
information regarding the different versions.


From source
-----------

``dtaianomaly`` can also be installed directly from the source code. First, download
the source from `GitLab <https://gitlab.kuleuven.be/u0143709/dtaianomaly>`_. It is also
possible to download the source code for a specific release on `this webpage <https://gitlab.kuleuven.be/u0143709/dtaianomaly/-/releases>`_.
Unzip the files, and navigate to the root directory of the repository in the terminal.
Finally, ``dtaianomaly`` can be installed through the following command:

.. code-block:: bash

    pip install .


From PyPi
---------

Up until version ``0.1.3``, ``dtaianomaly`` was available through `PyPI <https://pypi.org/project/dtaianomaly/>`_.
This means those versions can be installed as follows:

.. code-block:: bash

    pip install dtaianomaly

This is no longer possible due to the dependency on TSB-UAD, because PyPi does
not allow direct dependencies such as a GitHub repository.
