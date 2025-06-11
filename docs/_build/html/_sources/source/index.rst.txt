Welcome to madmatcher-tools documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/ml_model
   modules/tools
   modules/labeler
   modules/featurization
   modules/storage
   modules/utils
   modules/active_learning
   modules/feature
   modules/tokenizer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
========

madmatcher-tools is a Python library for entity matching and record linkage, providing tools for:

* Machine learning models for entity matching (both scikit-learn and PySpark)
* Active learning for efficient labeling
* Feature engineering and vectorization
* Data storage and management
* CLI-based labeling interface

The library is designed to be flexible and scalable, supporting both local (pandas) and distributed (PySpark) processing.

Installation
============

You can install madmatcher-tools using pip:

.. code-block:: bash

   pip install madmatcher-tools

For development, install with additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Quick Start
===========

Here's a simple example of using madmatcher-tools:

.. code-block:: python

   from madmatcher_tools import SKLearnModel, CLILabeler
   from sklearn.ensemble import HistGradientBoostingClassifier

   # Create and train a model
   model = SKLearnModel(HistGradientBoostingClassifier())
   model.train(training_data, vector_col='features', label_column='label')

   # Make predictions
   predictions = model.predict(test_data, vector_col='features', output_col='prediction')

For more detailed examples, see the individual module documentation. 