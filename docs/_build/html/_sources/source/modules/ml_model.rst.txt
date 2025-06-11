Machine Learning Models
=====================

.. automodule:: ml_model
   :members:
   :undoc-members:
   :show-inheritance:

MLModel Base Class
----------------

.. autoclass:: MLModel
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Abstract Properties

   .. autoattribute:: nan_fill
   .. autoattribute:: use_vectors
   .. autoattribute:: use_floats

   .. rubric:: Abstract Methods

   .. automethod:: predict
   .. automethod:: prediction_conf
   .. automethod:: entropy
   .. automethod:: train
   .. automethod:: params_dict

   .. rubric:: Concrete Methods

   .. automethod:: prep_fvs

SKLearnModel
-----------

.. autoclass:: SKLearnModel
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. automethod:: predict
   .. automethod:: prediction_conf
   .. automethod:: entropy
   .. automethod:: train
   .. automethod:: cross_val_scores

SparkMLModel
----------

.. autoclass:: SparkMLModel
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. automethod:: predict
   .. automethod:: prediction_conf
   .. automethod:: entropy
   .. automethod:: train

Utility Functions
--------------

.. autofunction:: convert_to_vector
.. autofunction:: convert_to_array 