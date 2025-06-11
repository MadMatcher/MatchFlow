"""Machine Learning Model base classes and implementations.

This module provides base classes and implementations for machine learning models,
supporting both scikit-learn and PySpark ML models. It includes functionality for
training, prediction, confidence estimation, and entropy calculation.
"""

from abc import abstractmethod, ABC, abstractproperty
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
import numpy as np
import warnings
from typing import Iterator, overload, Union, Optional
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator
from joblib import parallel_backend
from threadpoolctl import ThreadpoolController

class MLModel(ABC):
    """Abstract base class for machine learning models.
    
    This class defines the interface that all machine learning models must implement,
    whether they are scikit-learn models or PySpark ML models. It provides methods
    for training, prediction, confidence estimation, and entropy calculation.
    
    Attributes
    ----------
    nan_fill : float or None
        Value to use for filling NaN values in feature vectors
    use_vectors : bool
        Whether the model expects feature vectors in vector format
    use_floats : bool
        Whether the model uses float32 (True) or float64 (False) precision
    """
    
    @abstractproperty
    def nan_fill(self) -> Optional[float]:
        """Value to use for filling NaN values in feature vectors.
        
        Returns
        -------
        float or None
            The value to use for filling NaN values, or None if no filling is needed
        """
        pass

    @abstractproperty
    def use_vectors(self) -> bool:
        """Whether the model expects feature vectors in vector format.
        
        Returns
        -------
        bool
            True if the model expects vectors, False if it expects arrays
        """
        pass

    @abstractproperty
    def use_floats(self) -> bool:
        """Whether the model uses float32 or float64 precision.
        
        Returns
        -------
        bool
            True if the model uses float32, False if it uses float64
        """
        pass

    @abstractmethod
    def predict(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, output_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        """Make predictions using the trained model.
        
        Parameters
        ----------
        df : pandas.DataFrame or pyspark.sql.DataFrame
            The DataFrame containing the feature vectors to predict on
        vector_col : str
            Name of the column containing feature vectors
        output_col : str
            Name of the column to store predictions in
            
        Returns
        -------
        pandas.DataFrame or pyspark.sql.DataFrame
            The input DataFrame with predictions added in the output_col
        """
        pass

    @abstractmethod
    def prediction_conf(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, label_column: str) -> Union[pd.DataFrame, SparkDataFrame]:
        """Calculate prediction confidence scores.
        
        Parameters
        ----------
        df : pandas.DataFrame or pyspark.sql.DataFrame
            The DataFrame containing the feature vectors
        vector_col : str
            Name of the column containing feature vectors
        label_column : str
            Name of the column containing true labels
            
        Returns
        -------
        pandas.DataFrame or pyspark.sql.DataFrame
            The input DataFrame with confidence scores added
        """
        pass

    @abstractmethod
    def entropy(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, output_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        """Calculate entropy of predictions.
        
        Parameters
        ----------
        df : pandas.DataFrame or pyspark.sql.DataFrame
            The DataFrame containing the feature vectors
        vector_col : str
            Name of the column containing feature vectors
        output_col : str
            Name of the column to store entropy values in
            
        Returns
        -------
        pandas.DataFrame or pyspark.sql.DataFrame
            The input DataFrame with entropy values added in the output_col
        """
        pass

    @abstractmethod
    def train(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, label_column: str, return_estimator: bool = False) -> Union['MLModel', BaseEstimator]:
        """Train the model on the given data.
        
        Parameters
        ----------
        df : pandas.DataFrame or pyspark.sql.DataFrame
            The DataFrame containing training data
        vector_col : str
            Name of the column containing feature vectors
        label_column : str
            Name of the column containing labels
        return_estimator : bool, optional
            If True, return the trained estimator instead of self
            
        Returns
        -------
        MLModel or BaseEstimator
            The trained model (self) or the trained estimator if return_estimator is True
        """
        pass

    @abstractmethod
    def params_dict(self) -> dict:
        """Get a dictionary of model parameters.
        
        Returns
        -------
        dict
            Dictionary containing model parameters and configuration
        """
        pass

    def prep_fvs(self, fvs: Union[pd.DataFrame, SparkDataFrame], feature_col: str = 'features') -> Union[pd.DataFrame, SparkDataFrame]:
        """Prepare feature vectors for model input.
        
        This method handles NaN filling and conversion between vector and array formats
        based on the model's requirements.
        
        Parameters
        ----------
        fvs : pandas.DataFrame or pyspark.sql.DataFrame
            DataFrame containing feature vectors
        feature_col : str, optional
            Name of the column containing feature vectors
            
        Returns
        -------
        pandas.DataFrame or pyspark.sql.DataFrame
            DataFrame with prepared feature vectors
        """
        if self.nan_fill is not None:
            fvs = fvs.withColumn(feature_col, F.transform(feature_col, lambda x : F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(self._model.nan_fill)))

        if self.use_vectors:
            fvs = convert_to_vector(fvs, feature_col)
        else:
            fvs = convert_to_array(fvs, feature_col)
            if self.use_floats:
                fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<float>'))
            else:
                fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<double>'))

        return fvs

def convert_to_vector(df, col):
    if not isinstance(df.schema[col].dataType, VectorUDT):
        df = df.withColumn(col, array_to_vector(col))
    return df

_DOUBLE_ARRAY = T.ArrayType(T.DoubleType())
_FLOAT_ARRAY = T.ArrayType(T.FloatType())
_ARRAY_TYPES = {_DOUBLE_ARRAY, _FLOAT_ARRAY}

def convert_to_array(df, col):
    if df.schema[col].dataType not in _ARRAY_TYPES:
        df = df.withColumn(col, vector_to_array(col))
    return df

class SKLearnModel(MLModel):
    """Scikit-learn model wrapper.
    
    This class wraps scikit-learn models to provide a consistent interface
    with PySpark ML models. It handles conversion between pandas and PySpark
    DataFrames, and manages model training and prediction.
    
    Parameters
    ----------
    model : sklearn.base.BaseEstimator or type
        The scikit-learn model class or instance to use
    nan_fill : float or None, optional
        Value to use for filling NaN values
    use_floats : bool, optional
        Whether to use float32 (True) or float64 (False) precision
    execution : str, optional
        Execution mode: "local" for pandas or "spark" for PySpark
    **model_args : dict
        Additional arguments to pass to the model constructor
    """
    
    def __init__(self, model, nan_fill=None, use_floats=True, execution="local", **model_args):
        if hasattr(model, 'predict') and not isinstance(model, type):
            self._trained_model = model
            self._model = model.__class__
            self._model_args = {}
        else:
            self._trained_model = None
            self._model_args = model_args.copy()
            self._model = model
        self._nan_fill = nan_fill
        self._use_floats = use_floats
        self._vector_buffer = None
        self.execution = execution
    

    def params_dict(self):
        return {
                'model' : str(self._model),
                'nan_fill' : self._nan_fill,
                'model_args' : self._model_args.copy()
        }
    
    def _no_threads(self):
        tpc = ThreadpoolController()
        tpc.limit(limits=1, user_api='openmp')
        tpc.limit(limits=1, user_api='blas')
        pass

    @property
    def nan_fill(self):
        return self._nan_fill

    @property
    def use_vectors(self):
        return False

    @property
    def use_floats(self):
        return self._use_floats

    @property
    def trained_model(self):
        return self._trained_model

    def get_model(self):
        return self._model(**self._model_args)

    def _allocate_buffer(self, nrows, ncols):
        needed_size = nrows * ncols
        if self._vector_buffer is None or self._vector_buffer.size < needed_size:
            self._vector_buffer = np.empty(needed_size, dtype=(np.float32 if self.use_floats else np.float64) )

        return self._vector_buffer[:needed_size].reshape(nrows, ncols)

    def _make_feature_matrix(self, vecs):
        if len(vecs) == 0:
            return None
        buffer = self._allocate_buffer(len(vecs), len(vecs[0]))
        X = np.stack(vecs, axis=0, out=buffer)

        return X

    def _predict(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            yield pd.Series(self._trained_model.predict(X))

    @overload
    def predict(self, df: pd.DataFrame, vector_col: str, output_col: str) -> pd.DataFrame: ...

    @overload
    def predict(self, estimator: BaseEstimator, df: pd.DataFrame, vector_col: str, output_col: str) -> pd.DataFrame: ...

    def predict(self, *args):
        # Overloaded method
        if len(args) == 4 and isinstance(args[0], BaseEstimator):
            est, df, vector_col, out_col = args
            if not hasattr(est, 'predict'):
                raise TypeError("Estimator must have predict method.")
            return SKLearnModel(est).predict(df, vector_col, out_col)
        if len(args) != 3:
            raise ValueError("predict() expects (df, vector_col, output_col) or (estimator, df, vector_col, output_col)")
        df, vector_col, output_col = args  
        if isinstance(df, pd.DataFrame) and self.execution == "local":
            X = self._make_feature_matrix(df[vector_col].tolist())
            df[output_col] = self._trained_model.predict(X)
            return df
        if isinstance(df, SparkDataFrame) or self.execution == "spark":
            df = convert_to_array(df, vector_col)
            f = F.pandas_udf(self._predict, T.DoubleType())
            return df.withColumn(output_col, f(vector_col))
    
    def _prediction_conf(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            probs = self._trained_model.predict_proba(X)
            yield pd.Series(probs.max(axis=1))

    def prediction_conf(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        df = convert_to_array(df, vector_col)
        f = F.pandas_udf(self._prediction_conf, T.DoubleType())
        return df.withColumn(output_col, f(vector_col))

    def _entropy(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            probs = self._trained_model.predict_proba(X)
            yield pd.Series(np.nan_to_num((-probs * np.log2(probs)).sum(axis=1)))
    
    def entropy(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        df = convert_to_array(df, vector_col)
        f = F.pandas_udf(self._entropy, T.DoubleType())
        return df.withColumn(output_col, f(vector_col))

    def train(self, df, vector_col : str, label_column : str, return_estimator : bool = False):
        if isinstance(df, SparkDataFrame):
            df = convert_to_array(df, vector_col)
            df = df.toPandas()
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values
        self._trained_model = self._model(**self._model_args)
        self._trained_model.fit(X, y)
        return self._trained_model if return_estimator else self
    
    def cross_val_scores(self, df, vector_col : str, label_column : str, cv : int = 10):
        df = convert_to_array(df, vector_col)
        df = df.toPandas()
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values

        scores = cross_val_score(self.get_model(), X, y, cv=cv)
        return scores

class SparkMLModel(MLModel):

    def __init__(self, model, nan_fill = 0.0, **model_args):
        if hasattr(model, 'predict') and not isinstance(model, type):
            self._trained_model = model
            self._model = model.__class__
            self._model_args = {}
        else:
            self._trained_model = None
            self._model_args = model_args.copy()
            self._model = model
        self._nan_fill = nan_fill

    @property
    def nan_fill(self):
        return self._nan_fill

    @property
    def use_vectors(self):
        return True

    @property
    def use_floats(self):
        return False

    def get_model(self):
        return self._model(**self._model_args)

    def params_dict(self):
        return {
                'model' : str(self._model),
                'model_args' : self._model_args.copy()
        }

    def prediction_conf(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')

        df = convert_to_vector(df, vector_col)
        cols = df.columns
        out = F.array_max(vector_to_array(F.col(self._trained_model.getProbabilityCol()))).alias(output_col)

        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, out)

    @overload
    def predict(self, df: SparkDataFrame, vector_col: str, output_col: str) -> SparkDataFrame: ...

    @overload
    def predict(self, transformer: object, df: SparkDataFrame, vector_col: str, output_col: str) -> SparkDataFrame: ...

    def predict(self, *args):
        if len(args) == 4 and hasattr(args[0], 'transform'):
            transformer, df, vector_col, out_col = args
            return SparkMLModel(transformer).predict(df, vector_col, out_col)
        if len(args) != 3:
            raise ValueError("predict() expects (df, vector_col, output_col) or (transformer, df, vector_col, output_col)")
        df, vector_col, out_col = args
        model = self._trained_model
        if model is None:
            raise RuntimeError('Model must be trained to predict')
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            df = spark.createDataFrame(df) 
        if not hasattr(self, '_model_args'):
            return SparkMLModel(model).predict(df, vector_col, out_col)
        df = convert_to_vector(df, vector_col)
        cols = df.columns
        out = F.col(self._trained_model.getPredictionCol()).alias(out_col)

        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, out)
    
    def _entropy_component(self, p_col, idx):
        return F.when(p_col.getItem(idx) != 0.0, -p_col.getItem(idx) * F.log2(p_col.getItem(idx))).otherwise(0.0)

    def _entropy_expr(self, probs, classes=2):
        p_col = F.col(probs)

        e = self._entropy_component(p_col, 0)
        for i in range(1, classes):
            e = e + self._entropy_component(p_col, i)

        return e


    def entropy(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to compute entropy')
        df = convert_to_vector(df, vector_col)
        prob_col = self._trained_model.getProbabilityCol()
        prob_array = 'prob_array'
        cols = df.columns
        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, vector_to_array(prob_col).alias(prob_array))\
                                    .withColumn(output_col, self._entropy_expr(prob_array))\
                                    .drop(prob_array)
        
    def train(self, df, vector_col : str, label_column : str, return_estimator : bool = False):
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            df = spark.createDataFrame(df)
        df = convert_to_vector(df, vector_col)
        self._trained_model = self.get_model().setFeaturesCol(vector_col)\
                                            .setLabelCol(label_column)\
                                            .fit(df)\
                                            .setPredictionCol('__PREDICTION_TMP')\
                                            .setProbabilityCol('__PROB_TMP')\
                                            .setRawPredictionCol('__RAW_PREDICTION_TMP')
        return self._trained_model if return_estimator else self
