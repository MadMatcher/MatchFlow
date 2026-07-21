"""Machine Learning Model base classes and implementations.

This module provides base classes and implementations for machine learning models,
supporting both scikit-learn and PySpark ML models. It includes functionality for
training, prediction, confidence estimation, and entropy calculation.
"""

from abc import abstractmethod, ABC, abstractproperty
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
import pyspark.sql.types as T
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.linalg import VectorUDT, Vectors, DenseVector, SparseVector
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.ml import Transformer
import numpy as np
import warnings
from typing import Iterator, overload, Union, Optional
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from joblib import parallel_backend
from threadpoolctl import ThreadpoolController
from .utils import get_logger

log = get_logger(__name__)

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

    @abstractproperty
    def trained_model(self):
        """ The trained ML Model object
        
        Returns
        -------
        MLModel
            The trained ML Model object 
        """

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
    def predict_with_confidence(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, prediction_col: str, confidence_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        """Make predictions and confidence scores using the trained model.
        
        This method is more efficient than calling predict() and prediction_conf() separately
        as it computes both in a single pass when possible.
        
        Parameters
        ----------
        df : pandas.DataFrame or pyspark.sql.DataFrame
            The DataFrame containing the feature vectors to predict on
        vector_col : str
            Name of the column containing feature vectors
        prediction_col : str
            Name of the column to store predictions in
        confidence_col : str
            Name of the column to store confidence scores in
            
        Returns
        -------
        pandas.DataFrame or pyspark.sql.DataFrame
            The input DataFrame with predictions and confidence scores added
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
    def train(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, label_column: str):
        """Train the model on the given data.
        
        Parameters
        ----------
        df : pandas.DataFrame or pyspark.sql.DataFrame
            The DataFrame containing training data
        vector_col : str
            Name of the column containing feature vectors
        label_column : str
            Name of the column containing labels
            
        Returns
        -------
        MLModel
            The trained model (self) 
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

    def prep_fvs(self, fvs: Union[pd.DataFrame, SparkDataFrame], feature_col: str = 'feature_vectors') -> Union[pd.DataFrame, SparkDataFrame]:
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
        if isinstance(fvs, pd.DataFrame):
            if self.nan_fill is not None and len(fvs) > 0:
                fvs = fvs.copy()
                feature_data = fvs[feature_col]
                if hasattr(feature_data.iloc[0], '__iter__') and not isinstance(feature_data.iloc[0], str):
                    filled_features = []
                    for feature in feature_data:
                        if isinstance(feature, (list, np.ndarray, DenseVector, SparseVector)):
                            feature_array = np.asarray(_to_float_list(feature))
                            feature_array = np.nan_to_num(feature_array, nan=self.nan_fill)
                            filled_features.append(feature_array)
                        else:
                            filled_features.append(feature)
                    fvs[feature_col] = filled_features
                else:
                    fvs[feature_col] = fvs[feature_col].fillna(self.nan_fill)

            if self.use_vectors:
                fvs = convert_to_vector(fvs, feature_col)
            else:
                fvs = convert_to_array(fvs, feature_col)

        elif isinstance(fvs, SparkDataFrame):
            if self.nan_fill is not None:
                fvs = fvs.withColumn(feature_col, F.transform(feature_col, lambda x : F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(self.nan_fill)))

            if self.use_vectors:
                fvs = convert_to_vector(fvs, feature_col)
            else:
                fvs = convert_to_array(fvs, feature_col)
                if self.use_floats:
                    fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<float>'))
                else:
                    fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<double>'))
        
        else:
            raise TypeError(f"Unsupported DataFrame type: {type(fvs)}")

        return fvs

def convert_to_vector(df, col):
    """Convert array column to vector format for both pandas and Spark DataFrames.
    
    Parameters
    ----------
    df : pandas.DataFrame or pyspark.sql.DataFrame
        The DataFrame containing the column to convert
    col : str
        Name of the column to convert
        
    Returns
    -------
    pandas.DataFrame or pyspark.sql.DataFrame
        DataFrame with the column converted to vector format
    """
    if isinstance(df, pd.DataFrame):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        df = df.copy()
        df[col] = df[col].map(_to_dense_vector)
        return df
    elif isinstance(df, SparkDataFrame):
        if not isinstance(df.schema[col].dataType, VectorUDT):
            df = df.withColumn(col, array_to_vector(col))
        return df
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")

def _to_dense_vector(v):
    """
    convert a single feature vector to a pyspark DenseVector, leaving nulls
    and vectors that are already in the right format alone
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (DenseVector, SparseVector)):
        return v
    return Vectors.dense(np.asarray(v, dtype=np.float64))

def _to_float_list(v):
    """
    convert a single feature vector to a list of native python floats. numpy
    scalars are not accepted by spark's schema verifier, and ndarray.tolist()
    is what unboxes them
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (DenseVector, SparseVector)):
        v = v.toArray()
    return np.asarray(v, dtype=np.float64).tolist()

_DOUBLE_ARRAY = T.ArrayType(T.DoubleType())
_FLOAT_ARRAY = T.ArrayType(T.FloatType())
_ARRAY_TYPES = {_DOUBLE_ARRAY, _FLOAT_ARRAY}

def convert_to_array(df, col):
    """Convert vector column to array format for both pandas and Spark DataFrames.
    
    Parameters
    ----------
    df : pandas.DataFrame or pyspark.sql.DataFrame
        The DataFrame containing the column to convert
    col : str
        Name of the column to convert
        
    Returns
    -------
    pandas.DataFrame or pyspark.sql.DataFrame
        DataFrame with the column converted to array format
    """
    if isinstance(df, pd.DataFrame):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        df = df.copy()
        df[col] = df[col].map(_to_float_list)
        return df
    elif isinstance(df, SparkDataFrame):
        if df.schema[col].dataType not in _ARRAY_TYPES:
            df = df.withColumn(col, vector_to_array(col))
        return df
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def make_training_schema(fvs, columns, label_cols=('label', 'labeled_in_iteration')):
    """Build the spark schema for the local (pandas) training feature vectors.

    The schema is taken from `fvs` -- which has already been through
    `MLModel.prep_fvs` -- rather than inferred from the pandas DataFrame, so
    that `feature_vectors` is declared with the type the model actually uses
    (VectorUDT for spark models, array<float/double> for sklearn models).

    Parameters
    ----------
    fvs : pyspark.sql.DataFrame
        the prepped feature vectors that the training data is drawn from
    columns : iterable of str
        the columns of the local training DataFrame, only fvs fields that
        appear here are included
    label_cols : tuple of str
        the label columns added during active learning, appended as doubles

    Returns
    -------
    pyspark.sql.types.StructType
    """
    columns = set(columns)
    fields = [f for f in fvs.schema.fields if f.name in columns and f.name not in label_cols]
    fields += [T.StructField(c, T.DoubleType(), True) for c in label_cols]

    return T.StructType(fields)


def _unbox(v):
    return v.item() if isinstance(v, np.generic) else v


def to_spark_training_fvs(spark, pdf, schema, feature_col='feature_vectors'):
    """Create a spark DataFrame from the local (pandas) training feature vectors.

    `spark.createDataFrame` matches the columns of a pandas DataFrame to the
    fields of an explicit schema *by position* and then verifies each value
    with a strict `type(obj) in (float,)` style check, so the DataFrame has to
    be lined up with the schema first. This

    * reindexes the columns to the schema's fields (dropping extras),
    * puts `feature_col` in the representation the schema declares,
    * and replaces numpy scalars with the python types spark accepts.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
    pdf : pandas.DataFrame
        the local training feature vectors
    schema : pyspark.sql.types.StructType
        the schema from `make_training_schema`
    feature_col : str
        name of the column containing feature vectors

    Returns
    -------
    pyspark.sql.DataFrame
    """
    pdf = pdf.reindex(columns=[f.name for f in schema.fields])

    if isinstance(schema[feature_col].dataType, VectorUDT):
        pdf = convert_to_vector(pdf, feature_col)
    else:
        pdf = convert_to_array(pdf, feature_col)

    for field in schema.fields:
        if field.name == feature_col:
            continue
        # labels are written as both ints and floats during active learning,
        # DoubleType only accepts floats
        if isinstance(field.dataType, (T.DoubleType, T.FloatType)):
            pdf[field.name] = pdf[field.name].astype('float64')
        elif pdf[field.name].dtype == object:
            pdf[field.name] = pdf[field.name].map(_unbox)

    return spark.createDataFrame(pdf, schema=schema)

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

    **model_args : dict
        Additional arguments to pass to the model constructor
    """
    
    def __init__(self, model, nan_fill=None, use_floats=True, **model_args):
        try:
            check_is_fitted(model)
            self._trained_model = model
            self._model = model.__class__
            self._model_args = {}
        except (NotFittedError, TypeError):
            self._trained_model = None
            self._model_args = model_args.copy()
            self._model = model
        self._nan_fill = nan_fill
        self._use_floats = use_floats
        self._vector_buffer = None

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
        if self._nan_fill is not None:
            np.nan_to_num(X, copy=False, nan=self._nan_fill)
        return X

    def _predict(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            yield pd.Series(self._trained_model.predict(X))

    def predict(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, output_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        if isinstance(df, pd.DataFrame):
            X = self._make_feature_matrix(df[vector_col].tolist())
            df[output_col] = self._trained_model.predict(X)
            return df
        if isinstance(df, SparkDataFrame):
            df = convert_to_array(df, vector_col)
            f = F.pandas_udf(self._predict, T.DoubleType())
            return df.withColumn(output_col, f(vector_col))
    
    def _predict_with_confidence(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            predictions = self._trained_model.predict(X)
            confidences = self._trained_model.predict_proba(X).max(axis=1)
            yield pd.DataFrame({
                'prediction': predictions,
                'confidence': confidences
            })

    def predict_with_confidence(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, prediction_col: str, confidence_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        if isinstance(df, pd.DataFrame):
            X = self._make_feature_matrix(df[vector_col].tolist())
            predictions = self._trained_model.predict(X)
            probs = self._trained_model.predict_proba(X).max(axis=1)
            df[prediction_col] = predictions
            df[confidence_col] = probs
            return df
        if isinstance(df, SparkDataFrame):
            df = convert_to_array(df, vector_col)
            schema = StructType([
                StructField("prediction", DoubleType(), True),
                StructField("confidence", DoubleType(), True)
            ])
            f = F.pandas_udf(self._predict_with_confidence, schema)
            result = df.withColumn("temp_result", f(vector_col))
            result = result.withColumn(prediction_col, F.col("temp_result.prediction"))\
                          .withColumn(confidence_col, F.col("temp_result.confidence"))\
                          .drop("temp_result")
            return result
    
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

    def train(self, df, vector_col : str, label_column : str):
        if isinstance(df, SparkDataFrame):
            df = convert_to_array(df, vector_col)
            df = df.toPandas()
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values
        self._trained_model = self._model(**self._model_args)
        self._trained_model.fit(X, y)
        return self
    
    def cross_val_scores(self, df, vector_col : str, label_column : str, cv : int = 10):
        df = convert_to_array(df, vector_col)
        df = df.toPandas()
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values

        scores = cross_val_score(self.get_model(), X, y, cv=cv)
        return scores

class SparkMLModel(MLModel):

    def __init__(self, model, nan_fill = 0.0, **model_args):
        if isinstance(model, Transformer):
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

    @property
    def trained_model(self):
        return self._trained_model

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

    def predict(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, output_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        return_pandas = isinstance(df, pd.DataFrame)
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            df = spark.createDataFrame(df) 
        if not hasattr(self, '_model_args'):
            return SparkMLModel(self._trained_model).predict(df, vector_col, output_col)
        df = convert_to_vector(df, vector_col)
        cols = df.columns
        out = F.col(self._trained_model.getPredictionCol()).alias(output_col)
        predictions = self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, out)
        if return_pandas:
            return predictions.toPandas()
        else:
            return predictions
    
    def predict_with_confidence(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, prediction_col: str, confidence_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        return_pandas = isinstance(df, pd.DataFrame)
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            df = spark.createDataFrame(df) 
        if not hasattr(self, '_model_args'):
            return SparkMLModel(self._trained_model).predict_with_confidence(df, vector_col, prediction_col, confidence_col)
        df = convert_to_vector(df, vector_col)
        cols = df.columns
        pred_out = F.col(self._trained_model.getPredictionCol()).alias(prediction_col)
        conf_out = F.array_max(vector_to_array(F.col(self._trained_model.getProbabilityCol()))).alias(confidence_col)
        result = self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, pred_out, conf_out)
        if return_pandas:
            return result.toPandas()
        else:
            return result
    
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
        
    def train(self, df, vector_col : str, label_column : str):
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            df = spark.createDataFrame(df)
        df = convert_to_vector(df, vector_col)
        self._trained_model = self.get_model().setFeaturesCol(vector_col)\
                                            .setLabelCol(label_column)\
                                            .fit(df)\
                                            .setPredictionCol('__PREDICTION_TMP')\
                                            .setProbabilityCol('__PROB_TMP')\
                                            .setRawPredictionCol('__RAW_PREDICTION_TMP')
        return self
