import pytest
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from ml_model import SKLearnModel, SparkMLModel

@pytest.fixture
def spark_session():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("test") \
        .getOrCreate()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'features': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ],
        'label': [1.0, 0.0, 1.0, 0.0, 1.0]
    })

@pytest.fixture
def sample_spark_data(spark_session, sample_data):
    """Convert sample data to Spark DataFrame."""
    return spark_session.createDataFrame(sample_data)

def test_sklearn_model_properties():
    """Test SKLearnModel properties."""
    model = SKLearnModel(HistGradientBoostingClassifier, nan_fill=0.0, use_floats=True)
    
    assert model.nan_fill == 0.0
    assert not model.use_vectors
    assert model.use_floats
    assert model._trained_model is None

def test_sklearn_model_train_predict(sample_data):
    """Test SKLearnModel training and prediction."""
    model = SKLearnModel(HistGradientBoostingClassifier)
    
    # Train the model
    trained_model = model.train(
        sample_data,
        vector_col='features',
        label_column='label'
    )
    
    assert trained_model._trained_model is not None
    
    # Test prediction
    predictions = trained_model.predict(
        sample_data,
        'features',
        'prediction'
    )
    
    assert 'prediction' in predictions.columns
    assert len(predictions) == len(sample_data)
    assert all(predictions['prediction'].isin([0.0, 1.0]))

def test_sklearn_model_prediction_conf(sample_data):
    """Test SKLearnModel prediction confidence."""
    model = SKLearnModel(HistGradientBoostingClassifier)
    trained_model = model.train(
        sample_data,
        vector_col='features',
        label_column='label'
    )
    
    # Convert to Spark DataFrame for prediction_conf
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(sample_data)
    
    # Test prediction confidence
    conf = trained_model.prediction_conf(
        spark_df,
        vector_col='features',
        output_col='confidence'
    )
    
    conf_pd = conf.toPandas()
    assert 'confidence' in conf_pd.columns
    assert len(conf_pd) == len(sample_data)
    assert all((conf_pd['confidence'] >= 0.0) & (conf_pd['confidence'] <= 1.0))

def test_sklearn_model_entropy(sample_data):
    """Test SKLearnModel entropy calculation."""
    model = SKLearnModel(HistGradientBoostingClassifier)
    trained_model = model.train(
        sample_data,
        vector_col='features',
        label_column='label'
    )
    
    # Convert to Spark DataFrame for entropy
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(sample_data)
    
    # Test entropy calculation
    entropy = trained_model.entropy(
        spark_df,
        vector_col='features',
        output_col='entropy'
    )
    
    entropy_pd = entropy.toPandas()
    assert 'entropy' in entropy_pd.columns
    assert len(entropy_pd) == len(sample_data)
    assert all(entropy_pd['entropy'] >= 0.0)

def test_sklearn_model_cross_val(sample_data):
    """Test SKLearnModel cross-validation."""
    model = SKLearnModel(HistGradientBoostingClassifier)
    
    # Convert to Spark DataFrame for cross_val_scores
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(sample_data)
    
    # Test cross-validation scores with 3-fold CV for small dataset
    scores = model.cross_val_scores(
        spark_df,
        vector_col='features',
        label_column='label',
        cv=3  # Use 3-fold CV for small dataset
    )
    
    assert len(scores) == 3
    assert all((scores >= 0.0) & (scores <= 1.0))

def test_spark_ml_model_properties():
    """Test SparkMLModel properties."""
    model = SparkMLModel(RandomForestClassifier, nan_fill=0.0)
    
    assert model.nan_fill == 0.0
    assert model.use_vectors
    assert not model.use_floats
    assert model._trained_model is None

def test_spark_ml_model_train_predict(spark_session, sample_spark_data, sample_data):
    """Test SparkMLModel training and prediction."""
    model = SparkMLModel(RandomForestClassifier)
    
    # Train the model
    trained_model = model.train(
        sample_spark_data,
        vector_col='features',
        label_column='label'
    )
    
    assert trained_model._trained_model is not None
    
    # Test prediction using positional arguments
    predictions = trained_model.predict(
        sample_spark_data,
        'features',
        'prediction'
    )
    
    predictions_pd = predictions.toPandas()
    assert 'prediction' in predictions_pd.columns
    assert len(predictions_pd) == len(sample_data)
    assert all(predictions_pd['prediction'].isin([0.0, 1.0]))

def test_spark_ml_model_prediction_conf(spark_session, sample_spark_data, sample_data):
    """Test SparkMLModel prediction confidence."""
    model = SparkMLModel(RandomForestClassifier)
    trained_model = model.train(
        sample_spark_data,
        vector_col='features',
        label_column='label'
    )
    
    # Test prediction confidence using positional arguments
    conf = trained_model.prediction_conf(
        sample_spark_data,
        'features',
        'confidence'
    )
    
    conf_pd = conf.toPandas()
    assert 'confidence' in conf_pd.columns
    assert len(conf_pd) == len(sample_data)
    assert all((conf_pd['confidence'] >= 0.0) & (conf_pd['confidence'] <= 1.0))

def test_spark_ml_model_entropy(spark_session, sample_spark_data, sample_data):
    """Test SparkMLModel entropy calculation."""
    model = SparkMLModel(RandomForestClassifier)
    trained_model = model.train(
        sample_spark_data,
        vector_col='features',
        label_column='label'
    )
    
    # Test entropy calculation using positional arguments
    entropy = trained_model.entropy(
        sample_spark_data,
        'features',
        'entropy'
    )
    
    entropy_pd = entropy.toPandas()
    assert 'entropy' in entropy_pd.columns
    assert len(entropy_pd) == len(sample_data)
    assert all(entropy_pd['entropy'] >= 0.0)
