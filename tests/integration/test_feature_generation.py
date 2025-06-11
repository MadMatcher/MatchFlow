import pytest
import pandas as pd
import numpy as np
from featurization import create_features, featurize
from tokenizer import AlphaNumericTokenizer, NumericTokenizer
from feature import ExactMatchFeature, EditDistanceFeature, JaccardFeature, TFIDFFeature, RelDiffFeature
from pyspark.sql import SparkSession
from featurization import BuildCache

# Create tokenizer instances to be reused across tests
alpha_numeric_tokenizer = AlphaNumericTokenizer()
numeric_tokenizer = NumericTokenizer()

@pytest.fixture
def sample_data():
    """Create sample pandas DataFrames for testing."""
    A = pd.DataFrame({
        '_id': [1, 2, 3],
        'name': ['Hello World', 'Python Code', 'Test Case'],
        'age': ['25', '30', '35'],
        'description': ['A test description', 'Another description', 'Final test']
    })
    B = pd.DataFrame({
        '_id': [4, 5, 6],
        'name': ['Hello World', 'Python Script', 'Test Example'],
        'age': ['25', '31', '35'],
        'description': ['A test description', 'Different text', 'Final test case']
    })
    return A, B

@pytest.fixture
def candidates():
    """Create candidate pairs for testing."""
    return pd.DataFrame({
        'id1_list': [[1], [1], [2]],
        'id2': [4, 5, 6]
    })

@pytest.fixture
def spark_session():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .master('local[*]') \
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \
        .getOrCreate()

def test_feature_creation_and_generation(sample_data, candidates):
    """Test feature creation and generation with pandas DataFrames."""
    A, B = sample_data
    
    # Create features
    features = [
        ExactMatchFeature('name', 'name'),
        EditDistanceFeature('name', 'name'),
        JaccardFeature('description', 'description', AlphaNumericTokenizer()),
        TFIDFFeature('description', 'description', AlphaNumericTokenizer())
    ]
    
    # Generate features
    result = featurize(features, A, B, candidates)
    
    # Verify output
    assert len(result) == len(candidates)
    assert 'features' in result.columns
    assert '_id' in result.columns
    
    # Get rows by ID pairs
    test_example_vs_python = result[result['id1'].isin([2]) & (result['id2'] == 6)].iloc[0]
    hello_world_vs_hello = result[result['id1'].isin([1]) & (result['id2'] == 4)].iloc[0]
    python_script_vs_hello = result[result['id1'].isin([1]) & (result['id2'] == 5)].iloc[0]
    
    # Verify feature values for each pair
    # Test Example vs Python Code
    feature_vector = test_example_vs_python['features']
    assert len(feature_vector) == len(features)
    assert feature_vector[0] == pytest.approx(0.0)  # No exact match
    assert feature_vector[1] == pytest.approx(0.16666667, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(0.0, rel=0.1)  # No token match
    assert feature_vector[3] == pytest.approx(0.0, rel=0.1)  # No TFIDF match
    
    # Hello World vs Hello World
    feature_vector = hello_world_vs_hello['features']
    assert feature_vector[0] == pytest.approx(1.0)  # Exact match
    assert feature_vector[1] == pytest.approx(0.8181818, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(1.0, rel=0.1)  # Token match (identical tokens)
    assert feature_vector[3] == pytest.approx(1.0, rel=0.1)  # TFIDF match (identical text)
    
    # Python Script vs Hello World
    feature_vector = python_script_vs_hello['features']
    assert feature_vector[0] == pytest.approx(0.0)  # No exact match
    assert feature_vector[1] == pytest.approx(0.23076923, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(0.0, rel=0.1)  # No token match
    assert feature_vector[3] == pytest.approx(0.0, rel=0.1)  # No TFIDF match

def test_feature_generation_with_null_values(sample_data, candidates):
    """Test feature generation with null values."""
    A, B = sample_data
    
    # Add null values
    A.loc[0, 'name'] = None
    B.loc[0, 'name'] = None
    
    # Create features
    features = [
        ExactMatchFeature('name', 'name'),
        EditDistanceFeature('name', 'name'),
        JaccardFeature('description', 'description', AlphaNumericTokenizer())
    ]
    
    # Generate features
    result = featurize(features, A, B, candidates)
    
    # Verify output
    assert len(result) == len(candidates)
    assert 'features' in result.columns
    assert '_id' in result.columns
    
    # Get row with null values by ID
    null_pair = result[result['id1'].isin([1]) & (result['id2'] == 4)].iloc[0]
    
    # Verify null handling
    feature_vector = null_pair['features']
    assert len(feature_vector) == len(features)
    assert np.isnan(feature_vector[0])  # Exact match with null
    assert np.isnan(feature_vector[1])  # Edit distance with null
    assert feature_vector[2] == pytest.approx(1.0, rel=0.1)  # Token match (identical tokens)

def test_large_scale_feature_generation(sample_data, candidates, spark_session):
    """Test feature generation with large-scale data using Spark."""
    A, B = sample_data
    
    # Create larger datasets by duplicating
    n_records = 1000
    A_large = pd.concat([A] * (n_records // len(A) + 1)).iloc[:n_records]
    B_large = pd.concat([B] * (n_records // len(B) + 1)).iloc[:n_records]
    
    # Create features
    features = [
        ExactMatchFeature('name', 'name'),
        EditDistanceFeature('name', 'name'),
        JaccardFeature('description', 'description', AlphaNumericTokenizer())
    ]
    
    # Convert to Spark DataFrames
    A_spark = spark_session.createDataFrame(A_large)
    B_spark = spark_session.createDataFrame(B_large)
    candidates_spark = spark_session.createDataFrame(candidates)
    
    # Generate features
    result = featurize(features, A_spark, B_spark, candidates_spark)
    
    # Verify output
    assert len(result) == len(candidates)
    assert 'features' in result.columns
    assert '_id' in result.columns
    
    # Get specific pairs by ID
    test_example_vs_python = result[result['id1'].isin([2]) & (result['id2'] == 6)].iloc[0]
    hello_world_vs_hello = result[result['id1'].isin([1]) & (result['id2'] == 4)].iloc[0]
    python_script_vs_hello = result[result['id1'].isin([1]) & (result['id2'] == 5)].iloc[0]
    
    # Verify feature values for each pair
    # Test Example vs Python Code
    feature_vector = test_example_vs_python['features']
    assert feature_vector[0] == pytest.approx(0.0)  # No exact match
    assert feature_vector[1] == pytest.approx(0.16666667, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(0.0, rel=0.1)  # No token match
    
    # Hello World vs Hello World
    feature_vector = hello_world_vs_hello['features']
    assert feature_vector[0] == pytest.approx(1.0)  # Exact match
    assert feature_vector[1] == pytest.approx(0.8181818, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(1.0, rel=0.1)  # Token match (identical tokens)
    
    # Python Script vs Hello World
    feature_vector = python_script_vs_hello['features']
    assert feature_vector[0] == pytest.approx(0.0)  # No exact match
    assert feature_vector[1] == pytest.approx(0.23076923, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(0.0, rel=0.1)  # No token match

def test_feature_generation_with_custom_features(sample_data, candidates):
    """Test feature generation with custom feature combinations."""
    A, B = sample_data
    
    # Create custom features
    features = [
        ExactMatchFeature('name', 'name'),
        EditDistanceFeature('name', 'name'),
        JaccardFeature('description', 'description', AlphaNumericTokenizer()),
        TFIDFFeature('description', 'description', AlphaNumericTokenizer()),
        RelDiffFeature('age', 'age')
    ]
    
    # Generate features
    result = featurize(features, A, B, candidates)
    
    # Verify output
    assert len(result) == len(candidates)
    assert 'features' in result.columns
    assert '_id' in result.columns
    
    # Get specific pairs by ID
    test_example_vs_python = result[result['id1'].isin([2]) & (result['id2'] == 6)].iloc[0]
    hello_world_vs_hello = result[result['id1'].isin([1]) & (result['id2'] == 4)].iloc[0]
    python_script_vs_hello = result[result['id1'].isin([1]) & (result['id2'] == 5)].iloc[0]
    
    # Verify feature values for each pair
    # Test Example vs Python Code
    feature_vector = test_example_vs_python['features']
    assert len(feature_vector) == len(features)
    assert feature_vector[0] == pytest.approx(0.0)  # No exact match
    assert feature_vector[1] == pytest.approx(0.16666667, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(0.0, rel=0.1)  # No token match
    assert feature_vector[3] == pytest.approx(0.0, rel=0.1)  # No TFIDF match
    assert feature_vector[4] == pytest.approx(0.14285715, rel=0.1)  # Age difference (|30-35|/35 = 5/35)
    
    # Hello World vs Hello World
    feature_vector = hello_world_vs_hello['features']
    assert feature_vector[0] == pytest.approx(1.0)  # Exact match
    assert feature_vector[1] == pytest.approx(0.8181818, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(1.0, rel=0.1)  # Token match (identical tokens)
    assert feature_vector[3] == pytest.approx(1.0, rel=0.1)  # TFIDF match (identical text)
    assert feature_vector[4] == pytest.approx(0.0, rel=0.1)  # Same age
    
    # Python Script vs Hello World
    feature_vector = python_script_vs_hello['features']
    assert feature_vector[0] == pytest.approx(0.0)  # No exact match
    assert feature_vector[1] == pytest.approx(0.23076923, rel=0.01)  # Edit distance
    assert feature_vector[2] == pytest.approx(0.0, rel=0.1)  # No token match
    assert feature_vector[3] == pytest.approx(0.0, rel=0.1)  # No TFIDF match
    assert feature_vector[4] == pytest.approx(0.2, rel=0.1)  # Age difference
