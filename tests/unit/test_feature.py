import pytest
import pandas as pd
import numpy as np
from feature import (
    ExactMatchFeature,
    EditDistanceFeature,
    SmithWatermanFeature,
    NeedlemanWunschFeature,
    RelDiffFeature,
    JaccardFeature,
    MongeElkanFeature,
    TFIDFFeature
)
from tokenizer import AlphaNumericTokenizer
from featurization import BuildCache
from pyspark.sql import SparkSession

@pytest.fixture
def sample_data():
    """Create sample DataFrames A and B for testing."""
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
def spark_session():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .master('local[*]') \
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \
        .getOrCreate()

@pytest.fixture
def build_cache():
    """Create a BuildCache instance for testing."""
    return BuildCache()

def test_exact_match_feature_lifecycle(sample_data, build_cache):
    """Test the lifecycle of ExactMatchFeature."""
    A, B = sample_data
    feature = ExactMatchFeature('name', 'name')
    
    # Build phase
    feature.build(A, B, build_cache)
    
    # Preprocess phase
    A_processed = feature.preprocess(A, True)
    B_processed = feature.preprocess(B, False)
    
    # Compute phase
    result = feature(B_processed.iloc[0], A_processed)
    
    # Verify results
    assert result.iloc[0] == 1.0  # Exact match
    assert result.iloc[1] == 0.0  # Different strings
    assert result.iloc[2] == 0.0  # Different strings

def test_edit_distance_feature_lifecycle(sample_data, build_cache):
    """Test the lifecycle of EditDistanceFeature."""
    A, B = sample_data
    feature = EditDistanceFeature('name', 'name')
    
    # Build phase
    feature.build(A, B, build_cache)
    
    # Preprocess phase
    A_processed = feature.preprocess(A, True)
    B_processed = feature.preprocess(B, False)
    
    # Compute phase - comparing B[0] with all rows in A (B[0] is "Hello World")
    result = feature(B_processed.iloc[0], A_processed)
    
    # Find rows in A by their name values
    hello_world_idx = A[A['name'] == 'Hello World'].index[0]
    python_code_idx = A[A['name'] == 'Python Code'].index[0]
    test_case_idx = A[A['name'] == 'Test Case'].index[0]
    
    assert result.loc[hello_world_idx] == pytest.approx(0.8181818, rel=0.01)  # Hello World vs Hello World
    assert result.loc[python_code_idx] == pytest.approx(0.18181818, rel=0.01)  # Hello World vs Python Code
    assert result.loc[test_case_idx] == pytest.approx(0.18181818, rel=0.01)  # Hello World vs Test Case

def test_rel_diff_feature_lifecycle(sample_data, build_cache):
    """Test the lifecycle of RelDiffFeature."""
    A, B = sample_data
    feature = RelDiffFeature('age', 'age')
    
    # Build phase
    feature.build(A, B, build_cache)
    
    # Preprocess phase
    A_processed = feature.preprocess(A, True)
    B_processed = feature.preprocess(B, False)
    
    # Compute phase - comparing B[0] with all rows in A (B[0] is age "25")
    result = feature(B_processed.iloc[0], A_processed)
    
    # Find rows in A by their age values
    age_25_idx = A[A['age'] == '25'].index[0]
    age_30_idx = A[A['age'] == '30'].index[0]
    age_35_idx = A[A['age'] == '35'].index[0]
    
    assert result.loc[age_25_idx] == pytest.approx(0.0)  # 25 vs 25
    assert result.loc[age_30_idx] == pytest.approx(0.167, rel=0.01)  # 25 vs 30
    assert result.loc[age_35_idx] == pytest.approx(0.286, rel=0.01)  # 25 vs 35

def test_token_feature_lifecycle(sample_data, build_cache):
    """Test the lifecycle of TokenFeature (Jaccard)."""
    A, B = sample_data
    tokenizer = AlphaNumericTokenizer()
    feature = JaccardFeature('description', 'description', tokenizer)
    
    # Build phase
    feature.build(A, B, build_cache)
    
    # Preprocess phase
    A_processed = feature.preprocess(A, True)
    B_processed = feature.preprocess(B, False)
    
    # Compute phase - comparing B[0] with all rows in A (B[0] is "A test description")
    result = feature(B_processed.iloc[0], A_processed)
    
    # Find rows in A by their description values
    test_desc_idx = A[A['description'] == 'A test description'].index[0]
    another_desc_idx = A[A['description'] == 'Another description'].index[0]
    final_test_idx = A[A['description'] == 'Final test'].index[0]
    
    assert result.loc[test_desc_idx] == pytest.approx(1.0, rel=0.1)  # A test description vs A test description
    assert result.loc[another_desc_idx] == pytest.approx(0.25, rel=0.1)  # A test description vs Another description
    assert result.loc[final_test_idx] == pytest.approx(0.25, rel=0.1)  # A test description vs Final test (shares "test" token)

def test_perfect_hash_function():
    """Test PerfectHashFunction validation."""
    from utils import PerfectHashFunction
    
    # Test with duplicate keys
    keys = ['a', 'a', 'b']
    with pytest.raises(ValueError, match='keys must be unique'):
        PerfectHashFunction.create_for_keys(keys)
    
    # Test with unique keys
    keys = ['a', 'b', 'c']
    hash_func, hashes = PerfectHashFunction.create_for_keys(keys)
    assert len(hashes) == len(keys)
    
    # Test hash function
    for key in keys:
        assert hash_func.hash(key) in hashes
