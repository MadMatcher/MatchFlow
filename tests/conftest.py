import pytest
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    spark = (SparkSession.builder
            .master("local[2]")
            .appName("pytest-spark")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'age': [25, 30, 35, 40, 45],
        'score': [0.8, 0.6, 0.9, 0.7, 0.5]
    })

@pytest.fixture
def sample_spark_dataframe(spark):
    """Create a sample Spark DataFrame for testing."""
    data = [
        (1, "John", 25, 0.8),
        (2, "Jane", 30, 0.6),
        (3, "Bob", 35, 0.9),
        (4, "Alice", 40, 0.7),
        (5, "Charlie", 45, 0.5)
    ]
    return spark.createDataFrame(data, ["id", "name", "age", "score"])

@pytest.fixture
def sample_feature_vectors():
    """Create sample feature vectors for testing."""
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])

@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return np.array([0, 1, 0])

@pytest.fixture
def mock_labeler(mocker):
    """Create a mock labeler for testing."""
    mock = mocker.Mock()
    mock.return_value = 1.0
    return mock

@pytest.fixture
def sample_gold_pairs():
    """Create sample gold standard pairs for testing."""
    return {
        (1, 101),
        (2, 102),
        (3, 103)
    }

@pytest.fixture
def sample_tokenizer():
    """Create a sample tokenizer for testing."""
    class MockTokenizer:
        def tokenize(self, text):
            return text.split() if text else []
    return MockTokenizer()

@pytest.fixture
def sample_similarity_function():
    """Create a sample similarity function for testing."""
    def mock_similarity(x, y):
        return 1.0 if x == y else 0.0
    return mock_similarity

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    os.environ["PYTHONPATH"] = os.getcwd()
    yield
    # Cleanup if needed
