import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from utils import (
    type_check,
    type_check_iterable,
    is_null,
    persisted,
    is_persisted,
    get_logger,
    repartition_df,
    SparseVec,
    PerfectHashFunction
)

def test_type_check():
    """Test type checking utility."""
    # Test valid type
    type_check(5, "test_var", int)
    
    # Test invalid type
    with pytest.raises(TypeError):
        type_check("5", "test_var", int)

def test_type_check_iterable():
    """Test iterable type checking utility."""
    # Test valid types
    type_check_iterable([1, 2, 3], "test_list", list, int)
    type_check_iterable((1, 2, 3), "test_tuple", tuple, int)
    
    # Test invalid container type
    with pytest.raises(TypeError):
        type_check_iterable([1, 2, 3], "test_list", tuple, int)
    
    # Test invalid element type
    with pytest.raises(TypeError):
        type_check_iterable([1, "2", 3], "test_list", list, int)

def test_is_null():
    """Test null checking utility."""
    # Test various null values
    assert is_null(None)
    assert is_null(np.nan)
    assert is_null(pd.NA)
    assert is_null(pd.NaT)
    
    # Test non-null values
    assert not is_null(0)
    assert not is_null("")
    assert not is_null([])
    assert not is_null({})

@pytest.mark.spark
def test_persisted(spark):
    """Test dataframe persistence context manager."""
    df = spark.createDataFrame([(1,), (2,), (3,)], ["value"])
    
    with persisted(df) as persisted_df:
        assert is_persisted(persisted_df)
    
    assert not is_persisted(df)

def test_get_logger():
    """Test logger creation."""
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == 10  # DEBUG level

@pytest.mark.spark
def test_repartition_df(spark):
    """Test dataframe repartitioning."""
    df = spark.createDataFrame([(i,) for i in range(1000)], ["value"])
    
    # Test repartitioning by size
    repartitioned = repartition_df(df, part_size=100)
    assert repartitioned.rdd.getNumPartitions() <= 10
    
    # Test repartitioning by column
    repartitioned = repartition_df(df, part_size=100, by="value")
    assert repartitioned.rdd.getNumPartitions() <= 10

def test_sparse_vec():
    """Test sparse vector operations."""
    size = 10
    indexes = np.array([1, 3, 5], dtype=np.int32)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    vec1 = SparseVec(size, indexes, values)
    vec2 = SparseVec(size, indexes, values)
    
    # Test properties
    assert vec1._size == size
    assert np.array_equal(vec1.indexes, indexes)
    assert np.array_equal(vec1.values, values)
    
    # Test dot product
    assert vec1.dot(vec2) == pytest.approx(0.14)  # 0.1^2 + 0.2^2 + 0.3^2

def test_perfect_hash_function():
    """Test PerfectHashFunction validation."""
    # Test with duplicate keys
    keys = ['a', 'a', 'b']
    with pytest.raises(ValueError, match='keys must be unique'):
        PerfectHashFunction.create_for_keys(keys)
    
    # Test with unique keys including one that might hash to zero
    keys = ['a', 'b', '\x00']
    hash_func, hashes = PerfectHashFunction.create_for_keys(keys)
    assert len(hashes) == len(keys)
    for key in keys:
        assert hash_func.hash(key) in hashes

@pytest.mark.benchmark
def test_perfect_hash_function_performance(benchmark):
    """Benchmark perfect hash function performance."""
    keys = [f"key{i}" for i in range(1000)]
    
    def create_hash():
        return PerfectHashFunction.create_for_keys(keys)
    
    result = benchmark(create_hash)
    assert len(result[1]) == len(keys)
