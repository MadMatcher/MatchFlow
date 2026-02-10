"""Tests for MatchFlow._internal.utils module.

This module provides utility functions, compression, hashing, and math helpers.
"""
import pytest
import numpy as np
import pandas as pd

import MatchFlow._internal.utils as utils


class TestTypeChecks:
    """Tests for type checking helpers."""

    def test_type_check(self):
        """Verify type_check enforces expected types."""
        utils.type_check(5, 'var', int)
        utils.type_check('hello', 'var', str)
        utils.type_check([1, 2], 'var', list)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check(5, 'var', str)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check('hello', 'var', int)

    def test_type_check_iterable(self):
        """Validate type_check_iterable checks element types."""
        utils.type_check_iterable([1, 2, 3], 'var', list, int)
        utils.type_check_iterable((1, 2, 3), 'var', tuple, int)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check_iterable([1, 2, '3'], 'var', list, int)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check_iterable('not a list', 'var', list, int)


class TestNullHelpers:
    """Tests for null checking utilities."""

    def test_is_null(self):
        """Ensure is_null returns booleans for scalars."""
        assert utils.is_null(None) is True
        assert utils.is_null(np.nan) is True
        assert utils.is_null(pd.NA) is True
        assert utils.is_null(5) is False
        assert utils.is_null('hello') is False
        assert utils.is_null(0) is False
        assert utils.is_null('') is False


class TestPersistenceHelpers:
    """Tests for persistence utilities."""

    def test_persisted_context(self, spark_session):
        """Check persisted context manager persists and unpersists."""
        df = spark_session.createDataFrame([
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
        ])

        assert not utils.is_persisted(df)

        with utils.persisted(df) as persisted_df:
            assert utils.is_persisted(persisted_df)

        assert not utils.is_persisted(df)

    def test_persisted_context_none(self):
        """Test persisted context manager with None."""
        with utils.persisted(None) as df:
            assert df is None

    def test_is_persisted(self, spark_session):
        """Assert is_persisted reflects StorageLevel flags."""
        from pyspark import StorageLevel

        df = spark_session.createDataFrame([
            {"id": 1, "value": "a"},
        ])

        assert not utils.is_persisted(df)

        df = df.persist(StorageLevel.MEMORY_ONLY)
        assert utils.is_persisted(df)

        df.unpersist()
        assert not utils.is_persisted(df)


class TestLoggingAndRepartition:
    """Tests for logging and repartition helpers."""

    def test_get_logger(self):
        """Ensure get_logger returns configured logger."""
        logger = utils.get_logger('test_module')

        assert logger is not None
        assert logger.name == 'test_module'
        assert logger.level <= utils.logging.DEBUG

    def test_repartition_df(self, spark_session):
        """Confirm repartition_df returns expected partition count."""
        df = spark_session.createDataFrame([
            {"id": i, "value": f"val_{i}"} for i in range(100)
        ])

        repartitioned = utils.repartition_df(df, part_size=10)

        assert repartitioned is not None
        assert repartitioned.count() == 100

        repartitioned_by = utils.repartition_df(df, part_size=10, by='id')

        assert repartitioned_by is not None
        assert repartitioned_by.count() == 100


class TestSparseVec:
    """Tests for SparseVec operations."""

    def test_dot(self):
        """Validate SparseVec.dot uses underlying _sparse_dot correctly."""
        size = 10
        indexes1 = np.array([0, 2, 5], dtype=np.int32)
        values1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        indexes2 = np.array([0, 2, 7], dtype=np.int32)
        values2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        vec1 = utils.SparseVec(size, indexes1, values1)
        vec2 = utils.SparseVec(size, indexes2, values2)

        result = vec1.dot(vec2)

        assert abs(result - 8.0) < 1e-6

    def test_sparse_vec_properties(self):
        """Test SparseVec properties."""
        size = 10
        indexes = np.array([0, 2, 5], dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        vec = utils.SparseVec(size, indexes, values)

        assert np.array_equal(vec.indexes, indexes)
        assert np.array_equal(vec.values, values)
        assert vec.indexes.dtype == np.int32
        assert vec.values.dtype == np.float32


class TestPerfectHashFunction:
    """Tests for PerfectHashFunction behavior."""

    def test_create_for_keys(self):
        """Ensure create_for_keys produces unique hashes."""
        keys = ['key1', 'key2', 'key3', 'key4', 'key5']

        hash_func, hash_vals = utils.PerfectHashFunction.create_for_keys(keys)

        assert hash_func is not None
        assert len(hash_vals) == len(keys)

        hashes = [hash_func.hash(k) for k in keys]
        assert len(set(hashes)) == len(keys)

    def test_create_for_keys_duplicates(self):
        """Test create_for_keys raises error for duplicate keys."""
        keys = ['key1', 'key2', 'key1']

        with pytest.raises(ValueError, match='keys must be unique'):
            utils.PerfectHashFunction.create_for_keys(keys)

    def test_hash(self):
        """Test PerfectHashFunction hash method."""
        hash_func = utils.PerfectHashFunction(seed=42)

        hash1 = hash_func.hash('test')
        hash2 = hash_func.hash('test')

        assert hash1 == hash2

        hash3 = hash_func.hash('different')
        assert hash3 != hash1

    def test_init_with_seed(self):
        """Test PerfectHashFunction initialization with seed."""
        hash_func1 = utils.PerfectHashFunction(seed=42)
        hash_func2 = utils.PerfectHashFunction(seed=42)

        hash1 = hash_func1.hash('test')
        hash2 = hash_func2.hash('test')

        assert hash1 == hash2


class TestTrainingDataStreaming:
    """Tests for training data streaming utilities."""

    def test_save_training_data_streaming(self, temp_dir):
        """Write batch to parquet and confirm file contents."""
        parquet_file = temp_dir / 'training_data.parquet'

        new_batch = pd.DataFrame({
            '_id': [1, 2],
            'id1': [10, 11],
            'id2': [20, 21],
            'feature_vectors': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0],
        })

        utils.save_training_data_streaming(new_batch, str(parquet_file))

        assert parquet_file.exists()

        loaded = pd.read_parquet(parquet_file)
        assert len(loaded) == 2
        assert list(loaded['_id']) == [1, 2]

    def test_save_training_data_streaming_append(self, temp_dir):
        """Test appending to existing file."""
        parquet_file = temp_dir / 'training_data.parquet'

        batch1 = pd.DataFrame({
            '_id': [1, 2],
            'id1': [10, 11],
            'id2': [20, 21],
            'feature_vectors': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0],
        })

        utils.save_training_data_streaming(batch1, str(parquet_file))

        batch2 = pd.DataFrame({
            '_id': [3, 4],
            'id1': [12, 13],
            'id2': [22, 23],
            'feature_vectors': [[0.5, 0.6], [0.7, 0.8]],
            'label': [1.0, 0.0],
        })

        utils.save_training_data_streaming(batch2, str(parquet_file))

        loaded = pd.read_parquet(parquet_file)
        assert len(loaded) == 4
        assert list(loaded['_id']) == [1, 2, 3, 4]

    def test_load_training_data_streaming(self, temp_dir):
        """Load parquet and validate schema conversion."""
        parquet_file = temp_dir / 'training_data.parquet'

        batch = pd.DataFrame({
            '_id': [1, 2],
            'id1': [10, 11],
            'id2': [20, 21],
            'feature_vectors': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0],
        })

        batch.to_parquet(parquet_file, index=False)

        loaded = utils.load_training_data_streaming(str(parquet_file))

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded['_id'].dtype == 'int64'
        assert loaded['id1'].dtype == 'int64'
        assert loaded['id2'].dtype == 'int64'
        assert loaded['label'].dtype == 'float64'
        assert isinstance(loaded['feature_vectors'].iloc[0], list)

    def test_load_training_data_streaming_nonexistent(self, temp_dir):
        """Test loading non-existent file."""
        parquet_file = temp_dir / 'nonexistent.parquet'

        loaded = utils.load_training_data_streaming(str(parquet_file))

        assert loaded is None


class TestAdjustHelpers:
    """Tests for adjustment helper functions."""

    def test_adjust_iterations_for_existing_data(self):
        """Verify remaining iteration calculation."""
        result = utils.adjust_iterations_for_existing_data(0, 100, 5, 10)
        assert result == 10

        result = utils.adjust_iterations_for_existing_data(5, 100, 5, 10)
        assert result == 9

        result = utils.adjust_iterations_for_existing_data(10, 100, 5, 10)
        assert result == 8

        result = utils.adjust_iterations_for_existing_data(60, 100, 5, 10)
        assert result == 0

    def test_adjust_labeled_examples_for_existing_data(self):
        """Confirm remaining examples calculation respects bounds."""
        result = utils.adjust_labeled_examples_for_existing_data(0, 100)
        assert result == 100

        result = utils.adjust_labeled_examples_for_existing_data(30, 100)
        assert result == 70

        result = utils.adjust_labeled_examples_for_existing_data(100, 100)
        assert result == 0

        result = utils.adjust_labeled_examples_for_existing_data(150, 100)
        assert result == 0


class TestConvertArraysForSpark:
    """Tests for convert_arrays_for_spark function."""

    def test_convert_arrays_for_spark_numpy_arrays(self):
        """Test convert_arrays_for_spark with numpy arrays."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'features': [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])],
            'values': [10, 20, 30]
        })
        
        result = utils.convert_arrays_for_spark(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert isinstance(result['features'].iloc[0], list)
        assert result['features'].iloc[0] == [1.0, 2.0]
        assert result['values'].iloc[0] == 10

    def test_convert_arrays_for_spark_lists(self):
        """Test convert_arrays_for_spark with lists."""
        df = pd.DataFrame({
            'id': [1, 2],
            'features': [[1.0, 2.0], [3.0, 4.0]],
            'values': [10, 20]
        })
        
        result = utils.convert_arrays_for_spark(df)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result['features'].iloc[0], list)

    def test_convert_arrays_for_spark_empty_dataframe(self):
        """Test convert_arrays_for_spark with empty DataFrame."""
        df = pd.DataFrame()
        
        result = utils.convert_arrays_for_spark(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_convert_arrays_for_spark_non_dataframe(self):
        """Test convert_arrays_for_spark with non-DataFrame input."""
        result = utils.convert_arrays_for_spark(None)
        assert result is None
        
        result = utils.convert_arrays_for_spark([1, 2, 3])
        assert result == [1, 2, 3]

    def test_convert_arrays_for_spark_mixed_types(self):
        """Test convert_arrays_for_spark with mixed array and non-array columns."""
        df = pd.DataFrame({
            'id': [1, 2],
            'features': [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            'name': ['a', 'b'],
            'count': [5, 6]
        })
        
        result = utils.convert_arrays_for_spark(df)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result['features'].iloc[0], list)
        assert result['name'].iloc[0] == 'a'
        assert result['count'].iloc[0] == 5


class TestCheckTables:
    """Tests for check_tables function."""

    def test_check_tables_pandas_valid(self):
        """Test check_tables with valid pandas DataFrames."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})

        # Should not raise
        utils.check_tables(table_a, table_b)

    def test_check_tables_logs_success_on_valid(self, caplog):
        """Test check_tables logs that formats are correct when validation passes."""
        import logging
        caplog.set_level(logging.WARNING)
        table_a = pd.DataFrame({'_id': [1, 2], 'value': ['a', 'b']})
        table_b = pd.DataFrame({'_id': [3, 4], 'value': ['c', 'd']})
        utils.check_tables(table_a, table_b)
        assert "check_tables" in caplog.text and "formats are correct" in caplog.text

    def test_check_tables_pandas_missing_id_a(self):
        """Test check_tables raises error when table_a missing '_id'."""
        table_a = pd.DataFrame({'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        
        with pytest.raises(ValueError, match="table_a must have the column '_id'"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_pandas_missing_id_b(self):
        """Test check_tables raises error when table_b missing '_id'."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'value': ['d', 'e', 'f']})
        
        with pytest.raises(ValueError, match="table_b must have the column '_id'"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_pandas_non_unique_id_a(self):
        """Test check_tables raises error when table_a '_id' not unique."""
        table_a = pd.DataFrame({'_id': [1, 2, 2], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        
        with pytest.raises(ValueError, match="table_a '_id' column must be unique"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_pandas_non_unique_id_b(self):
        """Test check_tables raises error when table_b '_id' not unique."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 5], 'value': ['d', 'e', 'f']})
        
        with pytest.raises(ValueError, match="table_b '_id' column must be unique"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_spark_valid(self, spark_session):
        """Test check_tables with valid Spark DataFrames."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'},
            {'_id': 3, 'value': 'c'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'},
            {'_id': 6, 'value': 'f'}
        ])
        
        # Should not raise
        utils.check_tables(table_a, table_b)

    def test_check_tables_spark_missing_id_a(self, spark_session):
        """Test check_tables raises error when Spark table_a missing '_id'."""
        table_a = spark_session.createDataFrame([
            {'value': 'a'},
            {'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        
        with pytest.raises(ValueError, match="table_a must have the column '_id'"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_spark_missing_id_b(self, spark_session):
        """Test check_tables raises error when Spark table_b missing '_id'."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'value': 'd'},
            {'value': 'e'}
        ])
        
        with pytest.raises(ValueError, match="table_b must have the column '_id'"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_spark_non_unique_id_a(self, spark_session):
        """Test check_tables raises error when Spark table_a '_id' not unique."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'},
            {'_id': 2, 'value': 'c'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        
        with pytest.raises(ValueError, match="table_a '_id' column must be unique"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_spark_non_unique_id_b(self, spark_session):
        """Test check_tables raises error when Spark table_b '_id' not unique."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'},
            {'_id': 5, 'value': 'f'}
        ])
        
        with pytest.raises(ValueError, match="table_b '_id' column must be unique"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_mixed_types_pandas_spark(self, spark_session):
        """Test check_tables raises error when mixing pandas and Spark DataFrames."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        
        with pytest.raises(ValueError, match="table_a and table_b must both be either pandas DataFrames or Spark DataFrames"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_mixed_types_spark_pandas(self, spark_session):
        """Test check_tables raises error when mixing Spark and pandas DataFrames."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        
        with pytest.raises(ValueError, match="table_a and table_b must both be either pandas DataFrames or Spark DataFrames"):
            utils.check_tables(table_a, table_b)

    def test_check_tables_invalid_type(self):
        """Test check_tables raises error with non-DataFrame types."""
        table_a = [1, 2, 3]
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        
        with pytest.raises(ValueError, match="table_a and table_b must both be either pandas DataFrames or Spark DataFrames"):
            utils.check_tables(table_a, table_b)


class TestCheckCandidates:
    """Tests for check_candidates function."""

    def test_check_candidates_pandas_valid(self):
        """Test check_candidates with valid pandas DataFrame."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': [[1, 2], [3]]
        })
        
        # Should not raise
        utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_logs_success_on_valid(self, caplog):
        """Test check_candidates logs that formats are correct when validation passes."""
        import logging
        caplog.set_level(logging.WARNING)
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({'id2': [4, 5], 'id1_list': [[1, 2], [3]]})
        utils.check_candidates(candidates, table_a, table_b)
        assert "check_candidates" in caplog.text and "formats are correct" in caplog.text

    def test_check_candidates_pandas_missing_id2(self):
        """Test check_candidates raises error when missing 'id2' column."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({
            'id1_list': [[1, 2], [3]]
        })
        
        with pytest.raises(ValueError, match="candidates must have the column 'id2'"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_pandas_missing_id1_list(self):
        """Test check_candidates raises error when missing 'id1_list' column."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({
            'id2': [4, 5]
        })
        
        with pytest.raises(ValueError, match="candidates must have the column 'id1_list'"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_pandas_non_unique_id2(self):
        """Test check_candidates raises error when 'id2' not unique."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({
            'id2': [4, 4],
            'id1_list': [[1, 2], [3]]
        })
        
        with pytest.raises(ValueError, match="candidates 'id2' column must be unique"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_pandas_non_list_id1_list(self):
        """Test check_candidates raises error when 'id1_list' not a list."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': ['not a list', [3]]
        })
        
        with pytest.raises(ValueError, match="candidates 'id1_list' column must be a list of ids"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_pandas_invalid_id1_in_list(self):
        """Test check_candidates raises error when 'id1_list' contains invalid ids."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': [[1, 2], [99]]  # 99 not in table_a
        })
        
        with pytest.raises(ValueError, match="candidates 'id1_list' column must only contain ids that are present in the table_a '_id' column"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_pandas_invalid_id2(self):
        """Test check_candidates raises error when 'id2' contains invalid ids."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = pd.DataFrame({
            'id2': [99],  # 99 not in table_b
            'id1_list': [[1, 2]]
        })
        
        with pytest.raises(ValueError, match="candidates 'id2' column must only contain ids that are present in the table_b '_id' column"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_spark_valid(self, spark_session):
        """Test check_candidates with valid Spark DataFrame."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'},
            {'_id': 3, 'value': 'c'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'},
            {'_id': 6, 'value': 'f'}
        ])
        candidates = spark_session.createDataFrame([
            {'id2': 4, 'id1_list': [1, 2]},
            {'id2': 5, 'id1_list': [3]}
        ])
        
        # Should not raise
        utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_spark_missing_id2(self, spark_session):
        """Test check_candidates raises error when Spark DataFrame missing 'id2'."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        candidates = spark_session.createDataFrame([
            {'id1_list': [1, 2]}
        ])
        
        with pytest.raises(ValueError, match="candidates must have the column 'id2'"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_spark_missing_id1_list(self, spark_session):
        """Test check_candidates raises error when Spark DataFrame missing 'id1_list'."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        candidates = spark_session.createDataFrame([
            {'id2': 4}
        ])
        
        with pytest.raises(ValueError, match="candidates must have the column 'id1_list'"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_spark_non_unique_id2(self, spark_session):
        """Test check_candidates raises error when Spark 'id2' not unique."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        candidates = spark_session.createDataFrame([
            {'id2': 4, 'id1_list': [1, 2]},
            {'id2': 4, 'id1_list': [3]}
        ])
        
        with pytest.raises(ValueError, match="candidates 'id2' column must be unique"):
            utils.check_candidates(candidates, table_a, table_b)

    def test_check_candidates_invalid_type(self):
        """Test check_candidates raises error with non-DataFrame type."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        candidates = [1, 2, 3]
        
        with pytest.raises(ValueError, match="candidates must be a pandas DataFrame or Spark DataFrame"):
            utils.check_candidates(candidates, table_a, table_b)


class TestCheckLabeledData:
    """Tests for check_labeled_data function."""

    def test_check_labeled_data_pandas_valid(self):
        """Test check_labeled_data with valid pandas DataFrame."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': [[1, 2], [3]],
            'labels': [[0.5, 0.6], [0.7]]
        })
        
        # Should not raise
        utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_logs_success_on_valid(self, caplog):
        """Test check_labeled_data logs that formats are correct when validation passes."""
        import logging
        caplog.set_level(logging.WARNING)
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': [[1, 2], [3]],
            'labels': [[0.5, 0.6], [0.7]]
        })
        utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')
        assert "check_labeled_data" in caplog.text and "formats are correct" in caplog.text

    def test_check_labeled_data_pandas_missing_id2(self):
        """Test check_labeled_data raises error when missing 'id2' column."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id1_list': [[1, 2], [3]],
            'labels': [[0.5, 0.6], [0.7]]
        })
        
        with pytest.raises(ValueError, match="candidates must have the column 'id2'"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_pandas_missing_id1_list(self):
        """Test check_labeled_data raises error when missing 'id1_list' column."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 5],
            'labels': [[0.5, 0.6], [0.7]]
        })
        
        with pytest.raises(ValueError, match="candidates must have the column 'id1_list'"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_pandas_missing_label_column(self):
        """Test check_labeled_data raises error when missing label column."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': [[1, 2], [3]]
        })
        
        with pytest.raises(KeyError):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_pandas_non_unique_id2(self):
        """Test check_labeled_data raises error when 'id2' not unique."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 4],
            'id1_list': [[1, 2], [3]],
            'labels': [[0.5, 0.6], [0.7]]
        })
        
        with pytest.raises(ValueError, match="candidates 'id2' column must be unique"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_pandas_non_list_id1_list(self):
        """Test check_labeled_data raises error when 'id1_list' not a list."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': ['not a list', [3]],
            'labels': [[0.5, 0.6], [0.7]]
        })
        
        with pytest.raises(ValueError, match="candidates 'id1_list' column must be a list of ids"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_pandas_invalid_id1_in_list(self):
        """Test check_labeled_data raises error when 'id1_list' contains invalid ids."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': [[1, 2], [99]],  # 99 not in table_a
            'labels': [[0.5, 0.6], [0.7]]
        })
        
        with pytest.raises(ValueError, match="candidates 'id1_list' column must only contain ids that are present in the table_a '_id' column"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_pandas_invalid_id2(self):
        """Test check_labeled_data raises error when 'id2' contains invalid ids."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [99],  # 99 not in table_b
            'id1_list': [[1, 2]],
            'labels': [[0.5, 0.6]]
        })
        
        with pytest.raises(ValueError, match="candidates 'id2' column must only contain ids that are present in the table_b '_id' column"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_pandas_label_length_mismatch(self):
        """Test check_labeled_data raises error when label length doesn't match id1_list length."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = pd.DataFrame({
            'id2': [4, 5],
            'id1_list': [[1, 2], [3]],
            'labels': [[0.5], [0.7]]  # First label list has wrong length
        })
        
        with pytest.raises(ValueError, match="labeled_data 'labels' column must be a list the same length as its corresponding 'id1_list' column"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_spark_valid(self, spark_session):
        """Test check_labeled_data with valid Spark DataFrame."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'},
            {'_id': 3, 'value': 'c'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'},
            {'_id': 6, 'value': 'f'}
        ])
        labeled_data = spark_session.createDataFrame([
            {'id2': 4, 'id1_list': [1, 2], 'labels': [0.5, 0.6]},
            {'id2': 5, 'id1_list': [3], 'labels': [0.7]}
        ])
        
        # Should not raise
        utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_spark_missing_id2(self, spark_session):
        """Test check_labeled_data raises error when Spark DataFrame missing 'id2'."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        labeled_data = spark_session.createDataFrame([
            {'id1_list': [1, 2], 'labels': [0.5, 0.6]}
        ])
        
        with pytest.raises(ValueError, match="candidates must have the column 'id2'"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_spark_missing_id1_list(self, spark_session):
        """Test check_labeled_data raises error when Spark DataFrame missing 'id1_list'."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        labeled_data = spark_session.createDataFrame([
            {'id2': 4, 'labels': [0.5, 0.6]}
        ])
        
        with pytest.raises(ValueError, match="candidates must have the column 'id1_list'"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_spark_non_unique_id2(self, spark_session):
        """Test check_labeled_data raises error when Spark 'id2' not unique."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        labeled_data = spark_session.createDataFrame([
            {'id2': 4, 'id1_list': [1, 2], 'labels': [0.5, 0.6]},
            {'id2': 4, 'id1_list': [3], 'labels': [0.7]}
        ])
        
        with pytest.raises(ValueError, match="candidates 'id2' column must be unique"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')

    def test_check_labeled_data_invalid_type(self):
        """Test check_labeled_data raises error with non-DataFrame type."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        labeled_data = [1, 2, 3]
        
        with pytest.raises(ValueError, match="candidates must be a pandas DataFrame or Spark DataFrame"):
            utils.check_labeled_data(labeled_data, table_a, table_b, 'labels')


class TestCheckGoldData:
    """Tests for check_gold_data function."""

    def test_check_gold_data_pandas_valid(self):
        """Test check_gold_data with valid pandas DataFrame."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        gold_data = pd.DataFrame({
            'id1': [1, 2],
            'id2': [4, 5]
        })
        
        # Should not raise
        utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_logs_success_on_valid(self, caplog):
        """Test check_gold_data logs that formats are correct when validation passes."""
        import logging
        caplog.set_level(logging.WARNING)
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        gold_data = pd.DataFrame({'id1': [1, 2], 'id2': [4, 5]})
        utils.check_gold_data(gold_data, table_a, table_b)
        assert "check_gold_data" in caplog.text and "formats are correct" in caplog.text

    def test_check_gold_data_pandas_missing_id1(self):
        """Test check_gold_data raises error when missing 'id1' column."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        gold_data = pd.DataFrame({
            'id2': [4, 5]
        })
        
        with pytest.raises(ValueError, match="gold_data must have the column 'id1'"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_pandas_missing_id2(self):
        """Test check_gold_data raises error when missing 'id2' column."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        gold_data = pd.DataFrame({
            'id1': [1, 2]
        })
        
        with pytest.raises(ValueError, match="gold_data must have the column 'id2'"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_pandas_invalid_id1(self):
        """Test check_gold_data raises error when 'id1' contains invalid ids."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        gold_data = pd.DataFrame({
            'id1': [99],  # 99 not in table_a
            'id2': [4]
        })
        
        with pytest.raises(ValueError, match="gold_data 'id1' column must only contain ids that are present in the table_a '_id' column"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_pandas_invalid_id2(self):
        """Test check_gold_data raises error when 'id2' contains invalid ids."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        gold_data = pd.DataFrame({
            'id1': [1],
            'id2': [99]  # 99 not in table_b
        })
        
        with pytest.raises(ValueError, match="gold_data 'id2' column must only contain ids that are present in the table_b '_id' column"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_spark_valid(self, spark_session):
        """Test check_gold_data with valid Spark DataFrame."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'},
            {'_id': 3, 'value': 'c'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'},
            {'_id': 6, 'value': 'f'}
        ])
        gold_data = spark_session.createDataFrame([
            {'id1': 1, 'id2': 4},
            {'id1': 2, 'id2': 5}
        ])
        
        # Should not raise
        utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_spark_missing_id1(self, spark_session):
        """Test check_gold_data raises error when Spark DataFrame missing 'id1'."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        gold_data = spark_session.createDataFrame([
            {'id2': 4}
        ])
        
        with pytest.raises(ValueError, match="gold_data must have the column 'id1'"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_spark_missing_id2(self, spark_session):
        """Test check_gold_data raises error when Spark DataFrame missing 'id2'."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        gold_data = spark_session.createDataFrame([
            {'id1': 1}
        ])
        
        with pytest.raises(ValueError, match="gold_data must have the column 'id2'"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_spark_invalid_id1(self, spark_session):
        """Test check_gold_data raises error when Spark 'id1' contains invalid ids."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        gold_data = spark_session.createDataFrame([
            {'id1': 99, 'id2': 4}  # 99 not in table_a
        ])
        
        with pytest.raises(ValueError, match="gold_data 'id1' column must only contain ids that are present in the table_a '_id' column"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_spark_invalid_id2(self, spark_session):
        """Test check_gold_data raises error when Spark 'id2' contains invalid ids."""
        table_a = spark_session.createDataFrame([
            {'_id': 1, 'value': 'a'},
            {'_id': 2, 'value': 'b'}
        ])
        table_b = spark_session.createDataFrame([
            {'_id': 4, 'value': 'd'},
            {'_id': 5, 'value': 'e'}
        ])
        gold_data = spark_session.createDataFrame([
            {'id1': 1, 'id2': 99}  # 99 not in table_b
        ])
        
        with pytest.raises(ValueError, match="gold_data 'id2' column must only contain ids that are present in the table_b '_id' column"):
            utils.check_gold_data(gold_data, table_a, table_b)

    def test_check_gold_data_invalid_type(self):
        """Test check_gold_data raises error with non-DataFrame type."""
        table_a = pd.DataFrame({'_id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        table_b = pd.DataFrame({'_id': [4, 5, 6], 'value': ['d', 'e', 'f']})
        gold_data = [1, 2, 3]
        
        with pytest.raises(ValueError, match="gold_data must be a pandas DataFrame or Spark DataFrame"):
            utils.check_gold_data(gold_data, table_a, table_b)

