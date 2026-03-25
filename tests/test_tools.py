"""Tests for MatchFlow.tools module.

This module provides tests for the public API functions in tools.py.
"""
import pytest
import pandas as pd
from pathlib import Path

from MatchFlow import tools
from MatchFlow._internal.ml_model import SKLearnModel
from MatchFlow._internal.labeler import GoldLabeler
from xgboost import XGBClassifier
from pyspark.sql import DataFrame as SparkDataFrame


class TestDownSample:
    """Tests for down_sample function."""

    def test_down_sample_pandas(self):
        """Test down_sample with pandas DataFrame."""
        df = pd.DataFrame({
            '_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'value': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        })

        result = tools.down_sample(df, percent=0.5, search_id_column='_id', score_column='score')

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(df)
        assert len(result) > 0
        assert '_id' in result.columns
        assert 'score' in result.columns
        assert 'value' in result.columns
        # No temp columns should leak through
        assert '_HASH' not in result.columns
        assert '_PERCENTILE' not in result.columns
        assert '_SAMPLE' not in result.columns
        assert '_ROW_NUM' not in result.columns

    def test_down_sample_pandas_invalid_percent(self):
        """Test down_sample raises ValueError for invalid percent."""
        df = pd.DataFrame({'_id': [1, 2], 'score': [0.1, 0.2]})

        with pytest.raises(ValueError, match='percent must be in the range'):
            tools.down_sample(df, percent=0.0, search_id_column='_id', score_column='score')

        with pytest.raises(ValueError, match='percent must be in the range'):
            tools.down_sample(df, percent=1.5, search_id_column='_id', score_column='score')

    def test_down_sample_pandas_invalid_bucket_size(self):
        """Test down_sample raises ValueError for invalid bucket_size with pandas."""
        df = pd.DataFrame({'_id': [1, 2], 'score': [0.1, 0.2]})

        with pytest.raises(ValueError, match='bucket_size must be >= 1000'):
            tools.down_sample(df, percent=0.5, search_id_column='_id', score_column='score', bucket_size=500)

    def test_down_sample_spark(self, spark_session):
        """Test down_sample with Spark DataFrame."""
        df = spark_session.createDataFrame([
            {'_id': i, 'score': float(i) / 10.0, 'value': f'val_{i}'}
            for i in range(1, 11)
        ])

        result = tools.down_sample(df, percent=0.5, search_id_column='_id', score_column='score')

        assert hasattr(result, 'count')
        count = result.count()
        assert count <= 10
        assert count > 0

    def test_down_sample_spark_invalid_bucket_size(self, spark_session):
        """Test down_sample raises ValueError for invalid bucket_size."""
        df = spark_session.createDataFrame([{'_id': 1, 'score': 0.1}])

        with pytest.raises(ValueError, match='bucket_size must be >= 1000'):
            tools.down_sample(df, percent=0.5, search_id_column='_id', score_column='score', bucket_size=500)


class TestCreateSeeds:
    """Tests for create_seeds function."""

    def test_create_seeds_pandas(self, spark_session, temp_dir):
        """Test create_seeds with pandas DataFrame."""
        fvs = pd.DataFrame({
            '_id': [1, 2, 3, 4, 5],
            'id1': [10, 11, 12, 13, 14],
            'id2': [20, 21, 22, 23, 24],
            'feature_vectors': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
            'score': [0.1, 0.3, 0.5, 0.7, 0.9]
        })

        gold_df = spark_session.createDataFrame([{'id1': 14, 'id2': 24}, {'id1': 10, 'id2': 20}])
        labeler = GoldLabeler(gold_df)

        parquet_path = str(temp_dir / 'seeds.parquet')
        seeds = tools.create_seeds(fvs, nseeds=3, labeler=labeler, score_column='score', parquet_file_path=parquet_path)
        assert isinstance(seeds, pd.DataFrame)
        assert len(seeds) == 3
        assert 'label' in seeds.columns
        assert seeds['label'].isin([0.0, 1.0]).all()
        assert Path(parquet_path).exists()

    def test_create_seeds_spark(self, spark_session, temp_dir):
        """Test create_seeds with Spark DataFrame."""
        fvs = spark_session.createDataFrame([
            {'_id': 1, 'id1': 10, 'id2': 20, 'feature_vectors': [0.1, 0.2], 'score': 0.9},
            {'_id': 2, 'id1': 11, 'id2': 21, 'feature_vectors': [0.3, 0.4], 'score': 0.8},
            {'_id': 3, 'id1': 12, 'id2': 22, 'feature_vectors': [0.5, 0.6], 'score': 0.7},
        ])

        gold_df = spark_session.createDataFrame([{'id1': 10, 'id2': 20}])
        labeler = GoldLabeler(gold_df)

        parquet_path = str(temp_dir / 'seeds_spark.parquet')
        seeds = tools.create_seeds(fvs, nseeds=2, labeler=labeler, score_column='score', parquet_file_path=parquet_path)

        assert hasattr(seeds, 'count')
        assert seeds.count() == 2

    def test_create_seeds_zero_seeds(self, spark_session):
        """Test create_seeds raises ValueError for nseeds=0."""
        fvs = pd.DataFrame({'_id': [1], 'id1': [10], 'id2': [20], 'score': [0.5]})
        gold_df = spark_session.createDataFrame([{'id1': 10, 'id2': 20}])
        labeler = GoldLabeler(gold_df)

        with pytest.raises(ValueError, match='no seeds would be created'):
            tools.create_seeds(fvs, nseeds=0, labeler=labeler)

    def test_create_seeds_existing_data(self, spark_session, temp_dir):
        """Test create_seeds uses existing training data."""
        existing_data = pd.DataFrame({
            '_id': [1, 2],
            'id1': [10, 11],
            'id2': [20, 21],
            'feature_vectors': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0]
        })
        parquet_path = temp_dir / 'existing_seeds.parquet'
        existing_data.to_parquet(parquet_path, index=False)

        fvs = pd.DataFrame({
            '_id': [3, 4, 5],
            'id1': [12, 13, 14],
            'id2': [22, 23, 24],
            'feature_vectors': [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
            'score': [0.5, 0.6, 0.7]
        })

        gold_df = spark_session.createDataFrame([{'id1': 12, 'id2': 22}])
        labeler = GoldLabeler(gold_df)

        seeds = tools.create_seeds(fvs, nseeds=2, labeler=labeler, parquet_file_path=str(parquet_path))

        assert len(seeds) == 2


class TestTrainMatcher:
    """Tests for train_matcher function."""

    def test_train_matcher_pandas(self, seed_df):
        """Test train_matcher with pandas DataFrame."""
        model = SKLearnModel(XGBClassifier, n_estimators=5, random_state=42)

        if 'features' in seed_df.columns and 'feature_vectors' not in seed_df.columns:
            seed_df = seed_df.rename(columns={'features': 'feature_vectors'})

        trained = tools.train_matcher(model, seed_df, feature_col='feature_vectors', label_col='label')

        assert trained is not None
        assert trained._trained_model is not None

    def test_train_matcher_spark(self, spark_session, seed_df):
        """Test train_matcher with Spark DataFrame."""
        model = SKLearnModel(XGBClassifier, n_estimators=5, random_state=42)

        if 'features' in seed_df.columns and 'feature_vectors' not in seed_df.columns:
            seed_df = seed_df.rename(columns={'features': 'feature_vectors'})

        spark_df = spark_session.createDataFrame(seed_df)
        trained = tools.train_matcher(model, spark_df, feature_col='feature_vectors', label_col='label')

        assert trained is not None
        assert trained._trained_model is not None


class TestApplyMatcher:
    """Tests for apply_matcher function."""

    def test_apply_matcher_predict_only(self, seed_df, default_model):
        """Test apply_matcher with predictions only."""
        if 'features' in seed_df.columns and 'feature_vectors' not in seed_df.columns:
            seed_df = seed_df.rename(columns={'features': 'feature_vectors'})

        trained = default_model.train(seed_df, 'feature_vectors', 'label')

        test_df = pd.DataFrame({
            '_id': [3, 4],
            'id1': [13, 14],
            'id2': [23, 24],
            'feature_vectors': [[0.5, 0.6], [0.7, 0.8]]
        })

        result = tools.apply_matcher(trained, test_df, 'feature_vectors', 'prediction')

        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns
        assert result['prediction'].isin([0.0, 1.0]).all()

    def test_apply_matcher_with_confidence(self, seed_df, default_model):
        """Test apply_matcher with confidence scores."""
        if 'features' in seed_df.columns and 'feature_vectors' not in seed_df.columns:
            seed_df = seed_df.rename(columns={'features': 'feature_vectors'})

        trained = default_model.train(seed_df, 'feature_vectors', 'label')

        test_df = pd.DataFrame({
            '_id': [3, 4],
            'id1': [13, 14],
            'id2': [23, 24],
            'feature_vectors': [[0.5, 0.6], [0.7, 0.8]]
        })

        result = tools.apply_matcher(trained, test_df, 'feature_vectors', 'prediction', 'confidence')

        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns
        assert 'confidence' in result.columns
        assert (result['confidence'] >= 0.0).all()
        assert (result['confidence'] <= 1.0).all()

    def test_apply_matcher_spark(self, spark_session, seed_df, default_model):
        """Test apply_matcher with Spark DataFrame."""
        if 'features' in seed_df.columns and 'feature_vectors' not in seed_df.columns:
            seed_df = seed_df.rename(columns={'features': 'feature_vectors'})

        trained = default_model.train(seed_df, 'feature_vectors', 'label')

        test_df = spark_session.createDataFrame([
            {'_id': 3, 'id1': 13, 'id2': 23, 'feature_vectors': [0.5, 0.6]},
            {'_id': 4, 'id1': 14, 'id2': 24, 'feature_vectors': [0.7, 0.8]}
        ])

        result = tools.apply_matcher(trained, test_df, 'feature_vectors', 'prediction')

        assert hasattr(result, 'count')
        assert 'prediction' in [col.name for col in result.schema.fields]


class TestLabelPairs:
    """Tests for label_pairs function."""

    def test_label_pairs_pandas(self, spark_session):
        """Test label_pairs with pandas DataFrame."""
        pairs = pd.DataFrame({
            'id1': [10, 11, 12],
            'id2': [20, 21, 22]
        })

        gold_df = spark_session.createDataFrame([{'id1': 10, 'id2': 20}, {'id1': 11, 'id2': 21}])
        labeler = GoldLabeler(gold_df)

        result = tools.label_pairs(labeler, pairs)

        assert isinstance(result, pd.DataFrame)
        assert 'label' in result.columns
        assert len(result) == 3
        assert result['label'].tolist() == [1.0, 1.0, 0.0]

    def test_label_pairs_spark(self, spark_session):
        """Test label_pairs with Spark DataFrame."""
        pairs = spark_session.createDataFrame([
            {'id1': 10, 'id2': 20},
            {'id1': 11, 'id2': 21},
            {'id1': 12, 'id2': 22}
        ])

        gold_df = spark_session.createDataFrame([{'id1': 10, 'id2': 20}])
        labeler = GoldLabeler(gold_df)

        result = tools.label_pairs(labeler, pairs)

        assert hasattr(result, 'count')
        count = result.count()
        assert count == 3
        assert 'label' in [col.name for col in result.schema.fields]


class TestSaveLoadFeatures:
    """Tests for save_features and load_features functions."""

    def test_save_load_features(self, temp_dir, a_df, b_df):
        """Test saving and loading features."""
        from MatchFlow import create_features
        features = create_features(a_df, b_df, ['a_attr', 'a_num'], ['a_attr', 'a_num'])

        features_path = temp_dir / 'test_features.pkl'

        tools.save_features(features, features_path)
        assert features_path.exists()

        loaded_features = tools.load_features(features_path)

        assert len(loaded_features) == len(features)
        assert callable(loaded_features[0])
        assert callable(loaded_features[1])

    def test_load_features_nonexistent(self, temp_dir):
        """Test load_features with non-existent file."""
        features_path = temp_dir / 'nonexistent.pkl'

        with pytest.raises(FileNotFoundError):
            tools.load_features(features_path)


class TestSaveLoadDataframe:
    """Tests for save_dataframe and load_dataframe functions."""

    def test_save_load_dataframe_pandas(self, temp_dir):
        """Test saving and loading pandas DataFrame."""
        df = pd.DataFrame({
            '_id': [1, 2, 3],
            'value': [0.1, 0.2, 0.3],
            'name': ['a', 'b', 'c']
        })

        parquet_path = temp_dir / 'test_df.parquet'

        tools.save_dataframe(df, parquet_path)
        assert parquet_path.exists()

        loaded_df = tools.load_dataframe(parquet_path, 'pandas')

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_save_load_dataframe_spark(self, spark_session, temp_dir):
        """Test saving and loading Spark DataFrame."""
        df = spark_session.createDataFrame([
            {'_id': 1, 'value': 0.1, 'name': 'a'},
            {'_id': 2, 'value': 0.2, 'name': 'b'},
            {'_id': 3, 'value': 0.3, 'name': 'c'}
        ])

        parquet_path = temp_dir / 'test_spark_df.parquet'

        tools.save_dataframe(df, parquet_path)
        assert parquet_path.exists()

        loaded_df = tools.load_dataframe(parquet_path, 'sparkdf')

        assert hasattr(loaded_df, 'count')
        assert loaded_df.count() == 3
        assert '_id' in [col.name for col in loaded_df.schema.fields]

    def test_save_dataframe_invalid_type(self):
        """Test save_dataframe raises TypeError for invalid type."""
        with pytest.raises(TypeError, match='Unsupported DataFrame type'):
            tools.save_dataframe([1, 2, 3], Path('test.parquet'))

    def test_load_dataframe_invalid_type(self, temp_dir):
        """Test load_dataframe raises ValueError for invalid type."""
        parquet_path = temp_dir / 'test.parquet'
        pd.DataFrame({'_id': [1]}).to_parquet(parquet_path)

        with pytest.raises(ValueError, match='Unsupported DataFrame type'):
            tools.load_dataframe(parquet_path, 'invalid_type')


class TestLabelData:
    """Tests for label_data function."""

    def test_label_data_batch_mode(self, spark_session, default_model, fvs_df, seed_df, temp_dir):
        """Test label_data in batch mode."""
        gold_df = spark_session.createDataFrame([
            {'id1': 11, 'id2': 21},
            {'id1': 12, 'id2': 22}
        ])
        labeler = GoldLabeler(gold_df)

        if 'features' in seed_df.columns and 'feature_vectors' not in seed_df.columns:
            seed_df = seed_df.rename(columns={'features': 'feature_vectors'})

        parquet_path = str(temp_dir / 'batch_labeling.parquet')

        result = tools.label_data(
            model=default_model,
            mode='batch',
            labeler=labeler,
            fvs=fvs_df,
            seeds=seed_df,
            parquet_file_path=parquet_path,
            batch_size=1,
            max_iter=2
        )

        assert isinstance(result, SparkDataFrame)
        assert 'label' in [col.name for col in result.schema.fields]
        assert result.select('label').distinct().count() == 2

    def test_label_data_continuous_mode(self, spark_session, default_model, fvs_df, seed_df, temp_dir):
        """Test label_data in continuous mode."""
        gold_df = spark_session.createDataFrame([
            {'id1': 10, 'id2': 20},
            {'id1': 12, 'id2': 22}
        ])
        labeler = GoldLabeler(gold_df)

        if 'features' in seed_df.columns and 'feature_vectors' not in seed_df.columns:
            seed_df = seed_df.rename(columns={'features': 'feature_vectors'})

        parquet_path = str(temp_dir / 'continuous_labeling.parquet')

        result = tools.label_data(
            model=default_model,
            mode='continuous',
            labeler=labeler,
            fvs=fvs_df,
            seeds=seed_df,
            parquet_file_path=parquet_path,
            queue_size=3,
            max_labeled=5,
            on_demand_stop=True
        )

        assert isinstance(result, SparkDataFrame)
        assert 'label' in [col.name for col in result.schema.fields]

    def test_label_data_pandas_input(self, spark_session, default_model, temp_dir):
        """Test label_data with pandas DataFrame input."""
        fvs = pd.DataFrame({
            '_id': [0, 1, 2],
            'id1': [10, 11, 12],
            'id2': [20, 21, 22],
            'feature_vectors': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            'score': [0.5, 0.6, 0.7]
        })

        seeds = pd.DataFrame({
            '_id': [0, 1],
            'id1': [10, 11],
            'id2': [20, 21],
            'feature_vectors': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0]
        })

        gold_df = spark_session.createDataFrame([{'id1': 10, 'id2': 20}])
        labeler = GoldLabeler(gold_df)

        parquet_path = str(temp_dir / 'pandas_labeling.parquet')

        result = tools.label_data(
            model=default_model,
            mode='batch',
            labeler=labeler,
            fvs=fvs,
            seeds=seeds,
            parquet_file_path=parquet_path,
            batch_size=1,
            max_iter=1
        )

        assert isinstance(result, pd.DataFrame)
