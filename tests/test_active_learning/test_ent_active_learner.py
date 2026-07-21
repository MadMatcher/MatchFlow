"""Tests for MatchFlow._internal.active_learning.ent_active_learner module.

This module implements EntropyActiveLearner for batch selection.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import pyspark.sql.functions as F

from MatchFlow._internal.active_learning.ent_active_learner import (
    EntropyActiveLearner,
)
from MatchFlow._internal.labeler import CustomLabeler, GoldLabeler


class StopLabeler(CustomLabeler):
    def label_pair(self, row1, row2):
        if row1["_id"] == 10 and row2["_id"] == 20:
            return -1.0
        return 1.0 if (row1["_id"] + row2["_id"]) % 2 == 0 else 0.0


class TestEntropyActiveLearner:
    """Tests for EntropyActiveLearner behavior."""

    def test_init(self, spark_session, default_model):
        """Validate __init__ parameter checks and defaults."""
        gold_df = spark_session.createDataFrame([{"id1": 1, "id2": 2}])
        labeler = GoldLabeler(gold_df)

        learner = EntropyActiveLearner(default_model, labeler, batch_size=5, max_iter=2, parquet_file_path="file.parquet")
        assert learner._batch_size == 5
        assert learner._max_iter == 2

        with pytest.raises(ValueError):
            EntropyActiveLearner(default_model, labeler, batch_size=0)
        with pytest.raises(ValueError):
            EntropyActiveLearner(default_model, labeler, max_iter=0)

    def test_train_basic(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df):
        """Ensure train executes active learning and persists data."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 12, "id2": 22}])
        labeler = GoldLabeler(gold_df)
        parquet_path = temp_dir / "ent_training.parquet"

        learner = EntropyActiveLearner(default_model, labeler, batch_size=1, max_iter=3, parquet_file_path=str(parquet_path))

        filtered = learner._select_training_vectors(fvs_df, [0])
        assert filtered.count() == 1
        pos, neg = learner._get_pos_negative(seed_df)
        assert pos == 1.0
        assert neg == 1.0

        trained = learner.train(fvs_df, seed_df)
        assert trained is not None
        assert parquet_path.exists()
        parquet_df = pd.read_parquet(parquet_path)
        assert set(["_id", "id1", "id2", "feature_vectors", "label"]).issubset(parquet_df.columns)
        assert parquet_df["label"].isin([0.0, 1.0]).all()

        assert learner.local_training_fvs_ is not None
        assert "labeled_in_iteration" in learner.local_training_fvs_.columns

    def test_train_existing_data(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df):
        """Ensure existing parquet data path is used."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 12, "id2": 22}])
        labeler = GoldLabeler(gold_df)
        parquet_path = temp_dir / "ent_training.parquet"

        learner_existing = EntropyActiveLearner(
            default_model, labeler, batch_size=1, max_iter=3, parquet_file_path=str(parquet_path)
        )
        learner_existing.train(fvs_df, seed_df)

        trained_existing = learner_existing.train(fvs_df, seed_df)
        assert trained_existing is not None
        assert "labeled_in_iteration" in learner_existing.local_training_fvs_.columns

    def test_train_spark_model(self, spark_session, temp_dir: Path, spark_model, seed_df, fvs_df):
        """Ensure active learning works with a spark model, not just sklearn.

        Spark models turn feature_vectors into a vector column, so the batches
        selected during active learning come back as DenseVectors while the
        seeds are still plain lists. Both have to end up in the same spark
        DataFrame, which only works if the schema is taken from the prepped fvs
        rather than inferred from the seeds.
        """
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 12, "id2": 22}])
        labeler = GoldLabeler(gold_df)
        parquet_path = temp_dir / "ent_spark_training.parquet"

        learner = EntropyActiveLearner(spark_model, labeler, batch_size=1, max_iter=3, parquet_file_path=str(parquet_path))
        labeled = learner.train(fvs_df, seed_df)

        # more than one iteration must have run, the second one is where the
        # seed vectors and the selected vectors first share a DataFrame
        assert labeled["labeled_in_iteration"].max() >= 1.0
        # feature vectors come back as plain lists of floats, same as sklearn
        vec = labeled["feature_vectors"].iloc[-1]
        assert isinstance(vec, list)
        assert all(type(x) is float for x in vec)

    def test_train_spark_model_existing_data(self, spark_session, temp_dir: Path, spark_model, seed_df, fvs_df):
        """Ensure the resume-from-parquet path works with a spark model."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 12, "id2": 22}])
        labeler = GoldLabeler(gold_df)
        parquet_path = temp_dir / "ent_spark_existing.parquet"

        learner = EntropyActiveLearner(spark_model, labeler, batch_size=1, max_iter=3, parquet_file_path=str(parquet_path))
        learner.train(fvs_df, seed_df)

        resumed = learner.train(fvs_df, seed_df)
        assert resumed is not None
        assert "labeled_in_iteration" in learner.local_training_fvs_.columns

    def test_train_label_everything(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df):
        """Ensure label-everything path is exercised."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 12, "id2": 22}])
        labeler = GoldLabeler(gold_df)
        fvs_with_li = (
            fvs_df.limit(2)
            .withColumn("labeled_in_iteration", F.lit(0))
        )
        parquet_path = str(temp_dir / "ent_label_all.parquet")
        learner_label_all = EntropyActiveLearner(
            default_model, labeler, batch_size=10, max_iter=1,
            parquet_file_path=parquet_path
        )
        learner_label_all._terminate_if_label_everything = True
        label_all = learner_label_all.train(fvs_with_li, seed_df)
        assert label_all is not None

    def test_train_error_pos_neg(
        self, spark_session, temp_dir: Path, default_model, fvs_df, seed_df
    ):
        """Ensure runtime error is raised when all labels are same."""
        gold_df = spark_session.createDataFrame([{"id1": 10, "id2": 20}, {"id1": 12, "id2": 22}])
        labeler = GoldLabeler(gold_df)
        seed_all_positive = seed_df.copy()
        seed_all_positive["label"] = 1.0
        parquet_path = str(temp_dir / "ent_error.parquet")
        learner_error = EntropyActiveLearner(
            default_model, labeler, batch_size=1, max_iter=1,
            parquet_file_path=parquet_path
        )
        with pytest.raises(RuntimeError):
            learner_error.train(fvs_df, seed_all_positive)

    def test_train_user_stop(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df, id_df_factory):
        """Ensure user stop is handled gracefully."""
        a_df = id_df_factory([10, 11, 12, 13, 14, 15])
        b_df = id_df_factory([20, 21, 22, 23, 24, 25])
        stop_labeler = StopLabeler(a_df, b_df)
        learner_stop = EntropyActiveLearner(
            default_model, stop_labeler, batch_size=1, max_iter=1, parquet_file_path=str(temp_dir / "ent_stop.parquet")
        )
        stopped = learner_stop.train(fvs_df, seed_df)
        assert stopped is not None

