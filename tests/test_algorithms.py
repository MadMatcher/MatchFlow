"""Tests for MatchFlow algorithms module.

This module exposes helper functions for active learning selection.
"""
from pathlib import Path

from MatchFlow.tools import down_sample, create_seeds


class TestAlgorithms:
    """Tests for down_sample and create_seeds functions."""

    def test_down_sample(self, fvs_df):
        """Test down_sample reduces the pool appropriately."""
        total = fvs_df.count()
        down_sampled_fvs = down_sample(fvs_df, 0.5, search_id_column='id2')
        n = down_sampled_fvs.count()
        # down_sample uses an approximate bucketed scheme sized for large inputs
        # (bucket_size defaults to 1000), so on a tiny fixture the row count need
        # not equal exactly percent * total. It must still strictly reduce the
        # pool to at most the requested fraction and preserve the schema.
        assert 0 < n < total
        assert n <= total * 0.5
        assert down_sampled_fvs.columns == fvs_df.columns

    def test_select_seeds_sequence(self, fvs_df, labeler, temp_dir: Path):
        """Test create_seeds sequence: initial, enough existing, not enough."""
        parquet_path = str(
            temp_dir / "test-matchflow-training-data.parquet"
        )

        seeds = create_seeds(fvs_df, 4, labeler, 'score', parquet_path)
        assert seeds.count() == 4
        assert set(seeds.select('_id').toPandas()['_id'].tolist()) == set([0, 1, 4, 5])

        seeds = create_seeds(fvs_df, 4, labeler, 'score', parquet_path)
        assert seeds.count() == 4
        assert set(seeds.select('_id').toPandas()['_id'].tolist()) == set([0, 1, 4, 5])

        seeds = create_seeds(fvs_df, 6, labeler, 'score', parquet_path)
        assert seeds.count() == 6
        assert set(seeds.select('_id').toPandas()['_id'].tolist()) == set([0, 1, 2, 3, 4, 5])

