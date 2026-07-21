"""Pytest configuration and shared fixtures (mirroring delex style)."""
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "MatchFlow"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import MatchFlow
    if Path(getattr(MatchFlow, "__file__", "")).resolve() == (REPO_ROOT / "__init__.py").resolve():
        raise ImportError("Repo-level __init__.py shadowing package")
except Exception:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "MatchFlow",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["MatchFlow"] = module
    if spec and spec.loader:
        spec.loader.exec_module(module)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    path = Path(tempfile.mkdtemp())
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file within temp_dir."""
    file_path = temp_dir / "test.tmp"
    file_path.write_text("")
    try:
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a small deterministic pandas DataFrame."""
    return pd.DataFrame({"_id": [1, 2, 3], "value": [0.1, 0.2, 0.3]})


@pytest.fixture
def sample_feature_vectors() -> np.ndarray:
    """Return a small numpy array of feature vectors."""
    return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)


@pytest.fixture
def spark_session():
    """Provide a SparkSession if pyspark is installed."""
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("matchflow_tests").getOrCreate()
        yield spark
        spark.stop()
    except ImportError:
        pytest.skip("pyspark not installed")


@pytest.fixture
def default_model():
    """Return a default SKLearnModel for active learning tests."""
    from MatchFlow._internal.ml_model import SKLearnModel
    from xgboost import XGBClassifier

    return SKLearnModel(
        XGBClassifier, max_depth=3, n_estimators=10, random_state=42
    )


@pytest.fixture
def spark_model():
    """Return a SparkMLModel for active learning tests.

    Spark models use vector feature columns rather than arrays, which is the
    representation active learning has to keep consistent.
    """
    from MatchFlow._internal.ml_model import SparkMLModel
    from pyspark.ml.classification import RandomForestClassifier

    return SparkMLModel(RandomForestClassifier, numTrees=5, maxDepth=3, seed=42)


@pytest.fixture
def seed_df() -> pd.DataFrame:
    """Return a default seed DataFrame for active learning tests."""
    return pd.DataFrame(
        {
            "_id": [0, 2],
            "id1": [10, 12],
            "id2": [20, 22],
            "feature_vectors": [[0.1, 0.2], [0.3, 0.4]],
            "label": [1.0, 0.0],
        }
    )


@pytest.fixture
def fvs_rows():
    """Return a default list of feature vector rows for Spark DataFrames."""
    return [
        {"_id": 0, "id1": 10, "id2": 20, "feature_vectors": [0.1, 0.1], "score": 0.2},
        {"_id": 1, "id1": 11, "id2": 21, "feature_vectors": [0.2, 0.1], "score": 0.3},
        {"_id": 2, "id1": 12, "id2": 22, "feature_vectors": [0.3, 0.4], "score": 0.7},
        {"_id": 3, "id1": 13, "id2": 23, "feature_vectors": [0.4, 0.5], "score": 0.9},
        {"_id": 4, "id1": 14, "id2": 24, "feature_vectors": [0.5, 0.6], "score": 1.1},
        {"_id": 5, "id1": 15, "id2": 25, "feature_vectors": [0.6, 0.6], "score": 1.2},
    ]


@pytest.fixture
def fvs_df(spark_session, fvs_rows):
    """Return a Spark DataFrame of feature vectors."""
    return spark_session.createDataFrame(fvs_rows)


@pytest.fixture
def a_df(spark_session):
    """Return a Spark DataFrame of table A."""
    return spark_session.createDataFrame([
        {"_id": 10, "a_attr": "a", "a_num": 1.0},
        {"_id": 11, "a_attr": "b", "a_num": 2.0},
        {"_id": 12, "a_attr": "c", "a_num": 3.0},
    ])


@pytest.fixture
def b_df(spark_session):
    """Return a Spark DataFrame of table B."""
    return spark_session.createDataFrame([
        {"_id": 20, "a_attr": "a", "a_num": 1.0},
        {"_id": 21, "a_attr": "b", "a_num": 2.0},
        {"_id": 22, "a_attr": "c", "a_num": 3.0},
    ])


@pytest.fixture
def id_df_factory(spark_session):
    """Factory for creating _id-only Spark DataFrames."""
    def _factory(ids):
        rows = [{"_id": int(_id)} for _id in ids]
        return spark_session.createDataFrame(rows)

    return _factory


@pytest.fixture
def tokenizer():
    """Return a default tokenizer for token-based features."""
    from MatchFlow._internal.tokenizer.tokenizer import StrippedWhiteSpaceTokenizer
    return StrippedWhiteSpaceTokenizer()


@pytest.fixture
def labeler(spark_session):
    """Return a default labeler for active learning tests."""
    from MatchFlow._internal.labeler import GoldLabeler
    gold_df = spark_session.createDataFrame([{"id1": 13, "id2": 23}, {"id1": 14, "id2": 24}, {"id1": 15, "id2": 25}])
    return GoldLabeler(gold_df)
