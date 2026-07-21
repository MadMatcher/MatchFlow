"""Tests for MatchFlow._internal.ml_model module.
This module wraps MLModel abstractions and model utilities.
"""
import pytest
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml.classification import GBTClassifier
from MatchFlow._internal.ml_model import (
    MLModel,
    SKLearnModel,
    SparkMLModel,
    convert_to_vector,
    convert_to_array,
    make_training_schema,
    to_spark_training_fvs,
)


class TestMLModelBase:
    """Tests for MLModel abstract base class."""
    def test_requires_fit_predict(self):
        """Ensure abstract methods enforce implementation."""
        with pytest.raises(TypeError):
            MLModel()

        class IncompleteModel(MLModel):
            @property
            def nan_fill(self):
                return None

            @property
            def use_vectors(self):
                return False

            @property
            def use_floats(self):
                return False
        with pytest.raises(TypeError):
            IncompleteModel()

        class CompleteModel(MLModel):
            @property
            def nan_fill(self):
                return None

            @property
            def use_vectors(self):
                return False

            @property
            def use_floats(self):
                return False

            def predict(self, df, vector_col, output_col):
                return df

            def prediction_conf(self, df, vector_col, label_column):
                return df

            def entropy(self, df, vector_col, output_col):
                return df

            def train(self, df, vector_col, label_column):
                pass

            def params_dict(self):
                return {}

            def trained_model(self):
                return None

            def predict_with_confidence(
                self, df, vector_col, prediction_col, confidence_col
            ):
                return df

        model = CompleteModel()
        assert model.nan_fill is None
        assert model.use_vectors is False
        assert model.use_floats is False


class TestConvertHelpers:
    """Tests for conversion helpers."""

    def test_convert_to_vector(self, spark_session):
        """Verify convert_to_vector converts columns properly."""
        from pyspark.sql.types import StructType, StructField

        df_array = spark_session.createDataFrame(
            [([1.0, 2.0],), ([3.0, 4.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = convert_to_vector(df_array, "features")
        assert isinstance(result.schema["features"].dataType, VectorUDT)

        df_vector = spark_session.createDataFrame(
            [(Vectors.dense([1.0, 2.0]),), (Vectors.dense([3.0, 4.0]),)],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        result = convert_to_vector(df_vector, "features")
        assert isinstance(result.schema["features"].dataType, VectorUDT)

    def test_convert_to_array(self, spark_session):
        """Ensure convert_to_array returns numpy arrays."""
        from pyspark.sql.types import StructType, StructField

        df_array = spark_session.createDataFrame(
            [([1.0, 2.0],), ([3.0, 4.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = convert_to_array(df_array, "features")
        assert isinstance(result.schema["features"].dataType, ArrayType)
        assert isinstance(
            result.schema["features"].dataType.elementType, DoubleType
        )

        df_float_array = spark_session.createDataFrame(
            [([1.0, 2.0],), ([3.0, 4.0],)],
            schema=StructType([
                StructField("features", ArrayType(FloatType()), True)
            ])
        )
        result = convert_to_array(df_float_array, "features")
        assert isinstance(result.schema["features"].dataType, ArrayType)
        assert isinstance(
            result.schema["features"].dataType.elementType, FloatType
        )


class TestTrainingFvsHelpers:
    """Tests for the schema helpers used by active learning."""

    def test_make_training_schema(self, fvs_df):
        """Schema comes from the fvs, with the label columns appended."""
        columns = ["_id", "id1", "id2", "feature_vectors", "label", "labeled_in_iteration"]
        schema = make_training_schema(fvs_df, columns)

        # the fvs fields keep their order, the label columns go on the end
        fvs_fields = [c for c in fvs_df.columns if c in columns]
        assert schema.names == fvs_fields + ["label", "labeled_in_iteration"]
        # 'score' is in fvs_df but not in the training data, so it is left out
        assert "score" not in schema.names
        assert isinstance(schema["labeled_in_iteration"].dataType, DoubleType)

    def test_make_training_schema_no_duplicate_label_cols(self, fvs_df):
        """A label column already on the fvs is not added twice."""
        fvs = fvs_df.withColumn("labeled_in_iteration", F.lit(0))
        schema = make_training_schema(fvs, fvs.columns + ["label"])

        assert schema.names.count("labeled_in_iteration") == 1
        assert isinstance(schema["labeled_in_iteration"].dataType, DoubleType)

    def test_to_spark_training_fvs_mixed_representations(self, spark_session, fvs_df):
        """Vectors and lists in one column, against a vector schema.

        This is what active learning ends up with when a spark model is used:
        seeds are still lists while the selected batches come back from
        toPandas as DenseVectors.
        """
        schema = make_training_schema(
            convert_to_vector(fvs_df, "feature_vectors"),
            ["_id", "id1", "id2", "feature_vectors", "label", "labeled_in_iteration"],
        )
        pdf = pd.DataFrame({
            "_id": [0, 1],
            "id1": [10, 11],
            "id2": [20, 21],
            "feature_vectors": [[0.1, 0.2], Vectors.dense([1.0, 0.5])],
            "label": [1.0, 0.0],
            # written as an int by the batch learner, the schema says double
            "labeled_in_iteration": [-1, 0],
            # not part of the schema, must be dropped rather than shifting the
            # columns out of alignment
            "entropy": [0.9, 0.8],
        })

        result = to_spark_training_fvs(spark_session, pdf, schema)

        assert isinstance(result.schema["feature_vectors"].dataType, VectorUDT)
        assert result.columns == schema.names
        assert result.count() == 2

    def test_to_spark_training_fvs_array_schema(self, spark_session, fvs_df):
        """The same frame against an array schema, as sklearn models use."""
        schema = make_training_schema(
            fvs_df, ["_id", "id1", "id2", "feature_vectors", "label", "labeled_in_iteration"]
        )
        pdf = pd.DataFrame({
            "_id": [0, 1],
            "id1": [10, 11],
            "id2": [20, 21],
            "feature_vectors": [[0.1, 0.2], Vectors.dense([1.0, 0.5])],
            "label": [1.0, 0.0],
            "labeled_in_iteration": [-1.0, 0.0],
        })

        result = to_spark_training_fvs(spark_session, pdf, schema)

        assert isinstance(result.schema["feature_vectors"].dataType, ArrayType)
        assert result.count() == 2


class TestSKLearnModel:
    """Tests for SKLearnModel wrapper."""

    def test_init(self):
        """Test SKLearnModel initialization."""
        from xgboost import XGBClassifier
        model = SKLearnModel(
            XGBClassifier,
            nan_fill=0.0,
            use_floats=True,
            n_estimators=10,
            random_state=42
        )
        assert model.nan_fill == 0.0
        assert model.use_floats is True
        assert model.use_vectors is False
        assert model._model_args == {"n_estimators": 10, "random_state": 42}
        assert model._trained_model is None
        assert model._vector_buffer is None

    def test_init_defaults(self):
        """Test SKLearnModel with default parameters."""
        from xgboost import XGBClassifier
        model = SKLearnModel(XGBClassifier)
        assert model.nan_fill is None
        assert model.use_floats is True
        assert model.use_vectors is False

    def test_properties(self):
        """Test SKLearnModel properties."""
        from xgboost import XGBClassifier
        model = SKLearnModel(
            XGBClassifier,
            nan_fill=5.0,
            use_floats=False,
            random_state=42
        )
        assert model.nan_fill == 5.0
        assert model.use_floats is False
        assert model.use_vectors is False

    def test_params_dict(self):
        """Test params_dict returns correct dictionary."""
        from xgboost import XGBClassifier
        model = SKLearnModel(
            XGBClassifier, nan_fill=0.0, n_estimators=10, random_state=42
        )
        params = model.params_dict()
        assert params["nan_fill"] == 0.0
        assert params["model_args"] == {"n_estimators": 10, "random_state": 42}
        assert "XGBClassifier" in str(params["model"])

    def test_get_model(self):
        """Test get_model creates model instance."""
        from xgboost import XGBClassifier
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        sklearn_model = model.get_model()
        assert isinstance(sklearn_model, XGBClassifier)
        assert sklearn_model.n_estimators == 10
        assert sklearn_model.random_state == 42

    def test_allocate_buffer(self):
        """Test buffer allocation and reuse."""
        from xgboost import XGBClassifier
        model = SKLearnModel(XGBClassifier, use_floats=True, random_state=42)
        buffer1 = model._allocate_buffer(10, 5)
        assert buffer1.shape == (10, 5)
        assert buffer1.dtype == np.float32
        assert model._vector_buffer is not None
        buffer2 = model._allocate_buffer(5, 5)
        assert buffer2.shape == (5, 5)
        assert buffer2.dtype == np.float32
        if hasattr(buffer1, 'base'):
            assert model._vector_buffer is buffer1.base
        else:
            assert model._vector_buffer is not None
        buffer3 = model._allocate_buffer(20, 10)
        assert buffer3.shape == (20, 10)
        assert buffer3.dtype == np.float32

    def test_allocate_buffer_double(self):
        """Test buffer allocation with double precision."""
        from xgboost import XGBClassifier
        model = SKLearnModel(XGBClassifier, use_floats=False, random_state=42)
        buffer = model._allocate_buffer(5, 3)
        assert buffer.dtype == np.float64

    def test_make_feature_matrix(self):
        """Test feature matrix creation."""
        from xgboost import XGBClassifier
        model = SKLearnModel(XGBClassifier, use_floats=True, random_state=42)
        vecs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        X = model._make_feature_matrix(vecs)
        assert X.shape == (2, 2)
        assert X.dtype == np.float32
        np.testing.assert_array_equal(X[0], [1.0, 2.0])
        np.testing.assert_array_equal(X[1], [3.0, 4.0])

    def test_make_feature_matrix_empty(self):
        """Test feature matrix creation with empty input."""
        from xgboost import XGBClassifier
        model = SKLearnModel(XGBClassifier)
        X = model._make_feature_matrix([])
        assert X is None

    def test_train(self, spark_session):
        """Test model training."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        df = spark_session.createDataFrame(
            [
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 0.0),
                ([1.0, 1.0], 1.0),
                ([0.0, 0.0], 0.0),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(df, "features", "label")
        assert model._trained_model is not None
        assert hasattr(model._trained_model, "predict")

    def test_predict_untrained(self, spark_session):
        """Test predict raises error when model not trained."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, random_state=42)
        df = spark_session.createDataFrame(
            [([1.0, 2.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.predict(df, "features", "predictions")

    def test_predict_trained(self, spark_session):
        """Test prediction with trained model."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        train_df = spark_session.createDataFrame(
            [
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 0.0),
                ([1.0, 1.0], 1.0),
                ([0.0, 0.0], 0.0),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")
        test_df = spark_session.createDataFrame(
            [([1.0, 0.0],), ([0.0, 1.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.predict(test_df, "features", "predictions")
        assert "predictions" in result.columns
        assert result.count() == 2

    def test_prediction_conf_untrained(self, spark_session):
        """Test prediction_conf raises error when model not trained."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, random_state=42)
        df = spark_session.createDataFrame(
            [([1.0, 2.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.prediction_conf(df, "features", "confidence")

    def test_prediction_conf_trained(self, spark_session):
        """Test prediction confidence with trained model."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        train_df = spark_session.createDataFrame(
            [
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 0.0),
                ([1.0, 1.0], 1.0),
                ([0.0, 0.0], 0.0),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")
        test_df = spark_session.createDataFrame(
            [([1.0, 0.0],), ([0.0, 1.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.prediction_conf(test_df, "features", "confidence")
        assert "confidence" in result.columns
        assert result.count() == 2
        confidences = [row["confidence"] for row in result.collect()]
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_entropy_untrained(self, spark_session):
        """Test entropy raises error when model not trained."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, random_state=42)
        df = spark_session.createDataFrame(
            [([1.0, 2.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.entropy(df, "features", "entropy")

    def test_entropy_trained(self, spark_session):
        """Test entropy computation with trained model."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        train_df = spark_session.createDataFrame(
            [
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 0.0),
                ([1.0, 1.0], 1.0),
                ([0.0, 0.0], 0.0),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")
        test_df = spark_session.createDataFrame(
            [([1.0, 0.0],), ([0.0, 1.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.entropy(test_df, "features", "entropy")
        assert "entropy" in result.columns
        assert result.count() == 2
        entropies = [row["entropy"] for row in result.collect()]
        assert all(e >= 0.0 for e in entropies)

    def test_cross_val_scores(self, spark_session):
        """Test cross validation scores."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        df = spark_session.createDataFrame(
            [
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 0.0),
                ([1.0, 1.0], 1.0),
                ([0.0, 0.0], 0.0),
                ([1.0, 0.5], 1.0),
                ([0.5, 1.0], 0.0),
                ([0.5, 0.5], 1.0),
                ([0.8, 0.2], 1.0),
                ([0.2, 0.8], 0.0),
                ([0.9, 0.1], 1.0),
                ([0.3, 0.7], 0.0),
                ([0.7, 0.3], 1.0),
                ([0.4, 0.6], 0.0),
                ([0.6, 0.4], 1.0),
                ([0.1, 0.9], 0.0),
                ([0.9, 0.1], 0.0),
                ([0.2, 0.2], 1.0),
                ([0.8, 0.8], 0.0),
                ([0.3, 0.3], 1.0),
                ([0.7, 0.7], 0.0),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True),
                StructField("label", DoubleType(), True),
            ])
        )
        scores = model.cross_val_scores(df, "features", "label")
        assert len(scores) == 10
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_prep_fvs_with_nan_fill(self, spark_session):
        """Test prep_fvs with nan_fill."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(
            XGBClassifier, nan_fill=0.0, use_floats=True, random_state=42
        )
        df = spark_session.createDataFrame(
            [
                ([1.0, float('nan')],),
                ([float('nan'), 2.0],),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.prep_fvs(df, "features")
        assert "features" in result.columns
        rows = result.collect()
        assert rows[0]["features"] == [1.0, 0.0]
        assert rows[1]["features"] == [0.0, 2.0]
        assert len(rows) == 2

    def test_prep_fvs_with_floats(self, spark_session):
        """Test prep_fvs with float casting."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, use_floats=True, random_state=42)
        df = spark_session.createDataFrame(
            [([1.0, 2.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.prep_fvs(df, "features")
        assert isinstance(result.schema["features"].dataType, ArrayType)
        assert isinstance(
            result.schema["features"].dataType.elementType, FloatType
        )

    def test_prep_fvs_with_doubles(self, spark_session):
        """Test prep_fvs with double casting."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, use_floats=False, random_state=42)
        df = spark_session.createDataFrame(
            [([1.0, 2.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.prep_fvs(df, "features")
        assert isinstance(result.schema["features"].dataType, ArrayType)
        assert isinstance(
            result.schema["features"].dataType.elementType, DoubleType
        )

    def test_prep_fvs_no_nan_fill(self, spark_session):
        """Test prep_fvs without nan_fill."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, nan_fill=None, random_state=42)
        df = spark_session.createDataFrame(
            [([1.0, 2.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.prep_fvs(df, "features")
        assert "features" in result.columns

    def test_init_with_pretrained_model(self):
        """Test SKLearnModel init with pre-trained model."""
        from xgboost import XGBClassifier
        model_instance = XGBClassifier(n_estimators=5, random_state=42)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model_instance.fit(X, y)

        model = SKLearnModel(model_instance)
        assert model._trained_model is not None
        assert model._model_args == {}
        assert model._model == XGBClassifier

    def test_prep_fvs_pandas(self):
        """Test prep_fvs with pandas DataFrame."""
        from xgboost import XGBClassifier
        model = SKLearnModel(
            XGBClassifier, nan_fill=0.0, use_floats=True, random_state=42
        )
        df = pd.DataFrame({
            'features': [[1.0, 2.0], [3.0, 4.0]],
            'other': [1, 2]
        })
        result = model.prep_fvs(df, 'features')
        assert isinstance(result, pd.DataFrame)
        assert 'features' in result.columns

    def test_prep_fvs_pandas_with_nan(self):
        """Test prep_fvs with pandas DataFrame containing NaN."""
        from xgboost import XGBClassifier
        model = SKLearnModel(
            XGBClassifier, nan_fill=0.0, use_floats=True, random_state=42
        )
        df = pd.DataFrame({
            'features': [[1.0, np.nan], [np.nan, 2.0]],
        })
        result = model.prep_fvs(df, 'features')
        assert isinstance(result, pd.DataFrame)
        assert not np.isnan(result['features'].iloc[0][1])

    def test_convert_to_vector_pandas(self):
        """Test convert_to_vector with pandas DataFrame."""
        df = pd.DataFrame({
            'features': [[1.0, 2.0], np.array([3.0, 4.0]), Vectors.dense([5.0, 6.0])]
        })
        result = convert_to_vector(df, 'features')
        assert isinstance(result, pd.DataFrame)
        assert all(isinstance(v, DenseVector) for v in result['features'])
        assert list(result['features'].iloc[0]) == [1.0, 2.0]
        # the input must not be modified in place
        assert isinstance(df['features'].iloc[0], list)

    def test_convert_to_array_pandas(self):
        """Test convert_to_array with pandas DataFrame."""
        df = pd.DataFrame({
            'features': [[1.0, 2.0], np.array([3.0, 4.0]), Vectors.dense([5.0, 6.0])]
        })
        result = convert_to_array(df, 'features')
        assert isinstance(result, pd.DataFrame)
        # spark's schema verifier rejects numpy scalars, so the values have to
        # come back as native python floats
        for v in result['features']:
            assert isinstance(v, list)
            assert all(type(x) is float for x in v)
        assert result['features'].iloc[2] == [5.0, 6.0]

    def test_predict_pandas(self, spark_session):
        """Test predict with pandas DataFrame."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        train_df = spark_session.createDataFrame(
            [
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 0.0),
                ([1.0, 1.0], 1.0),
                ([0.0, 0.0], 0.0),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")

        test_df = pd.DataFrame({
            'features': [[1.0, 0.0], [0.0, 1.0]]
        })
        result = model.predict(test_df, 'features', 'predictions')
        assert isinstance(result, pd.DataFrame)
        assert 'predictions' in result.columns

    def test_predict_with_confidence_pandas(self, spark_session):
        """Test predict_with_confidence with pandas DataFrame."""
        from xgboost import XGBClassifier
        from pyspark.sql.types import StructType, StructField
        model = SKLearnModel(XGBClassifier, n_estimators=10, random_state=42)
        train_df = spark_session.createDataFrame(
            [
                ([1.0, 0.0], 1.0),
                ([0.0, 1.0], 0.0),
                ([1.0, 1.0], 1.0),
                ([0.0, 0.0], 0.0),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")

        test_df = pd.DataFrame({
            'features': [[1.0, 0.0], [0.0, 1.0]]
        })
        result = model.predict_with_confidence(
            test_df, 'features', 'predictions', 'confidence'
        )
        assert isinstance(result, pd.DataFrame)
        assert 'predictions' in result.columns
        assert 'confidence' in result.columns

    def test_make_feature_matrix_with_nan_fill(self):
        """Test _make_feature_matrix with nan_fill."""
        from xgboost import XGBClassifier
        model = SKLearnModel(
            XGBClassifier, nan_fill=0.0, use_floats=True, random_state=42
        )
        vecs = [np.array([1.0, np.nan]), np.array([np.nan, 2.0])]
        X = model._make_feature_matrix(vecs)
        assert X.shape == (2, 2)
        assert not np.isnan(X).any()


class TestSparkMLModel:
    """Tests for SparkMLModel wrapper."""
    def test_init(self):
        """Test SparkMLModel initialization."""
        model = SparkMLModel(GBTClassifier, nan_fill=0.0, maxIter=10)
        assert model.nan_fill == 0.0
        assert model.use_vectors is True
        assert model.use_floats is False
        assert model._model_args == {"maxIter": 10}
        assert model._trained_model is None

    def test_init_defaults(self):
        """Test SparkMLModel with default parameters."""
        model = SparkMLModel(GBTClassifier)
        assert model.nan_fill == 0.0
        assert model.use_vectors is True
        assert model.use_floats is False

    def test_properties(self):
        """Test SparkMLModel properties."""
        model = SparkMLModel(GBTClassifier, nan_fill=5.0)
        assert model.nan_fill == 5.0
        assert model.use_vectors is True
        assert model.use_floats is False

    def test_params_dict(self):
        """Test params_dict returns correct dictionary."""
        model = SparkMLModel(GBTClassifier, maxIter=10, maxDepth=5)
        params = model.params_dict()
        assert params["model_args"] == {"maxIter": 10, "maxDepth": 5}
        assert "GBTClassifier" in str(params["model"])

    def test_get_model(self, spark_session):
        """Test get_model creates model instance."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        spark_model = model.get_model()
        assert isinstance(spark_model, GBTClassifier)
        assert spark_model.getMaxIter() == 10

    def test_train(self, spark_session):
        """Test model training."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField, DoubleType
        df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]), 1.0),
                (Vectors.dense([0.0, 1.0]), 0.0),
                (Vectors.dense([1.0, 1.0]), 1.0),
                (Vectors.dense([0.0, 0.0]), 0.0),
            ],
            schema=StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(df, "features", "label")
        assert model._trained_model is not None
        assert hasattr(model._trained_model, "transform")

    def test_predict_untrained(self, spark_session):
        """Test predict raises error when model not trained."""
        model = SparkMLModel(GBTClassifier)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField
        df = spark_session.createDataFrame(
            [(Vectors.dense([1.0, 2.0]),)],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.predict(df, "features", "predictions")

    def test_predict_trained(self, spark_session):
        """Test prediction with trained model."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField, DoubleType
        train_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]), 1.0),
                (Vectors.dense([0.0, 1.0]), 0.0),
                (Vectors.dense([1.0, 1.0]), 1.0),
                (Vectors.dense([0.0, 0.0]), 0.0),
            ],
            schema=StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")
        test_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]),),
                (Vectors.dense([0.0, 1.0]),),
            ],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        result = model.predict(test_df, "features", "predictions")
        assert "predictions" in result.columns
        assert result.count() == 2
        predictions = [row["predictions"] for row in result.collect()]
        assert all(p is not None for p in predictions)

    def test_prediction_conf_untrained(self, spark_session):
        """Test prediction_conf raises error when model not trained."""
        model = SparkMLModel(GBTClassifier)
        from pyspark.sql.types import StructType, StructField
        df = spark_session.createDataFrame(
            [(Vectors.dense([1.0, 2.0]),)],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.prediction_conf(df, "features", "confidence")

    def test_prediction_conf_trained(self, spark_session):
        """Test prediction confidence with trained model."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField, DoubleType
        train_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]), 1.0),
                (Vectors.dense([0.0, 1.0]), 0.0),
                (Vectors.dense([1.0, 1.0]), 1.0),
                (Vectors.dense([0.0, 0.0]), 0.0),
            ],
            schema=StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")
        test_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]),),
                (Vectors.dense([0.0, 1.0]),),
            ],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        result = model.prediction_conf(test_df, "features", "confidence")
        assert "confidence" in result.columns
        assert result.count() == 2
        confidences = [row["confidence"] for row in result.collect()]
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_entropy_component(self, spark_session):
        """Test entropy component calculation."""
        model = SparkMLModel(GBTClassifier)
        from pyspark.sql.functions import col
        from pyspark.sql.types import (
            StructType, StructField, ArrayType, DoubleType
        )
        df = spark_session.createDataFrame(
            [([0.5, 0.5],), ([0.8, 0.2],), ([0.0, 1.0],)],
            schema=StructType([
                StructField("probs", ArrayType(DoubleType()), True)
            ])
        )
        result = df.withColumn(
            "entropy_0", model._entropy_component(col("probs"), 0)
        )
        rows = result.collect()
        assert len(rows) == 3
        assert rows[0]["entropy_0"] is not None
        assert rows[2]["entropy_0"] == 0.0

    def test_entropy_expr(self, spark_session):
        """Test entropy expression calculation."""
        model = SparkMLModel(GBTClassifier)
        from pyspark.sql.types import (
            StructType, StructField, ArrayType, DoubleType
        )
        df = spark_session.createDataFrame(
            [([0.5, 0.5],), ([0.8, 0.2],)],
            schema=StructType([
                StructField("probs", ArrayType(DoubleType()), True)
            ])
        )
        result = df.withColumn(
            "entropy", model._entropy_expr("probs", classes=2)
        )
        rows = result.collect()
        assert len(rows) == 2
        assert all(row["entropy"] >= 0.0 for row in rows)

    def test_entropy_expr_multiple_classes(self, spark_session):
        """Test entropy expression with multiple classes."""
        model = SparkMLModel(GBTClassifier)
        from pyspark.sql.types import (
            StructType, StructField, ArrayType, DoubleType
        )
        df = spark_session.createDataFrame(
            [([0.33, 0.33, 0.34],), ([0.5, 0.3, 0.2],)],
            schema=StructType([
                StructField("probs", ArrayType(DoubleType()), True)
            ])
        )
        result = df.withColumn(
            "entropy", model._entropy_expr("probs", classes=3)
        )
        rows = result.collect()
        assert len(rows) == 2
        assert all(row["entropy"] >= 0.0 for row in rows)

    def test_entropy_untrained(self, spark_session):
        """Test entropy raises error when model not trained."""
        model = SparkMLModel(GBTClassifier)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField
        df = spark_session.createDataFrame(
            [(Vectors.dense([1.0, 2.0]),)],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.entropy(df, "features", "entropy")

    def test_entropy_trained(self, spark_session):
        """Test entropy computation with trained model."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField, DoubleType
        train_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]), 1.0),
                (Vectors.dense([0.0, 1.0]), 0.0),
                (Vectors.dense([1.0, 1.0]), 1.0),
                (Vectors.dense([0.0, 0.0]), 0.0),
            ],
            schema=StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")
        test_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]),),
                (Vectors.dense([0.0, 1.0]),),
            ],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        result = model.entropy(test_df, "features", "entropy")
        assert "entropy" in result.columns
        assert result.count() == 2
        entropies = [row["entropy"] for row in result.collect()]
        assert all(e >= 0.0 for e in entropies)

    def test_prep_fvs_with_nan_fill(self, spark_session):
        """Test prep_fvs with nan_fill."""
        model = SparkMLModel(GBTClassifier, nan_fill=0.0)
        from pyspark.sql.types import (
            StructType, StructField, ArrayType, DoubleType
        )
        df = spark_session.createDataFrame(
            [
                ([1.0, float('nan')],),
                ([float('nan'), 2.0],),
            ],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.prep_fvs(df, "features")
        assert "features" in result.columns
        assert isinstance(
            result.schema["features"].dataType, VectorUDT
        )

    def test_prep_fvs_no_nan_fill(self, spark_session):
        """Test prep_fvs without nan_fill."""
        model = SparkMLModel(GBTClassifier, nan_fill=None)
        from pyspark.sql.types import (
            StructType, StructField, ArrayType, DoubleType
        )
        df = spark_session.createDataFrame(
            [([1.0, 2.0],)],
            schema=StructType([
                StructField("features", ArrayType(DoubleType()), True)
            ])
        )
        result = model.prep_fvs(df, "features")
        assert "features" in result.columns
        assert isinstance(result.schema["features"].dataType, VectorUDT)

    def test_prep_fvs_vector_input(self, spark_session):
        """Test prep_fvs with vector input."""
        model = SparkMLModel(GBTClassifier, nan_fill=None)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField
        df = spark_session.createDataFrame(
            [(Vectors.dense([1.0, 2.0]),)],
            schema=StructType([StructField("features", VectorUDT(), True)])
        )
        result = model.prep_fvs(df, "features")
        assert isinstance(result.schema["features"].dataType, VectorUDT)

    def test_init_with_pretrained_transformer(self, spark_session):
        """Test SparkMLModel init with pre-trained Transformer."""
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField, DoubleType
        model = SparkMLModel(GBTClassifier, maxIter=10)
        train_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]), 1.0),
                (Vectors.dense([0.0, 1.0]), 0.0),
            ],
            schema=StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", DoubleType(), True),
            ])
        )
        trained = model.train(train_df, "features", "label")

        new_model = SparkMLModel(trained._trained_model)
        assert new_model._trained_model is not None
        assert new_model._model_args == {}

    def test_predict_pandas_sparkml(self, spark_session):
        """Test predict with pandas DataFrame."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField, DoubleType
        train_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]), 1.0),
                (Vectors.dense([0.0, 1.0]), 0.0),
            ],
            schema=StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")

        test_df = pd.DataFrame({
            'features': [
                Vectors.dense([1.0, 0.0]),
                Vectors.dense([0.0, 1.0])
            ]
        })
        result = model.predict(test_df, 'features', 'predictions')
        assert isinstance(result, pd.DataFrame)
        assert 'predictions' in result.columns

    def test_predict_with_confidence_pandas_sparkml(self, spark_session):
        """Test predict_with_confidence with pandas DataFrame."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.types import StructType, StructField, DoubleType
        train_df = spark_session.createDataFrame(
            [
                (Vectors.dense([1.0, 0.0]), 1.0),
                (Vectors.dense([0.0, 1.0]), 0.0),
            ],
            schema=StructType([
                StructField("features", VectorUDT(), True),
                StructField("label", DoubleType(), True),
            ])
        )
        model.train(train_df, "features", "label")

        test_df = pd.DataFrame({
            'features': [
                Vectors.dense([1.0, 0.0]),
                Vectors.dense([0.0, 1.0])
            ]
        })
        result = model.predict_with_confidence(
            test_df, 'features', 'predictions', 'confidence'
        )
        assert isinstance(result, pd.DataFrame)
        assert 'predictions' in result.columns
        assert 'confidence' in result.columns

    def test_train_pandas(self, spark_session):
        """Test train with pandas DataFrame."""
        model = SparkMLModel(GBTClassifier, maxIter=10)
        from pyspark.ml.linalg import Vectors
        train_df = pd.DataFrame({
            'features': [
                Vectors.dense([1.0, 0.0]),
                Vectors.dense([0.0, 1.0])
            ],
            'label': [1.0, 0.0]
        })
        model.train(train_df, 'features', 'label')
        assert model._trained_model is not None
