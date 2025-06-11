import pytest
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from active_learning.ent_active_learner import EntropyActiveLearner
from active_learning.cont_entropy_active_learner import ContinuousEntropyActiveLearner
from ml_model import SKLearnModel
from sklearn.ensemble import HistGradientBoostingClassifier
from labeler import GoldLabeler

@pytest.fixture
def spark_session():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .master("local[1]") \
        .appName("test") \
        .getOrCreate()

@pytest.fixture
def sample_data(spark_session):
    """Create sample data for testing."""
    # Create feature vectors
    data = {
        '_id': list(range(1, 11)),
        'id1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'id2': [6, 7, 8, 9, 10, 7, 8, 9, 10, 6],
        'features': [
            [1.0, 0.0, 0.0],  # Match
            [0.0, 1.0, 0.0],  # Non-match
            [0.0, 0.0, 1.0],  # Match
            [1.0, 1.0, 0.0],  # Non-match
            [0.0, 1.0, 1.0],  # Match
            [0.9, 0.1, 0.0],  # Match
            [0.1, 0.9, 0.0],  # Non-match
            [0.0, 0.1, 0.9],  # Match
            [0.9, 0.9, 0.0],  # Non-match
            [0.0, 0.9, 0.9]   # Match
        ]
    }
    return spark_session.createDataFrame(pd.DataFrame(data))

@pytest.fixture
def gold_pairs():
    """Create sample gold standard pairs."""
    return {(1, 6), (3, 8), (5, 10), (1, 7), (3, 9), (5, 6)}

@pytest.fixture
def seeds():
    """Create seed examples for active learning."""
    return pd.DataFrame({
        '_id': [1, 2],
        'id1': [1, 2],
        'id2': [6, 7],
        'features': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        'label': [1.0, 0.0]  # One positive, one negative example
    })

def test_entropy_active_learner_init():
    """Test EntropyActiveLearner initialization."""
    model = SKLearnModel(HistGradientBoostingClassifier())
    labeler = GoldLabeler({(1, 2)})
    
    # Test valid initialization
    learner = EntropyActiveLearner(model, labeler, batch_size=5, max_iter=10)
    assert learner._batch_size == 5
    assert learner._max_iter == 10
    assert learner._model is not None
    assert learner._labeler is not None
    
    # Test invalid batch size
    with pytest.raises(ValueError, match='batch_size must be > 0'):
        EntropyActiveLearner(model, labeler, batch_size=0)
    
    # Test invalid max iterations
    with pytest.raises(ValueError, match='max_iter must be > 0'):
        EntropyActiveLearner(model, labeler, max_iter=0)
    
    # Test invalid model type
    with pytest.raises(TypeError):
        EntropyActiveLearner("not a model", labeler)
    
    # Test invalid labeler type
    with pytest.raises(TypeError):
        EntropyActiveLearner(model, "not a labeler")

def test_entropy_active_learner_train(spark_session, sample_data, gold_pairs, seeds):
    """Test EntropyActiveLearner training process."""
    model = SKLearnModel(HistGradientBoostingClassifier())
    labeler = GoldLabeler(gold_pairs)
    learner = EntropyActiveLearner(model, labeler, batch_size=2, max_iter=3)
    
    # Run active learning
    labeled_data = learner.train(sample_data, seeds)
    
    # Check results
    assert isinstance(labeled_data, pd.DataFrame)
    assert 'label' in labeled_data.columns
    assert 'labeled_in_iteration' in labeled_data.columns
    assert len(labeled_data) > len(seeds)  # Should have more labeled examples than seeds
    
    # Check that all labels are valid
    assert set(labeled_data['label'].unique()).issubset({0.0, 1.0})
    
    # Check that we have both positive and negative examples
    pos_count = (labeled_data['label'] == 1.0).sum()
    neg_count = (labeled_data['label'] == 0.0).sum()
    assert pos_count > 0
    assert neg_count > 0

def test_entropy_active_learner_small_dataset(spark_session, sample_data, gold_pairs, seeds):
    """Test EntropyActiveLearner with a small dataset that would be fully labeled."""
    model = SKLearnModel(HistGradientBoostingClassifier())
    labeler = GoldLabeler(gold_pairs)
    learner = EntropyActiveLearner(model, labeler, batch_size=2, max_iter=3)
    learner._terminate_if_label_everything = True

    # Create a small dataset
    small_data = sample_data.limit(3)

    # Run active learning
    result = learner.train(small_data, seeds)

    # Should have labeled all examples
    assert isinstance(result, pd.DataFrame)
    assert len(result) == small_data.count()
    assert 'label' in result.columns

def test_entropy_active_learner_user_stop(spark_session, sample_data, seeds):
    """Test EntropyActiveLearner when user stops labeling."""
    model = SKLearnModel(HistGradientBoostingClassifier())
    
    # Create a labeler that stops after first batch
    class StopAfterFirstBatch(GoldLabeler):
        def __init__(self, gold_pairs):
            super().__init__(gold_pairs)
            self.called = False
        
        def __call__(self, id1, id2):
            if not self.called:
                self.called = True
                return -1.0  # Stop signal
            return super().__call__(id1, id2)
    
    labeler = StopAfterFirstBatch({(1, 2)})
    learner = EntropyActiveLearner(model, labeler, batch_size=2, max_iter=3)
    
    # Run active learning
    labeled_data = learner.train(sample_data, seeds)
    
    # Should only have seed examples
    assert len(labeled_data) == len(seeds)
    assert all(labeled_data['labeled_in_iteration'] == -1)  # All from seeds

def test_continuous_entropy_active_learner(spark_session, sample_data, gold_pairs, seeds):
    """Test ContinuousEntropyActiveLearner training process."""
    model = SKLearnModel(HistGradientBoostingClassifier())
    
    # Create a labeler that stops after first batch
    class StopAfterFirstBatch(GoldLabeler):
        def __init__(self, gold_pairs):
            super().__init__(gold_pairs)
            self.called = False
        
        def __call__(self, id1, id2):
            if not self.called:
                self.called = True
                return -1.0  # Stop signal
            return super().__call__(id1, id2)
    
    labeler = StopAfterFirstBatch(gold_pairs)
    learner = ContinuousEntropyActiveLearner(
        model,
        labeler,
        queue_size=3,
        max_labeled=100,
        on_demand_stop=True
    )

    # Run active learning
    result = learner.train(sample_data, seeds)

    # Verify results
    assert isinstance(result, pd.DataFrame)
    assert 'label' in result.columns
    assert len(result) == len(seeds)  # Should only have seed examples since we stopped after first batch
    
    # Check that all labels are valid
    assert set(result['label'].unique()).issubset({0.0, 1.0})
    
    # Check that we have both positive and negative examples
    pos_count = (result['label'] == 1.0).sum()
    neg_count = (result['label'] == 0.0).sum()
    assert pos_count > 0
    assert neg_count > 0 