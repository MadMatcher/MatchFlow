import pytest
import pandas as pd
import numpy as np
from tools import down_sample, select_seeds, train_matcher, apply_matcher
from ml_model import SKLearnModel
from sklearn.ensemble import HistGradientBoostingClassifier

class MockLabeler:
    def __init__(self, labels):
        self.labels = labels
        self.current_idx = 0
    
    def __call__(self, id1, id2):
        label = self.labels[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.labels)
        return label

@pytest.fixture
def sample_feature_vectors():
    """Create sample feature vectors for testing."""
    return pd.DataFrame({
        'id1': [1, 2, 3, 4, 5],
        'id2': [6, 7, 8, 9, 10],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5],
        'features': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ]
    })

def test_down_sample(sample_feature_vectors):
    """Test down_sample functionality."""
    # Test 50% downsampling
    result = down_sample(
        sample_feature_vectors,
        percent=0.5,
        search_id_column='id1',
        score_column='score'
    )
    assert len(result) == 3  # Should keep top 50% by score
    assert all(result['score'] >= 0.7)  # Should keep highest scores
    
    # Test 100% downsampling (should return all rows)
    result = down_sample(
        sample_feature_vectors,
        percent=1.0,
        search_id_column='id1',
        score_column='score'
    )
    assert len(result) == len(sample_feature_vectors)
    assert set(result['id1']) == set(sample_feature_vectors['id1'])

def test_select_seeds(sample_feature_vectors):
    """Test select_seeds functionality."""
    # Create a mock labeler that alternates between positive and negative labels
    labeler = MockLabeler([1.0, 0.0, 1.0, 0.0, 1.0])
    
    # Test selecting 4 seeds
    seeds = select_seeds(
        sample_feature_vectors,
        nseeds=4,
        labeler=labeler,
        score_column='score'
    )
    
    assert len(seeds) == 4
    assert 'label' in seeds.columns
    assert set(seeds['label']) == {0.0, 1.0}  # Should have both positive and negative labels
    
    # Test with all positive labels
    labeler = MockLabeler([1.0, 1.0, 1.0, 1.0, 1.0])
    seeds = select_seeds(
        sample_feature_vectors,
        nseeds=3,
        labeler=labeler,
        score_column='score'
    )
    assert len(seeds) == 3
    assert all(seeds['label'] == 1.0)

def test_train_matcher(sample_feature_vectors):
    """Test train_matcher functionality."""
    # Create a model and training data
    model = SKLearnModel(HistGradientBoostingClassifier())
    training_data = pd.DataFrame({
        'features': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        'label': [1.0, 0.0, 1.0]
    })

    # Train the model
    trained_model = train_matcher(
        model,
        training_data,
        feature_col='features',
        label_col='label'
    )

    # Verify the model was trained
    assert trained_model._trained_model is not None
    assert isinstance(trained_model, SKLearnModel)

def test_apply_matcher(sample_feature_vectors):
    """Test apply_matcher functionality."""
    # Create and train a model
    model = SKLearnModel(HistGradientBoostingClassifier())
    training_data = pd.DataFrame({
        'features': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        'label': [1.0, 0.0, 1.0]
    })

    # Train the model first
    trained_model = model.train(
        training_data,
        vector_col='features',
        label_column='label'
    )

    # Apply the model to new data
    print(type(trained_model))
    predictions = apply_matcher(
        trained_model,
        sample_feature_vectors,
        feature_col='features',
        output_col='prediction'
    )

    # Verify predictions
    assert 'prediction' in predictions.columns
    assert len(predictions) == len(sample_feature_vectors)
    assert all(predictions['prediction'].isin([0.0, 1.0]))
