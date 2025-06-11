import pytest
import pandas as pd
import time
from labeler import GoldLabeler, DelayedGoldLabeler, CLILabeler, CustomLabeler, Labeler

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'a_df': pd.DataFrame({
            '_id': [1, 2, 3],
            'name': ['John Smith', 'Jane Doe', 'Bob Johnson'],
            'age': [30, 25, 35],
            'city': ['New York', 'Boston', 'Chicago']
        }),
        'b_df': pd.DataFrame({
            '_id': [4, 5, 6],
            'name': ['John Smith', 'Jane Smith', 'Bob Johnson'],
            'age': [30, 28, 35],
            'city': ['NYC', 'Boston', 'Chicago']
        })
    }

@pytest.fixture
def gold_pairs():
    """Create sample gold standard pairs."""
    return {(1, 4), (3, 6)}  # (1,4) and (3,6) are matches

def test_gold_labeler(gold_pairs):
    """Test GoldLabeler functionality."""
    labeler = GoldLabeler(gold_pairs)
    
    # Test matching pairs
    assert labeler(1, 4) == 1.0
    assert labeler(3, 6) == 1.0
    
    # Test non-matching pairs
    assert labeler(1, 5) == 0.0
    assert labeler(2, 4) == 0.0
    assert labeler(2, 5) == 0.0

def test_delayed_gold_labeler(gold_pairs):
    """Test DelayedGoldLabeler functionality."""
    delay = 0.1  # 100ms delay
    labeler = DelayedGoldLabeler(gold_pairs, delay)
    
    # Test timing
    start = time.time()
    label = labeler(1, 4)
    end = time.time()
    
    assert label == 1.0
    assert end - start >= delay  # Should take at least delay seconds
    
    # Test non-matching pair
    start = time.time()
    label = labeler(1, 5)
    end = time.time()
    
    assert label == 0.0
    assert end - start >= delay

class MockCLILabeler(CLILabeler):
    def __init__(self, a_df, b_df, responses):
        super().__init__(a_df, b_df)
        self.responses = responses
        self.response_index = 0

    def __call__(self, id1, id2):
        # Simulate CLILabeler behavior without actual user input
        response = self.responses[self.response_index]
        self.response_index = (self.response_index + 1) % len(self.responses)
        
        if response == 'y':
            return 1.0
        elif response == 'n':
            return 0.0
        elif response == 'u':
            return 2.0
        elif response == 's':
            return -1.0
        elif response == 'h':
            # For help, get next response
            next_response = self.responses[self.response_index]
            self.response_index = (self.response_index + 1) % len(self.responses)
            return 1.0 if next_response == 'y' else 0.0
        else:
            raise ValueError(f"Unexpected response: {response}")

def test_custom_labeler(spark_session, sample_data):
    """Test CustomLabeler functionality."""
    # Convert pandas DataFrames to Spark DataFrames
    a_df = spark_session.createDataFrame(sample_data['a_df'])
    b_df = spark_session.createDataFrame(sample_data['b_df'])
    
    labeler = TestCustomLabeler(a_df, b_df)

    # Test exact match
    assert labeler(1, 4) == 1.0  # John Smith, same age
    assert labeler(2, 5) == 0.0  # Jane Doe vs Jane Smith
    assert labeler(3, 6) == 1.0  # Bob Johnson, same age

    # Test error handling
    with pytest.raises(KeyError):
        labeler(999, 4)  # Non-existent ID

def test_cli_labeler(spark_session, sample_data):
    """Test CLILabeler functionality with mock input."""
    # Convert pandas DataFrames to Spark DataFrames
    a_df = spark_session.createDataFrame(sample_data['a_df'])
    b_df = spark_session.createDataFrame(sample_data['b_df'])
    
    # Test with yes/no/unsure responses
    labeler = MockCLILabeler(
        a_df,
        b_df,
        ['y', 'n', 'u', 's', 'y', 'y']
    )

    # Test yes response
    assert labeler(1, 4) == 1.0

    # Test no response
    assert labeler(2, 5) == 0.0

    # Test unsure response
    assert labeler(3, 6) == 2.0

    # Test stop response
    assert labeler(1, 4) == -1.0

    # Test help response
    labeler = MockCLILabeler(a_df, b_df, ['h', 'y'])
    assert labeler(1, 4) == 1.0  # Should show help and then accept yes

class TestCustomLabeler(CustomLabeler):
    """Test implementation of CustomLabeler."""
    def label_pair(self, row1, row2):
        # Simple matching logic: match if names are similar and ages are equal
        name1 = row1['name'].lower()
        name2 = row2['name'].lower()
        age_match = row1['age'] == row2['age']
        
        if name1 == name2 and age_match:
            return 1.0
        else:
            return 0.0
