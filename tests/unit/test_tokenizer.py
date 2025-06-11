import pytest
import pandas as pd
import numpy as np
from tokenizer import (
    AlphaNumericTokenizer,
    NumericTokenizer,
    WhiteSpaceTokenizer,
    ShingleTokenizer,
    QGramTokenizer
)
from tokenizer.vectorizer import TFIDFVectorizer
from utils import PerfectHashFunction
from feature import DocFreqBuilder

@pytest.fixture
def sample_text():
    """Create sample text data for testing."""
    return pd.DataFrame({
        'text': [
            'Hello World 123',
            'Python 3.9 Code',
            'Test Case 456',
            'Mixed Text 789',
            'Special Chars @#$',
            'Numbers 123 456',
            'Empty String',
            None
        ]
    })

def test_alpha_numeric_tokenizer(sample_text):
    """Test AlphaNumericTokenizer functionality."""
    tokenizer = AlphaNumericTokenizer()
    
    # Test tokenization
    tokens = tokenizer.tokenize(sample_text['text'].iloc[0])
    assert set(tokens) == {'hello', 'world', '123'}
    
    # Test case insensitivity
    tokens = tokenizer.tokenize('Hello WORLD')
    assert set(tokens) == {'hello', 'world'}
    
    # Test handling of special characters
    tokens = tokenizer.tokenize('Hello@World#123')
    assert set(tokens) == {'hello', 'world', '123'}
    
    # Test empty string
    tokens = tokenizer.tokenize('')
    assert len(tokens) == 0
    
    # Test None
    tokens = tokenizer.tokenize(None)
    assert tokens is None

def test_numeric_tokenizer(sample_text):
    """Test NumericTokenizer functionality."""
    tokenizer = NumericTokenizer()
    
    # Test tokenization
    tokens = tokenizer.tokenize(sample_text['text'].iloc[0])
    assert set(tokens) == {'123'}
    
    # Test decimal numbers
    tokens = tokenizer.tokenize('3.14 42')
    assert set(tokens) == {'3', '14', '42'}
    
    # Test non-numeric text
    tokens = tokenizer.tokenize('Hello World')
    assert len(tokens) == 0
    
    # Test mixed content
    tokens = tokenizer.tokenize('Price: $19.99')
    assert set(tokens) == {'19', '99'}
    
    # Test empty string
    tokens = tokenizer.tokenize('')
    assert len(tokens) == 0
    
    # Test None
    tokens = tokenizer.tokenize(None)
    assert tokens is None

def test_white_space_tokenizer(sample_text):
    """Test WhiteSpaceTokenizer functionality."""
    tokenizer = WhiteSpaceTokenizer()
    
    # Test tokenization
    tokens = tokenizer.tokenize(sample_text['text'].iloc[0])
    assert set(tokens) == {'hello', 'world', '123'}
    
    # Test multiple spaces
    tokens = tokenizer.tokenize('Hello   World')
    assert set(tokens) == {'hello', 'world'}
    
    # Test tabs and newlines
    tokens = tokenizer.tokenize('Hello\tWorld\n123')
    assert set(tokens) == {'hello', 'world', '123'}
    
    # Test empty string
    tokens = tokenizer.tokenize('')
    assert len(tokens) == 0
    
    # Test None
    tokens = tokenizer.tokenize(None)
    assert tokens is None

def test_shingle_tokenizer(sample_text):
    """Test ShingleTokenizer functionality."""
    tokenizer = ShingleTokenizer(n=2)
    
    # Test tokenization
    tokens = tokenizer.tokenize('hello world 123')
    assert set(tokens) == {'helloworld'}
    
    # Test empty string
    tokens = tokenizer.tokenize('')
    assert len(tokens) == 0
    
    # Test None input
    tokens = tokenizer.tokenize(None)
    assert tokens is None

def test_qgram_tokenizer(sample_text):
    """Test QGramTokenizer functionality."""
    tokenizer = QGramTokenizer(n=2)
    
    # Test tokenization
    tokens = tokenizer.tokenize(sample_text['text'].iloc[0])
    assert set(tokens) == {'he', 'el', 'll', 'lo', 'o ', ' w', 'wo', 'or', 'rl', 'ld', 'd ', ' 1', '12', '23'}
    
    # Test different q-gram size
    tokenizer = QGramTokenizer(n=3)
    tokens = tokenizer.tokenize('Hello')
    assert set(tokens) == {'hel', 'ell', 'llo'}
    
    # Test string shorter than q
    tokenizer = QGramTokenizer(n=5)
    tokens = tokenizer.tokenize('Hi')
    assert len(tokens) == 0
    
    # Test empty string
    tokens = tokenizer.tokenize('')
    assert len(tokens) == 0
    
    # Test None
    tokens = tokenizer.tokenize(None)
    assert tokens is None


def test_perfect_hash_function():
    """Test PerfectHashFunction functionality."""
    # Test hash function creation
    keys = ['hello', 'world', 'test']
    hash_func, hash_vals = PerfectHashFunction.create_for_keys(keys)
    
    # Test hash values
    assert len(hash_vals) == len(keys)
    assert len(set(hash_vals)) == len(keys)  # No collisions
    
    # Test hash function
    for key in keys:
        h = hash_func.hash(key)
        assert h in hash_vals
    
    # Test non-existent key
    h = hash_func.hash('nonexistent')
    assert h not in hash_vals
    
    # Test duplicate keys
    with pytest.raises(ValueError, match='keys must be unique'):
        PerfectHashFunction.create_for_keys(['a', 'b', 'a'])
