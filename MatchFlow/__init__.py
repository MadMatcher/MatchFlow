"""
MatchFlow - A toolkit for entity matching.

This package provides tools for creating, training, and applying entity matchers
using various tokenization, featurization, and machine learning techniques.
"""

# Abstract base classes that users can extend
from ._internal.tokenizer.tokenizer import Tokenizer
from ._internal.tokenizer.vectorizer import Vectorizer
from ._internal.feature.feature import Feature
from ._internal.ml_model import MLModel, SKLearnModel, SparkMLModel
from ._internal.labeler import Labeler, CustomLabeler, CLILabeler, GoldLabeler, WebUILabeler

# Public API functions
from .tools import (
    create_features,
    get_base_sim_functions,
    get_base_tokenizers,
    get_extra_tokenizers,
    featurize,
    down_sample,
    create_seeds,
    train_matcher,
    apply_matcher,
    label_data,
    save_features,
    load_features,
    save_dataframe,
    load_dataframe,
    label_pairs
)

# Validation/checking utilities
from ._internal.utils import (
    check_tables,
    check_candidates,
    check_labeled_data,
    check_gold_data
)

__all__ = [
    # Abstract base classes
    'Tokenizer',
    'Vectorizer',
    'Feature',
    'MLModel',
    'SKLearnModel',
    'SparkMLModel',
    'Labeler',
    'CustomLabeler',
    'CLILabeler',
    'GoldLabeler',
    'WebUILabeler',
    
    # Public API functions
    'create_features',
    'get_base_sim_functions',
    'get_base_tokenizers',
    'get_extra_tokenizers',
    'featurize',
    'down_sample',
    'create_seeds',
    'train_matcher',
    'apply_matcher',
    'label_data',
    'label_pairs',
    'save_features',
    'load_features',
    'save_dataframe',
    'load_dataframe',
    
    # Validation/checking utilities
    'check_tables',
    'check_candidates',
    'check_labeled_data',
    'check_gold_data'
] 