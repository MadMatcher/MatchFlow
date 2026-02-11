"""
This workflow runs Spark on a cluster. It implements the entire matching step, *using passive learning*. 
It reads in Table A, Table B, the candidate set C (which is a set of tuple pairs output by the blocker), 
and a set of labeled tuple pairs P. It then featurizes C and P, trains a matcher M on P, 
then applies M to match the pairs in C. 
 
The schema for the labeled pairs (that is, set P) should be 'id2', 'id1_list', 'label'. 
'label' is a list of labels for each id1 in the id1_list.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, collect_list
from xgboost import XGBClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import MatchFlow functions
from MatchFlow import (
    create_features, featurize, 
    train_matcher, apply_matcher,
    check_tables, check_candidates, check_labeled_data
)
from MatchFlow import SKLearnModel

# Initialize Spark session
spark = SparkSession.builder \
    .master("{url of spark master node}") \
    .appName("MatchFlow Spark Passive Learning Example") \
    .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \
    .getOrCreate()

# Load data using Spark
data_dir = Path(__file__).resolve().parent.parent / 'data' / 'dblp_acm'
table_a = spark.read.parquet(str(data_dir / 'table_a.parquet'))
table_b = spark.read.parquet(str(data_dir / 'table_b.parquet'))
candidates = spark.read.parquet(str(data_dir / 'cand.parquet'))


# In our example, we only need the id2 and id1_list columns.
# Unless it is absolutely necessary to keep other columns that are present in the candidates set, we recommend selecting only the id2 and id1_list columns to reduce memory usage.
candidates = candidates.select('id2', 'id1_list')

# Read in set P of labeled tuple pairs
labeled_pairs = spark.read.parquet(str(data_dir / 'labeled_pairs.parquet'))

# Validate that both table_a and table_b contain a column named '_id',
# and that this column is non-null and unique within each table.
# This check must be performed before invoking any core MatchFlow functions.
try:
    check_tables(table_a, table_b)
except ValueError as e:
    print(e)
    exit(1)

# Validate that candidates has 'id2' and 'id1_list' columns with valid IDs
# This check ensures the candidates table is properly formatted for featurization
try:
    check_candidates(candidates, table_a, table_b)
except ValueError as e:
    print(e)
    exit(1)

# Validate that labeled_pairs has the required columns and structure for passive learning
# This check verifies 'id2', 'id1_list', and 'label' columns exist and are in the correct format
try:
    check_labeled_data(labeled_pairs, table_a, table_b, 'label')
except ValueError as e:
    print(e)
    exit(1)

# Create features
features = create_features(
    A=table_a,
    B=table_b,
    a_cols=['title', 'authors', 'venue', 'year'],
    b_cols=['title', 'authors', 'venue', 'year']
)

# Featurize labeled pairs
labeled_pairs_feature_vectors = featurize(
    features=features,
    A=table_a,
    B=table_b,
    candidates=labeled_pairs,  # Use grouped data with id1_list
    output_col='feature_vectors',
    fill_na=0.0
)

# Featurize candidate set
candidate_feature_vectors = featurize(
    features=features,
    A=table_a,
    B=table_b,
    candidates=candidates,
    output_col='feature_vectors',
    fill_na=0.0
)

# Create ML model
model = SKLearnModel(
    model=XGBClassifier,
    eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42,
    nan_fill=0.0
)

# Train matcher
trained_model = train_matcher(
    model=model,
    labeled_data=labeled_pairs_feature_vectors,
    feature_col='feature_vectors',
    label_col='label'
)

# Apply matcher
predictions = apply_matcher(
    model=trained_model,
    df=candidate_feature_vectors,
    feature_col='feature_vectors',
    prediction_col='prediction',
    confidence_col='confidence'
)

predictions.show()
