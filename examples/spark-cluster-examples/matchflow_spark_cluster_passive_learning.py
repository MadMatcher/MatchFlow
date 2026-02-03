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

# READ THIS IMPORTANT NOTE: Using Sparkly/Delex candidates as MatchFlow input
#
# Sparkly/Delex produce a candidate set with:
#   - <Table B id column>: the id of the Table B record being matched (one row per Table B record)
#   - "ids":              list of candidate ids from Table A for that Table B record
#
# In our reference workflows, the Table B id column is named "_id", so the output columns include:
#   "_id" (Table B id) and "ids" (list of Table A ids)
# If you adapted the workflow to your own data, the Table B id column may have a different name than "_id". 
# You need to know the name of the column that contains the id from table B.
#
# MatchFlow expects the same data (id from a record in table B and a list of ids from table A), but with standardized column names:
#   Sparkly/Delex: (<Table B id column>, ids)  ->  MatchFlow: (id2, id1_list)
#
# Therefore, if you used Sparkly/Delex to block your data, you would need to rename your columns accordingly before calling MatchFlow.
candidates = candidates.withColumnRenamed('_id', 'id2').withColumnRenamed('ids', 'id1_list') # if your Table B id column is "_id"
# If your Table B id column is not "_id", rename that column to "id2" instead:
# candidates = candidates.withColumnRenamed(YOUR_TABLE_B_ID_COL, 'id2').withColumnRenamed('ids', 'id1_list')

# In our example, we only need the id2 and id1_list columns.
# Unless it is absolutely necessary to keep other columns that are present in the candidates set, we recommend selecting only the id2 and id1_list columns to reduce memory usage.
candidates = candidates.select('id2', 'id1_list')

# Read in set P of labeled tuple pairs
labeled_pairs = spark.read.parquet(str(data_dir / 'labeled_pairs.parquet'))


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
