"""
This workflows runs Pandas on a single machine. It implements the entire matching step for matching DBLP and ACM tables, 
using active learning. It loads the data, creates features, featurizes the candidate set, 
takes a sample, creates seeds, does active learning on the sample in batch mode 
to label more examples, trains a matcher, and applies the matcher. Here we use the gold labeler.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import MatchFlow functions
from MatchFlow import (
    create_features, featurize, down_sample, create_seeds, 
    train_matcher, apply_matcher, label_data
)
from MatchFlow import GoldLabeler, SKLearnModel

# Load data
table_a = pd.read_parquet('../data/dblp_acm/table_a.parquet')
table_b = pd.read_parquet('../data/dblp_acm/table_b.parquet')
candidates = pd.read_parquet('../data/dblp_acm/cand.parquet')
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
candidates = candidates.rename(columns={"_id": "id2", "ids": "id1_list"})  # if your Table B id column is "_id"
# If your Table B id column is not "_id", rename that column to "id2" instead:
# candidates = candidates.rename(columns={YOUR_TABLE_B_ID_COL: "id2", "ids": "id1_list"})

# In our example, we only need the id2 and id1_list columns.
# Unless it is absolutely necessary to keep other columns that are present in the candidates set, we recommend selecting only the id2 and id1_list columns to reduce memory usage.
candidates = candidates[['id2', 'id1_list']]

gold_labels = pd.read_parquet('../data/dblp_acm/gold.parquet')

# Create features
features = create_features(
    A=table_a,
    B=table_b,
    a_cols=['title', 'authors', 'venue', 'year'],
    b_cols=['title', 'authors', 'venue', 'year']
)

# Featurize
feature_vectors = featurize(
    features=features,
    A=table_a,
    B=table_b,
    candidates=candidates,
    output_col='feature_vectors',
    fill_na=0.0
)

# Downsample
downsampled_fvs = down_sample(
    fvs=feature_vectors,
    percent=0.3,
    search_id_column='_id',
    score_column='score',
    bucket_size=1000
)

# Create gold labeler
gold_labeler = GoldLabeler(gold=gold_labels)

# Create seeds
seeds = create_seeds(
    fvs=downsampled_fvs,
    nseeds=50,
    labeler=gold_labeler,
    score_column='score'
)

# Create ML model
model = SKLearnModel(
    model=XGBClassifier,
    eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42,
    nan_fill=0.0
)

# Label data
labeled_data = label_data(
    model=model,
    mode='batch',
    labeler=gold_labeler,
    fvs=downsampled_fvs,
    seeds=seeds,
    batch_size=10,
    max_iter=50
)

# Train matcher
trained_model = train_matcher(
    model=model,
    labeled_data=labeled_data,
    feature_col='feature_vectors',
    label_col='label'
)

# Apply matcher
predictions = apply_matcher(
    model=trained_model,
    df=feature_vectors,
    feature_col='feature_vectors',
    prediction_col='prediction',
    confidence_col='confidence'
)


# Calculate evaluation metrics
gold_dict = {}
for _, row in gold_labels.iterrows():
    gold_dict[(row['id1'], row['id2'])] = 1.0

predictions['gold_label'] = predictions.apply(
    lambda row: gold_dict.get((row['id1'], row['id2']), 0.0), axis=1
)

evaluation_data = predictions[predictions['gold_label'] >= 0.0]

y_true = evaluation_data['gold_label'].values
y_pred = evaluation_data['prediction'].values

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
