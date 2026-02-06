"""
This workflows runs Spark on a single machine. It implements the entire matching step for matching DBLP and ACM tables, 
using active learning. It loads the data, creates features, featurizes the candidate set, 
takes a sample, creates seeds, does active learning on the sample in batch mode 
to label more examples, trains a matcher, and applies the matcher. Here we use the gold labeler.
"""


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, sum as spark_sum
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# Import MatchFlow functions
from MatchFlow import (
   create_features, featurize, down_sample, create_seeds,
   train_matcher, apply_matcher, label_data
)
from MatchFlow import GoldLabeler, SKLearnModel


# Initialize Spark session
spark = SparkSession.builder \
   .master("local[*]") \
   .appName("MatchFlow Spark Example") \
   .config('spark.sql.execution.arrow.pyspark.enabled', 'true')\
   .getOrCreate()


# Load data using Spark
table_a = spark.read.parquet('../data/dblp_acm/table_a.parquet')
table_b = spark.read.parquet('../data/dblp_acm/table_b.parquet')
candidates = spark.read.parquet('../data/dblp_acm/cand.parquet')

# In our example, we only need the id2 and id1_list columns.
# Unless it is absolutely necessary to keep other columns that are present in the candidates set, we recommend selecting only the id2 and id1_list columns to reduce memory usage.
candidates = candidates.select('id2', 'id1_list')

gold_labels = spark.read.parquet('../data/dblp_acm/gold.parquet')


# create feature objects 
features = create_features(
   A=table_a,
   B=table_b,
   a_cols=['title', 'authors', 'venue', 'year'],
   b_cols=['title', 'authors', 'venue', 'year']
)


# featurize the candidate pairs
feature_vectors = featurize(
   features=features,
   A=table_a,
   B=table_b,
   candidates=candidates,
   output_col='feature_vectors',
   fill_na=0.0
)


# down sample the feature vectors dataframe
downsampled_fvs = down_sample(
   fvs=feature_vectors,
   percent=0.3,
   search_id_column='_id',
   score_column='score',
   bucket_size=1000
)
# create a gold labeler object
gold_labeler = GoldLabeler(gold=gold_labels)


# create seeds
seeds = create_seeds(
   fvs=downsampled_fvs,
   nseeds=50,
   labeler=gold_labeler,
   score_column='score'
)
print(f"   Created {seeds.count()} initial seeds")
print(f"   Positive seeds: {seeds.filter(seeds['label'] == 1.0).count()}")
print(f"   Negative seeds: {seeds.filter(seeds['label'] == 0.0).count()}")


# specify an ML model
model = SKLearnModel(
   model=XGBClassifier,
   eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42,
   nan_fill=0.0
)
# label data
labeled_data = label_data(
   model=model,
   mode='batch',
   labeler=gold_labeler,
   fvs=downsampled_fvs,
   seeds=seeds,
   batch_size=10,
   max_iter=50
)


# train a matcher
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

predictions.show()

# Calculate metrics
gold_labels = gold_labels.select('id1', 'id2').withColumn('gold_label', lit(1.0))

evaluation_data = predictions.join(
   gold_labels,
   on=['id1', 'id2'],
   how='left'
).fillna(0.0, subset=['gold_label'])


confusion_matrix = evaluation_data.groupBy().agg(
   spark_sum(when((col('gold_label') == 1.0) & (col('prediction') == 1.0), 1).otherwise(0)).alias('tp'),
   spark_sum(when((col('gold_label') == 0.0) & (col('prediction') == 1.0), 1).otherwise(0)).alias('fp'),
   spark_sum(when((col('gold_label') == 1.0) & (col('prediction') == 0.0), 1).otherwise(0)).alias('fn'),
   spark_sum(when((col('gold_label') == 0.0) & (col('prediction') == 0.0), 1).otherwise(0)).alias('tn')
).collect()[0]


tp = confusion_matrix['tp']
fp = confusion_matrix['fp']
fn = confusion_matrix['fn']
tn = confusion_matrix['tn']


precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


print(f"   Evaluation on {evaluation_data.count()} pairs with gold labels:")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1-Score: {f1:.4f}")


# Stop Spark session
spark.stop()
