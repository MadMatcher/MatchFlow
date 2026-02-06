"""
This workflow runs Spark on a cluster. It shows how you can use MatchFlow functions, 
especially labeled_pairs() to label a set of tuple pairs.  It loads the data, then asks 
the user to label the tuple pairs in the candidate set, using the CLI labeler (or the Web labeler). 
Note that it is straightforward to modify this script to take a sample from the candidate set 
using down_sample(), then asks the user to label only the pairs in the sample. 
"""

import warnings
from pyspark.sql import SparkSession
from pathlib import Path
import pyspark.sql.functions as F
# Import MatchFlow functions
from MatchFlow import WebUILabeler, save_dataframe, load_dataframe, label_pairs
warnings.filterwarnings('ignore')

spark = SparkSession.builder \
   .master("{url of spark master node}") \
   .appName("MatchFlow Spark Labeling Only") \
   .config('spark.sql.execution.arrow.pyspark.enabled', 'true')\
   .getOrCreate()

# Load data
data_dir = Path(__file__).resolve().parent.parent / 'data' / 'dblp_acm'
table_a = spark.read.parquet(str(data_dir / 'table_a.parquet'))
table_b = spark.read.parquet(str(data_dir / 'table_b.parquet'))
candidates = spark.read.parquet(str(data_dir / 'cand.parquet'))

# Convert from id2: id1_list to id1: id2 pairs.
# label_pairs() expects columns named "id1" and "id2", with "id1" as the first (left) column and "id2" as the second.
candidates = candidates.select(F.explode('id1_list').alias('id1'), 'id2')

# with default settings, you can open the web UI labeler by going to:
# http://{public ip of spark master node}:8501
# from your local machine's browser
labeler = WebUILabeler(
    a_df=table_a,
    b_df=table_b,
    id_col='_id'
)

labeled_pairs = label_pairs(
    labeler=labeler,
    pairs=candidates
)

labeled_pairs = spark.createDataFrame(labeled_pairs)

# Save the labeled pairs to a parquet file on the master node
save_path = str(data_dir / 'labeled_pairs.parquet')
save_dataframe(labeled_pairs, save_path)

# to load the labeled pairs back in from the master node:
labeled_pairs = load_dataframe(save_path, 'sparkdf')
labeled_pairs.show()
