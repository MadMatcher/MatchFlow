"""
This workflow runs Spark on a single machine. It shows how you can use MatchFlow functions, 
especially labeled_pairs() to label a set of tuple pairs.  It loads the data, then asks 
the user to label the tuple pairs in the candidate set, using the CLI labeler (or the Web labeler). 
Note that it is straightforward to modify this script to take a sample from the candidate set 
using down_sample(), then asks the user to label only the pairs in the sample. 
"""

import warnings
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
# Import MatchFlow functions
from MatchFlow import CLILabeler, WebUILabeler, save_dataframe, load_dataframe, label_pairs
warnings.filterwarnings('ignore')

spark = SparkSession.builder \
   .master("local[*]") \
   .appName("MatchFlow Spark Labeling Only") \
   .config('spark.sql.execution.arrow.pyspark.enabled', 'true')\
   .getOrCreate()

# Load data
table_a = spark.read.parquet('../data/dblp_acm/table_a.parquet')
table_b = spark.read.parquet('../data/dblp_acm/table_b.parquet')
candidates = spark.read.parquet('../data/dblp_acm/cand.parquet')
candidates = candidates.withColumnRenamed('_id', 'id2').withColumnRenamed('ids', 'id1_list')

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

# Convert from id2: id1_list to id1: id2 pairs.
# label_pairs() expects columns named "id1" and "id2", with "id1" as the first (left) column and "id2" as the second.
candidates = candidates.select(F.explode('id1_list').alias('id1'), 'id2')


# Create CLI labeler
labeler = CLILabeler(
    a_df=table_a,
    b_df=table_b,
    id_col='_id'
)

"""
# Or, uncomment this block to create Web UI labeler
labeler = WebUILabeler(
    a_df=table_a,
    b_df=table_b,
    id_col='_id'
)
"""

labeled_pairs = label_pairs(
    labeler=labeler,
    pairs=candidates
)

# Save the labeled pairs to a parquet file
save_dataframe(labeled_pairs, 'labeled_pairs.parquet')

# to load the labeled pairs back in:
labeled_pairs = load_dataframe('labeled_pairs.parquet', 'sparkdf')
labeled_pairs.show()
