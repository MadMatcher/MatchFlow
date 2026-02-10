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
from MatchFlow import CLILabeler, WebUILabeler, save_dataframe, load_dataframe, label_pairs, check_tables
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

# check that table_a and table_b have '_id' column and the values are unique
check_tables(table_a, table_b)


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
save_dataframe(labeled_pairs, '../data/dblp_acm/labeled_pairs.parquet')

# to load the labeled pairs back in:
labeled_pairs = load_dataframe('../data/dblp_acm/labeled_pairs.parquet', 'sparkdf')
labeled_pairs.show()
