## MatchFlow Technical Guide

This document provides a comprehensive overview of the functions in MatchFlow: how they work and how to use them. After you have installed MatchFlow, we recommend that you read this document, then study the [sample Python scripts](https://github.com/MadMatcher/MatchFlow/blob/main/docs/workflow-examples.md) that implement various matching workflows. This should give you sufficient knowledge to implement your own matching workflows.

### Preliminaries

We start by discussing the basic concepts underlying MatchFlow. You can skip this section if you are already familiar with them.

#### The Matching Step

Let A and B be two tables that we want to match, that is, find tuple pairs (x,y) where tuple x of A matches tuple y of B. We refer to such pair (x,y) as a _match_. We assume blocking has been performed on A and B, producing a set C of <u>candidate tuple pairs</u> (we call these pairs "candidates" because each may be a candidate for a match).

Now we enter the matching step, in which we will apply a rule- or machine-learning (ML) based matcher to each pair (x,y) in C to predict match/non-match. Today ML-based matchers are most common, so in MatchFlow we provide support for these matchers. The overall matching workflow is as follows:

1. We create a set of features, then convert each pair of tuples (x,y) in C into a feature vector. Let D be the set of all feature vectors.
2. We create training data T, which is a set of tuple pairs where each pair has been labeled match/non-match.
3. We convert each pair in T into a feature vector, then use them to train a ML classification model M. We will refer to M as "matcher".
4. We apply matcher M to each feature vector in D to predict whether the vector (and thus the corresponding pair (x,y) in C) is a match/non-match.

#### Challenges

The above workflow is a very standard ML workflow for classification. For the EM setting it raises the following challenges:

- **How to create the features?** We do this by using a set of heuristics that analyze the columns of Tables A and B, and use a set of well-known similarity functions and tokenizers. We discuss more below.
- **How to create training data?** There are two scenarios:
  - You may already have a set of labeled tuple pairs (perhaps obtained from a related project). In that case you can just use that data. We call this **passive learning**.
  - But a more common scenario is that you do not have anything yet, and you don't know where to start. In general, you cannot just take a random sample of tuple pairs from the candidate set C then label them, because it is very likely that this sample will contain very few true matches, and thus is not a very good sample for training matcher M. We provide a solution to this problem that examines the tuple pairs in the set C to select a relatively small set of "good" examples (that is, tuple pairs) for you to label. This is called **active learning** in the ML literature.
- **How to scale?** If Tables A and B have millions of tuples (which is very commmon in practice), the candidate set C (obtained after blocking) can be huge, having 100M to 1B tuple pairs or more. This would create several serious problems:
  - Featurizing C, that is, converting each tuple pair in C into a feature vector can take hours or days. We provide a fast solution to this problem, using Spark on a cluster of machines.
  - Using active learning to select a few hundreds "good" tuple pairs from the candiate set C for you to label is going to be very slow, because C is so large and we have to examine all tuple pairs in C to select the "good" ones. We provide a solution that takes a far smaller sample S from C (having say just 5M tuple pairs), then performs active learning on S (not on C) to select "good" examples for you to label. As mentioned earlier, S cannot be a random sample of C because in that case it is likely to contain very few true matches, and thus is not a good sample from which to select "good" examples to label.
  - The default way that we do active learning, say on S, is in the **batch mode**. That is, we examine all examples in S, select 10 "good" examples, ask you to label them as match/non-match, then re-examine the examples in S, select another 10 "good" examples, ask you to label those, and so on. This sometimes has a lag time: you label 10 examples, submit, then _must wait a few seconds before you are given the next 10 examples to label_. To avoid such lag time, we provide a solution that do active learning in the **continuous mode**. In this mode, you do not have to wait and can just label continuously. The downside is that you may have to label a bit more examples (compared to the batch mode) to reach the same matching accuracy.

- **What are the runtime environments?** We provide three runtime environments:
  - **Spark on a cluster of machines**: This is well suited for when you have a lot of data.
  - **Spark on a single machine**: Use this if you do not have a lot of data, or if you want to test your Python script before you run it on a cluster of machines.
  - **Pandas on a single machine**: Use this if you do not have a lot of data and you do not know Spark or do not want to run it.

- **How to label examples?** We provide three labelers: gold, CLI, and Web-based.
  - The gold labeler assumes you have the set of all true matches between Tables A and B. We call this set **the gold set**. Given a pair of tuples (x,y) to be labeled, the gold labeler just consults this set, and return 1.0 (that is, match) if (x,y) is in the gold set, and return 0.0 (that is, non-match) otherwise. The gold labeler is commonly used to test and debug code and for computing matching accuracy.
  - Given a pair (x,y) to be labeled, the CLI (command-line interface) labeler will display this pair on the CLI and ask you to label the pair as match, non-match, or unsure. This labeler only works if you run Pandas or Spark on a single machine. It will not work for Spark on a cluster (use the Web labeler below instead).
  - The Web-baser labeler runs a browser and a Web server. When a MatchFlow function wants you to label a pair of tuples (x,y), it sends this pair to the Web server, which in turn sends it to the browser, where you can label the pair as match, non-match, or unsure. If you run on a local machine, then the Web server will run locally on that machine. If you run Spark on a cluster, then the Web server is likely to run on the master node (assuming that you submit the Python script on this node).

#### Different EM Workflows

You can combine the MatchFlow functions (in a Python script) to create a variety of EM workflows. For example:

- A Pandas workflow that uses active learning to find and label a set of examples, use them to train a matcher M, then apply M to predict match/non-match for all examples in the candidate set C.
- A variation of the above workflow that uses Spark on a cluster of machines to scale to a very large set C.
- A workflow in which the user has been given a set of labeled examples. The workflow trains a matcher M on these examples then apply it to the examples in C.
- A workflow in which you just want to label a set of examples.

We provide [Python scripts for several such workflows](https://github.com/MadMatcher/MatchFlow/blob/main/docs/workflow-examples.md). More workflows can be constructed using MatchFlow functions.

### The Core Functions of MatchFlow

We now describe the core functions that you can combine to create a variety of EM workflows.

### create_features()

```python
def create_features(
    A: Union[pd.DataFrame, SparkDataFrame],         # Your first dataset
    B: Union[pd.DataFrame, SparkDataFrame],         # Your second dataset
    a_cols: List[str],                              # Columns from A to compare
    b_cols: List[str],                              # Columns from B to compare
    sim_functions: Optional[List[Callable]] = None, # List of similarity functions
    tokenizers: Optional[List[Tokenizer]] = None,   # List of tokenizers
    null_threshold: float = 0.5                     # Max null percentage allowed
) -> List[Callable]
```

This function uses heuristics that analyzes the columns of Tables A and B to create a set of features. The features uses a combination of similarity functions and tokenizers. See [here](./sim-functions-tokenizers.md) for a brief discussion of similarity functions and tokenizers for MatchFlow.

- Parameters A and B are Pandas or Spark dataframes (depending on whether your runtime environment is Pandas on a single machine, Spark on a single machine, or Spark on a cluster of machines). They are the two tables to match. Both dataframes must contain an `_id` column.
- a_cols and b_cols are lists of column names from A and B. We use these columns to create features. As of now, a_cols and b_cols must be the same list. But in the future we will modify the function create_features to handle the case where these two lists can be different.
- sim_functions is a list of sim_functions (if provided, this will override the default sim functions used by MatchFlow, see below).
- tokenizers is a list of tokenizers (if provided, this will override the default tokenizers used by MatchFlow, see below).
- null_threshold is a number in [0,1] specifying the maximal fraction of missing values allowed in a column. MatchFlow automatically excludes columns with too much missing data from feature generation (see below).

The above function returns a list of features. Each feature is an executable object (as indicated by the `Callable` keyword).

**Example:** Suppose A has columns `['_id', 'name', 'address', 'phone']` and B has columns `['_id', 'name', 'address', 'phone', 'comment']`. Then both a_cols and b_cols can be `['name', 'address', 'phone']`, meaning we only want to create features involving these columns. Suppose null_threshold = 0.6, and suppose that column 'phone' has more than 70% of its values missing. Then we drop 'phone' and will create features involving just 'name' and 'address'.

**Creating Features for the Default Case:** If the parameter sim_function has value None, then we use a set of default similarity functions. As of now this set is the following five functions:

- `TFIDFFeature`: Term frequency-inverse document frequency similarity
- `JaccardFeature`: Set-based similarity using intersection over union
- `SIFFeature`: Smooth inverse frequency similarity
- `OverlapCoeffFeature`: Set overlap coefficient
- `CosineFeature`: Vector space cosine similarity

If the parameter tokenizers has value None, then we use a set of default tokenizers. As of now this set is the following three functions:

- `StrippedWhiteSpaceTokenizer()`: Splits on whitespace and normalizes
- `NumericTokenizer()`: Extracts numeric values from text
- `QGramTokenizer(3)`: Creates 3-character sequences (3-grams)

In this case, we create the features as follows:

- First, we detect and drop all columns with too many missing values (above null_threshold). Then we analyze the remaining columns to detect their types (e.g., numeric vs text). We also compute the average token count for each tokenizer-column combination.
- Then we create the following features:
  - _Exact Match Features_: Created for all columns.
  ```python
  # Every column gets an exact match feature
  ExactMatchFeature(column_name, column_name)
  ```

  - _Numeric Features_: Created for columns that are detected as numeric.
    ```python
    # Numeric columns get relative difference features
    RelDiffFeature(column_name, column_name)
    ```
  - _Token-based Features_: Created based on tokenizer analysis
  ```python
  # For each tokenizer-column combination with avg_count >= 3:
  # Creates the five features using the default similarity functions (TF-IDF, Jaccard, SIF, Overlap, Cosine)
  TFIDFFeature(column_name, column_name, tokenizer=tokenizer)
  JaccardFeature(column_name, column_name, tokenizer=tokenizer)
  SIFFeature(column_name, column_name, tokenizer=tokenizer)
  OverlapCoeffFeature(column_name, column_name, tokenizer=tokenizer)
  CosineFeature(column_name, column_name, tokenizer=tokenizer)
  ```

**Creating Features for Other Cases:** You can override the similarity functions using the parameter sim_functions and override the tokenizers using the parameter tokenizers.

In particular, we have implemented the following non-default tokenizers that you can use:

- `AlphaNumericTokenizer()`: Extracts alphanumeric sequences
- `QGramTokenizer(5)`: Creates 5-character sequences (5-grams)
- `StrippedQGramTokenizer(3)`: Creates 3-grams with whitespace stripped
- `StrippedQGramTokenizer(5)`: Creates 5-grams with whitespace stripped

In the case where the user specifies similarity functions and/or tokenizers, we create the features as follows:

- We still create the exact match features and the numeric features, as described in the default case.
- We will create features for each similarity function and tokenizer combination provided. If only a list of similarity functions are provided, the features will be created by combinations of the provided similarity functions with the default tokenizers. Similarly, if only a list of tokenizers are provided, the features will be created by combinations of the default similarity functions with the provided tokenizers. If both similarity functions and tokenizers are provied, the features will be created by combinations of the provided similarity functions and the provided tokenizers. Each of the generated features will be applied to columns where the average number of tokens produced by the feature's tokenizer is at least 3.
- Finally, we create the following special features if you have included the AlphaNumeric tokenizer.

  ```python
  # Only for AlphaNumericTokenizer with avg_count <= 10:
  MongeElkanFeature(column_name, column_name, tokenizer=tokenizer)
  EditDistanceFeature(column_name, column_name)
  SmithWatermanFeature(column_name, column_name)
  ```

  **Example:** The following is a simple example of creating features using the default similarity functions and tokenizers:

  ```python
  features = create_features(
      customers_df,
      prospects_df,
      ['name', 'address', 'revenue'],
      ['name', 'address', 'revenue']
  )

  # What happens behind the scenes:
  # 1. All columns get ExactMatchFeature
  # 2. If 'revenue' is numeric, it gets RelDiffFeature
  # 3. For each tokenizer (Defaults: StrippedWhiteSpace, Numeric, QGram):
  #    - If average token count >= 3 over the column, creates features for all 5 default similarity functions


  # Typical output might include:
  # - ExactMatchFeature for name, address, revenue
  # - RelDiffFeature for revenue (if numeric)
  # - TFIDFFeature, JaccardFeature, SIFFeature, OverlapCoeffFeature, CosineFeature
  #   for each tokenizer-column combination with sufficient tokens
  ```

If you want to use some set of our default similarity functions, default tokenizers, or extra tokenizers, we provide the following helper methods:

```
get_base_sim_functions() # returns the default similarity function classes used in MatchFlow: TFIDF, Jaccard, SIF, OverlapCoeffecient, Cosine
```

```
get_base_tokenizers() # returns the default tokenizer classes used in MatchFlow: StrippedWhiteSpace, Numeric, 3gram
```

```
get_extra_tokenizers() # returns the extra tokenizer classes that are implemented, but not used in MatchFlow: AlphaNumeric, 5gram, Stripped3gram, Stripped5gram
```

**Example**: If you wanted to use all of the default and extra tokenizers, your script would include the following:

```python3
from MatchFlow import get_base_tokenizers, get_extra_tokenizers, create_featuress

default_tokenizers = get_base_tokenizers()
extra_tokenizers = get_extra_tokenizers()

all_tokenizers = default_tokenizers + extra_tokenizers

features = create_features(
   customers_df,
   prospects_df,
   ['name', 'address', 'revenue'],
   ['name', 'address', 'revenue'],
   tokenizers=all_tokenizers
)

```

### featurize()

```python
def featurize(
    features: List[Callable],           # Feature objects
    A: Union[pd.DataFrame, SparkDataFrame],  # Your first dataset
    B: Union[pd.DataFrame, SparkDataFrame],  # Your second dataset
    candidates: Union[pd.DataFrame, SparkDataFrame],  # Which pairs to compare
    output_col: str = 'feature_vectors',       # Name for feature vector column
    fill_na: float = 0.0                # Value for missing data
) -> Union[pd.DataFrame, SparkDataFrame]
```

This function converts each tuple pair in the candidates set into a feature vector, using the features created in create_features.

- `features` is a list of callable feature objects (typically from `create_features()`, but can be from another source.
- A and B are Pandas or Spark dataframes that store the two tables to be matched. Both dataframes must have an `_id` column.
- `candidates` is a Pandas or Spark dataframe that specifies a set of pairs of record IDs. This dataframe has two required columns:
  - `id2`: Record ID from table B (must appear in the `_id` column of dataframe B)
  - `id1_list`: Record IDs from table A (must appear in the `_id` column of dataframe A)

**Note: Sparkly/Delex candidate-set column names**

If your `candidates` DataFrame was produced by Sparkly or Delex, the column names may differ from the `featurize()` examples. In our reference blocking workflows, Sparkly/Delex typically output:

- `_id`: the record id from table **B**
- `ids`: the list of candidate record ids from table **A**

`featurize()` expects the same information, but with standardized column names:

- `id2`: record id from table **B**
- `id1_list`: list of candidate record ids from table **A**

**IMPORTANT:** If your columns are `_id` / `ids`, rename them to `id2` / `id1_list` before calling `featurize()`.

- `output_col` is the name of the column in the output dataframe that we will use to store feature vectors.
- `fill_na` is the value for missing data. We will use this value to fill in when similarity computation fails due to missing data. The default value is 0.0 (no similarity). Other common values are -1.0 (unknown), numpy.nan, or other float values. This is because missing data is common, and the system needs a consistent way to handle it.

**Understanding the Candidates DataFrame:** The following example explains the candidates dataframe.

```python
# Example candidates DataFrame (typically produced by a blocking solution)
candidates = pd.DataFrame({
    'id2': [1, 2],                     # Record IDs from dataframe B
    'id1_list': [[10, 11], [12, 13]],  # Record IDs from dataframe A (as lists)
    'source': ['blocking_system', 'manual_review'] # Possible metadata, will not impact featurization
})

# This means candidates consists of four pairs of record IDs: (1,10), (1, 11), (2, 12), (2, 13)
# where the first ID refers to a record in dataframe B and the second ID to a record in dataframe A
# We structure candidates this way to save space.
```

**Usage Example:**

```python
feature_vectors = featurize(
    features,              # Features from create_features() (or other program)
    customers_df,          # Your first dataset
    prospects_df,          # Your second dataset
    candidates_df,         # Pairs we want to compare
    output_col='feature_vectors' # Name of the column we want to save the vectors in
    # fill_na defaults to 0.0, no need to specify unless you want a different fill value
)

# Resulting dataFrame may look as follows:
result = pd.DataFrame({
    'id2': [1, 1, 2, 2],
    'id1': [10, 11, 12, 13],                # id1_list gets expanded to individual id1 entries
    'feature_vectors': [
        [0.9, 0.95, 0.98, 0.85, 0.92],  # Feature vector for A record 10 vs B record 1
        [0.3, 0.2, 0.1, 0.15, 0.25],    # Feature vector for A record 11 vs B record 1
        [0.8, 0.7, 0.9, 0.75, 0.88],    # Feature vector for A record 12 vs B record 2
        [0.1, 0.05, 0.02, 0.08, 0.12]   # Feature vector for A record 13 vs B record 2
    ],
    'source': ['blocking_system', 'blocking_system', 'manual_review', 'manual_review'],  # Preserved from candidates
    'score': [4.6, 1.0, 4.03, 0.37] # score generated by the call to featurize
    '_id': [0, 1, 2, 3]   # unique row identifiers generated by MatchFlow
})
```

We can visualize the resulting dataframe as the following table:
| id2 | id1 | feature_vectors | source | score | \_id |
|-----|-----|----------------|--------|-------|-----|
| 1 | 10 | [0.9, 0.95, 0.98, 0.85, 0.92] | blocking_system | 4.6 | 0 |
| 1 | 11 | [0.3, 0.2, 0.1, 0.15, 0.25] | blocking_system | 1.0 | 1 |
| 2 | 12 | [0.8, 0.7, 0.9, 0.75, 0.88] | manual_review | 4.03 | 2 |
| 2 | 13 | [0.1, 0.05, 0.02, 0.08, 0.12] | manual_review | 0.37 | 3 |

**The Importance of Columns 'score' and '\_id':** You can see that the output dataframe has two columns 'score' and '\_id'. Column 'score' contains a score for each feature vector, indicating how likely it is that this vector (that is, the record pair corresponding to this vector) is a match. The higher the score, the more likely that the vector is a match. Right now we compute this score by summing up the values of all features we know to be positively correlated with the likelihood of being a match. This score is important for various downstream functions, such as down_sample() and create_seeds(). For example, if we need to select a few vectors that are likely to be matches, then we will select vectors with high scores.

The output dataframe also has a column '\_id', which assigns an ID to each record pair. This column will be used by downstream functions, such as down_sample and others.

**Creating Features that Require Global Statistics:** Featurizing on a Spark cluster turns out to be non-trivial. For example, suppose we send a tuple pair (x,y) to a worker node and want to compute a TF/IDF feature. It turns out that this feature requires statistics that are global. So the worker node would need to "talk" to all other worker nodes to obtain all necessary numbers, to compute these "global" statistics. Our featurizing solution in featurize() avoids this problem.

### down_sample()

```python
def down_sample(
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Your feature vectors
    percent: float,                            # How much to keep (0.0 to 1.0)
    search_id_column: str,                     # Column with unique IDs
    score_column: str = 'score',               # Column with similarity scores
    bucket_size: int = 1000                    # Hash bucket size for representative sampling
) -> Union[pd.DataFrame, SparkDataFrame]
```

As discussed earlier, if the candidates set (the output of blocking) is large (e.g., having 50M+ examples), then performing certain operations, such as active learning, on it takes a long time. In such cases, we may want to perform these operations on a sample of the candidates set instead. This function returns such a sample. This is not a random sample because a random sample is likely to contain very few true matches, making it unsuitable for training a matcher.

- 'fvs' is a Pandas or Spark dataframe where each row contains a feature vector. This dataframe must contain the column named in score_column and the column named in search_id_column. We will use these two columns to sample (see below). Typically fvs is the dataframe output by the featurize() function.
- 'percent' is a number in [0,1] indicating the size of the sample as a fraction of the size of 'fvs'. For example 'percent = 0.1' means we want the sample's size to be 10% of the size of 'fvs'. The smaller the sample size, the faster downstream operations that use the sample will run, but we may also lose important patterns in 'fvs'.
- 'search_id_column' is the ID column in 'fvs'. Its values uniquely identify the rows of 'fvs'. Note that if you use featurize() to create 'fvs', then featurize() automatically adds an ID column called '\_id' to 'fvs'.
- 'score_column' is a column in 'fvs' that contains a numeric score for each row. This score must be such that the higher the score, the more likely that the row is a match (recall that each row of 'fvs' refers to a pair of records from A and B). If you use featurize() to create 'fvs', then featurize() automatically add such a column called 'score'.
- 'bucket_size' is a parameter used by down_sample, as we describe below.

**How It Works:** This function works as follows:

1. Scans through all rows in 'fvs' and assigns the rows into a set of buckets, using a hash function on the values of 'search_id_column'. It ensures that the size of each bucket never exceeds 'bucket_size'.
2. For each bucket of size _n_, the function first sorts the rows in that bucket in decreasing order of score in 'score_column', then takes the top _n x 'percent'_ rows in that order.
3. The function returns the union of all the rows taken from the buckets to be the desired sample.

Intuitively, the sample contains the top-scoring rows of each bucket. The top-scoring rows are likely to contain matches, and so the sample is likely to contain a reasonable number of matches. The above function may perform Steps 1-3 using Spark to save time.

**Example:** The following code returns 10% of 'feature_vectors' as the sample:

```python
sampled_data = down_sample(
    feature_vectors,
    percent=0.1,                    # Keep 10% of data
    search_id_column='pair_id',     # Your unique pair identifier
    score_column='similarity',      # Your similarity score column
    bucket_size=500                 # The number of records in each bucket
)
```

### create_seeds()

```python
def create_seeds(
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Your feature vectors
    nseeds: int,                               # How many examples to create
    labeler: Labeler,                          # A labeler object
    score_column: str = 'score'                # Column with similarity scores
) -> pd.DataFrame
```

This function selects 'nseeds' rows from a set of feature vectors 'fvs', then asks the user to label these rows as match/non-match, using a 'labeler' object. This function is typically used to provide a set of 'nseeds' labeled examples for the first iteration of active learning (which typically examines 'fvs' to find more rows to label, to create training data for the matcher).

- 'fvs' is a Pandas or Spark dataframe consisting of a set of feature vectors. This is typically the output of calling function featurize().
  - This dataframe must have the column named in score_column, which contains a score for each feature vector. The higher the score, the more likely that the vector is a match. This score column is typically in the dataframe output by function featurize(). This score column will be used to select seeds, as we will see.
  - Each row of the dataframe 'fvs' must also contain two columns named 'id1' and 'id2', which refer to the ID of a record in Table A and Table B, respectively. Thus, each row represents a record pair.
- 'nseeds' is the number of examples we will select for the user to label, to create seed examples. Usually a number between 20-50 is sufficient for starting the active learning process of creating more labeled examples.
- 'labeler' is an object of the Labeler class.
- 'score_column' is a column in 'fvs' that contains a score per feature vector, as discuss earlier.

**How It Works:** Roughly speaking, this function will sort the examples (that is, feature vectors) in 'fvs' in decreasing order of the score in 'score_column'. Next it selects a total of 'nseeds' examples, some of which come from the top of the sorted list and the rest from the bottom of the sorted list. The idea is to select some examples that are likely to be matches and some that are likely to be non-matches. Finally, the function will pass each example to the 'labeler' object, which asks the user to label the example as match/non-match.

Currently we provide three kinds of labeler: gold, command-line (CLI), and Web based. See [here](#built-in-labeler-classes) for detail.

- The gold labeler knows all the correct matches between Tables A and B. So given an example, that is, the IDs of two records, this labeler can consult the set of correct matches to see if the ID pair is in there. If yes, then it returns saying the two records match. Otherwise it returns non-match.
- The CLI labeler takes as input an example, that is, the IDs of two records. It looks up Tables A and B with these IDs to retrieve the two records, then shows these two records to the user, so that the user can label match, non-match, or unsure.
- The Web labeler is similar to the CLI labeler, but provides a Web-based labeling interface to the user (via a Web browser).

The function will terminate returning a set of 'nseeds' examples labeled as match or non-match. Note that if the user labels an example as 'unsure', then the function will ignore that example and select another replacement example. Note also that all 'nseeds' examples may be matches or non-matches. In other words, there is no guarantee that the output will contain both matches and non-matches.

**Example:** The following example uses a gold labeler:

```python
from MatchFlow import GoldLabeler

# Suppose you have a DataFrame of known matches:
gold_df = pd.DataFrame({
    'id1': [10, 12, 14],
    'id2': [1, 2, 3]
})

# Create a GoldLabeler instance as your labeler spec
labeler = GoldLabeler(gold_df)

# Create 100 seeds
initial_labeled_data = create_seeds(feature_vectors, nseeds=100, labeler=labeler)

# Typical output DataFrame from create_seeds:
print(initial_labeled_data.head())

# Example output:
#    id1  id2                features   score  label
# 0   10    1  [0.9, 0.95, 0.98]       0.95   1.0
# 1   11    1  [0.3, 0.2, 0.1]         0.20   0.0
# 2   12    2  [0.8, 0.7, 0.9]         0.80   1.0
# 3   13    2  [0.1, 0.05, 0.02]       0.05   0.0
# 4   14    3  [0.85, 0.8, 0.9]        0.85   1.0
```

### train_matcher()

```python
def train_matcher(
    model: MLModel,                       # What type of model to train
    labeled_data: Union[pd.DataFrame, SparkDataFrame],  # Your labeled examples
    feature_col: str = "feature_vectors", # Column name for your feature vectors
    label_col: str = "label"              # Column name for your labels
) -> MLModel
```

This function trains a matcher, that is, a classification model, on a set of labeled examples, each consisting of a feature vector and a label.

- 'model' is a pre-configured MLModel object. Common classification models being used are random forest, XGBoost, logistic regression, etc.
- 'labeled_data' is a Pandas or Spark dataframe. This dataframe must have two columns. The column named in 'feature_col' contains feature vectors, and the column named in 'label_col' contains the label (typically 0.0 and 1.0) for the corresponding vector.

**Examples of MLModel Objects:** In what follows we discuss several common MLModel object types:

1. **XGBoost** (recommended, often achieves high accuracy):

   ```python
   from MatchFlow import SKLearnModel
   from xgboost import XGBClassifier

    model = SKLearnModel(
        model=XGBClassifier,
        eval_metric='logloss', # How to evaluate the performance of the model (XGBoost model argument)
        objective='binary:logistic', # The use cases, used for optimization (XGBoost model argument)
        max_depth=6, # How deep each tree can go (XGBoost model argument)
        seed=42,    #  For reproducible results (XGBoost model argument)
        nan_fill=0.0
    )
   ```

2. **Random Forest** (good performance, fewer parameters to tune than XGBoost):

   ```python
   from MatchFlow import SKLearnModel
   from sklearn.ensemble import RandomForestClassifier

   model = SKLearnModel(
         model=RandomForestClassifier,
         n_estimators=100,        # Number of trees (RandomForestClassifier model argument)
         max_depth=10,            # How deep each tree can go (RandomForestClassifier model argument)
         random_state=42,         # For reproducible results (RandomForestClassifier model argument)
         nan_fill=0.0             # Important: Fill NaN values with 0.0
   )
   ```

3. **Logistic Regression** (simple and interpretable):

   ```python
   from MatchFlow import SKLearnModel
   from sklearn.linear_model import LogisticRegression

   model = SKLearnModel(
         model=LogisticRegression,
         C=1.0,                   # Regularization strength (LogisticRegression model argument)
         max_iter=1000            # Maximum training iterations (LogisticRegression model argument)
         nan_fill=0.0             # Important: Fill NaN values with 0.0
   )
   ```

4. **Spark ML Models** (for distributed training):

   ```python
   from MatchFlow import SparkMLModel
   from pyspark.ml.classification import RandomForestClassifier
   model = SparkMLModel(
         model=RandomForestClassifier,  # Spark ML RandomForestClassifier
         numTrees=100,                  # Number of Decision Trees (RandomForestClassifier model argument)
         maxDepth=10                    # How deep each tree can go (RandomForestClassifier model argument)
   )
   ```

5. **Additional Options** (for sklearn models):

   ```python
   from MatchFlow import SKLearnModel
   from sklearn.ensemble import RandomForestClassifier

   model = SKLearnModel(
         model=RandomForestClassifier,
         n_estimators=100,        # Number of trees (RandomForestClassifier model argument)
         max_depth=10,            # How deep each tree can go (RandomForestClassifier model argument)
         random_state=42,         # For reproducible results (RandomForestClassifier model argument)
         nan_fill=0.0             # **REQUIRED**: Fill NaN values with 0.0 for sklearn models
         use_floats=True          # Use float32 for memory efficiency (default: True)
   )
   ```

**Example of the DataFrame labeled_data:**

```python
# Example labeled_data DataFrame
labeled_data = pd.DataFrame({
    'id2': [1, 1, 2, 2, 3, 3],
    'id1': [10, 11, 12, 13, 14, 15],
    'feature_vectors': [
        [0.9, 0.95, 0.98],    # Feature Vector between id2 = 1 and id1 = 10
        [0.3, 0.2, 0.1],      # Feature Vector between id2 = 1 and id1 = 11
        [0.8, 0.7, 0.9],      # Feature Vector between id2 = 2 and id1 = 12
        [0.1, 0.05, 0.02],    # Feature Vector between id2 = 2 and id1 = 13
        [0.85, 0.8, 0.9],     # Feature Vector between id2 = 3 and id1 = 14
        [0.2, 0.15, 0.1]      # Feature Vector between id2 = 3 and id1 = 15
    ],
    'label': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # Labels: 1.0=match, 0.0=non-match
})
```

**Example Usage:**

```python
# Train a model
from sklearn.ensemble import RandomForestClassifier
model = SKLearnModel(
         model=RandomForestClassifier,
         n_estimators=100,        # Number of trees (RandomForestClassifier model argument)
         max_depth=10,            # How deep each tree can go (RandomForestClassifier model argument)
         random_state=42,         # For reproducible results (RandomForestClassifier model argument)
         nan_fill=0.0             # Important: Fill NaN values with 0.0
)

trained_matcher = train_matcher(
    model=model,
    labeled_data=labeled_examples,
    feature_col='feature_vectors',
    label_col='label'
)

# The model can learn patterns such as the following:
# "When name similarity > 0.8 AND address similarity > 0.7 → likely match"
# "When all similarities < 0.3 → likely non-match"
# "When name similarity > 0.9 but address similarity < 0.2 → uncertain"
```

### apply_matcher()

```python
def apply_matcher(
    model: MLModel,                          # Your trained model object
    df: Union[pd.DataFrame, SparkDataFrame], # Data to make predictions on
    feature_col: str,                        # Column with feature vectors (the input to the model)
    prediction_col: str,                     # The column that will have the predictions
    confidence_col: Optional[str] = None,    # The column that will have the confidence scores; if omitted, confidence scores will not be included in the result dataframe
) -> Union[pd.DataFrame, SparkDataFrame]
```

This function applies a trained matcher to new examples to predict match/non-match.

- 'model' is a trained MLModel object. It is typically the output of train_matcher(), and is a trained SKLearn model or a SparkML Transformer.
- 'df' is a Pandas or Spark dataframe of examples (feature vectors). This dataframe must contain the column named in 'feature_col', referring to a feature vector.
- 'prediction_col' will be a new column added to 'df'. The function will store the prediction match or non-match (usually 1.0 or 0.0) in this column.
- 'confidence_col' will be a new column added to 'df'. The function will store a confidence score in the range [0.5, 1.0] in this column. If omitted, the function will not store any confidence score in the output dataframe.

For example, the output dataframe may look as follows (without a confidence score column):

```python
# Your input data with predictions added
result = pd.DataFrame({
    'id1': [20, 21, 22, 23],
    'id2': [3, 3, 4, 4],
    'feature_vectors': [
        [0.85, 0.9, 0.92],    # High scores for some similarity functions
        [0.2, 0.15, 0.1],     # Low scores for some similarity functions
        [0.75, 0.8, 0.85],    # Medium-high scores for some similarity functions
        [0.4, 0.45, 0.5]      # Medium scores for some similarity functions
    ],
    'predictions': [1.0, 0.0, 1.0, 0.0]       # Model predictions
})
```

**Example Usage:**

```python
# Apply your trained model to new data
predictions = apply_matcher(
    model,                              # Your trained model
    new_feature_vectors,                # New data to predict on
    feature_col='feature_vectors',      # Column with features
    prediction_col='match_prediction'   # Where to put results predictions
    confidence_col='match_conf'         # Where to put results confidence scores
)

# Now you can see which pairs the model thinks are matches
matches = predictions[predictions['match_prediction'] == 1.0]
print(f"Found {len(matches)} potential matches")
```

### label_data()

```python
def label_data(
    model: MLModel,        # Untrained model object
    mode: Literal["batch", "continuous"], # How to do the labeling
    labeler: Labeler,                # Labeler object
    fvs: Union[pd.DataFrame, SparkDataFrame],  # Unlabeled data
    seeds: Optional[Union[pd.DataFrame, SparkDataFrame]] = None  # Existing labeled data
    parquet_file_path: str = 'active-matcher-training-data.parquet' # Path to file of where to save labeled data
    batch_size: Optional[int] = 10 # only for use with "batch mode", specifies how many examples to label in each batch
    max_iter: Optional[int] = 50 # only for use with "batch mode", specifies how many iterations of active learning to complete
    queue_size: Optional[int] = 50 # only for use with "continuous mode", specifies how many labeled examples must be in queue to wait for the user to label more
    max_labeled: Optional[int] = 100000 # only for use with "continuous mode", specifies how many examples to label before terminating active learning
    on_demand_stop: Optional[bool] = True # only for use with "continuous mode", if set to True, ignores max_labeled and waits for a stop label from the user. If using gold data, this must be set to False
) -> Union[pd.DataFrame, SparkDataFrame]
```

This function implements an active learning process, which selects informative examples for a user to label, to create a set of labeled examples for later purposes, such as training a matcher. The active learning process can be batch or continuous, as we discuss at the start of this guide.

To understand this function, you should carefully read Section "Preliminaries" at the start of this guide.

In what follows we explain the parameters of this function:

- 'model' is an MLModel object, which represents a matcher M.
- 'mode' indicates how we want to do active learning: in the batch mode or the continuous mode.
- 'labeler' is a Labeler object that the user will use to label examples (returned by the matcher M).
- 'fvs' is a dataframe that consists of a set of examples from which we will select examples for the user to label.
- 'seeds' is a small set of labeled examples we use to start the active learning process.
- 'parquet_file_path' is the path to the file where we will store all examples that the user has labeled.
- 'batch_size' and 'max_iter' are optional parameters used by active learning in the batch mode (explained below).
- 'queue_size', 'max_labeled', and 'on_demand_stop' are parameters used by active learning in the continuous mode (explained below).

**How the Batch Mode Works:** Consider active learning (AL) in the batch mode. It works as follows:

1. Use 'seeds' to train the matcher M represented by 'model'. Note: 'seeds' must contain both matches and non-matches. Otherwise the function will terminate, reporting error.
2. Iterate until the number of iterations reach 'max_iter':
   - Apply matcher M to all examples in 'fvs', then select b = 'batch_size' examples that are unlabeled and appear most "informative" (in the sense that if they are labeled, then matcher M can learn the most from these examples).
   - Ask the user to label these b examples using 'labeler'.
   - Retrain matcher M using all examples that have been labeled so far (including also the seed examples).
3. Return the trained matcher M, and save all labeled examples into the file indicated in 'parquet_file_path'.

Here is an example for the batch mode:

```python
def label_data(
    model = model,
    mode = "batch",
    labeler = Labeler,
    fvs = fvs,
    seeds = seeds,
    parquet_file_path = 'active-matcher-training-data.parquet'
    batch_size = 15    # label 15 examples per iteration
    max_iter = 100    # complete 100 iterations of active learning; results in 100*15 = 1500 labeled examples
) -> Union[pd.DataFrame, SparkDataFrame]
```

**How the Continuous Mode Works:** Consider AL in the continuous mode. Here we no longer have the notion of iterations. Instead, the user will continously label examples, as discussed in Section "Preliminaries" at the start of this guide.

- In this mode, 'queue_size' is the size of the queue where we store the examples that the user will label. When the user indicates that he or she is ready to label, we take the example at the front of this queue and give it to the user to label. As soon as the queue becomes somewhat empty (say less than 2/3 full), we find more informative examples to add to the queue.
- 'max_labeled' is the maximal number of examples that the user will label. AL will terminates after this.
- 'on_demand_stop': If this is True, then ignore 'max_labeled', let the user keep labeling, and stop only when the user hits a button on the labeler's UI indicating that the user wants to stop the labeling process.

So if you use the gold labeler, then 'on_demand_stop' should be False. Otherwise,

- you can either set a value for 'max_labeled' and set 'on_demand_stop' to False, in order to label just a fixed number of examples.
- Or you can set 'on_demand_stop' to True, in order to label as many examples as you want.

Here is an example for the continuous mode:

```python
def label_data(
    model = model,
    mode = "continuous",
    labeler = Labeler,
    fvs = fvs,
    seeds = seeds,
    parquet_file_path = 'active-matcher-training-data.parquet'
    queue_size = 100    # allow the queue to have 100 unlabeled examples
    max_labeled = 1500    # continue the active learning process until 1500 examples are labeled
    on_demand_stop = False   # since we want to use the max_labeled value, we will set on_demand_stop to False
) -> Union[pd.DataFrame, SparkDataFrame]
```

**What This Function Returns:** This function returns a dataframe, which has the same columns as the 'fvs' dataframe that was passed in as an input, plus the columns 'label' and 'labeled_in_iteration'. 'label' contains 1.0 for matches, 0.0 for non-matches.

'labeled_in_iteration' is which model number this labeled data came from. So, if it was labeled in batch 5, 'labeled_in_iteration' is 5. If 5 models had been trained in the continuous mode, and the new labeled examples came from this lastest model (model 5), then 'labeled_in_iteration' would be 5.

#### label_pairs()

```python
def label_pairs(
    labeler: Labeler,                # Labeler object
    pairs: Union[pd.DataFrame, SparkDataFrame] # DataFrame with pairs of id's
) -> Union[pd.DataFrame, SparkDataFrame]
```

This function takes a set of examples (each is a pair of record IDs), then asks the user to label these examples as match/non-match, using a labeler.

- 'labeler' is a Labeler object. See [Built-in Labeler Classes](#built-in-labeler-classes) for available options and usage. Currently, label_pairs supports all of the built-in labeler types, except for the gold labeler.
- 'pairs' is a Pandas or Spark dataframe that must have at least two columns: a column 'x' that refers to the record ID from Table A and a column 'y' that refers to the record ID from Table B (these two columns can be named anything). Note that when you create the labeler object, you set the field a_df to point to a dataframe storing Table A and set the field b_df to point to a dataframe storing Table B.

This function returns a Pandas or Spark dataframe with the columns 'x', 'y' (described above), 'label'. For example, if the `pairs` DataFrame had the columns 'x' being `id1` and 'y' being `id2`, then the returned DataFrame would have the columns `id1`, `id2`, `label`. The return type (Pandas or Spark DataFrame) will match the input type of `pairs`.

### Saving and Loading Functions

MatchFlow provides functions to help you save/load features (e.g., the output of create_features()) and dataframes (e.g., the output of featurize()). Saving/loading features and dataframes is very important when you work in long EM sessions, when you need to take a break, or when it takes a long time to produce a dataframe, such as the set of feature vectors output by featurize().

#### save_features(features, path)

```python
def save_features(
    features: List[Callable]   # List of feature objects to save
    path: str                  # Path where to save the features file
) -> None
```

This function saves a list of feature objects to disk using pickle serialization.

**Usage Examples:** If you use Pandas or Spark on a local machine to save features:

```python
from MatchFlow import save_features

features = create_features(
    customers_df, prospects_df,
    ['name', 'address', 'revenue'],
    ['name', 'address', 'revenue']
)

save_features(features=features, path='./features.pkl')
```

This will create a file called 'features.pkl' in the directory where your Python script lives on your local machine.

If you use Spark on a cluster to save features:

```python
from MatchFlow import save_features
from pathlib import Path

features = create_features(
    customers_df, prospects_df,
    ['name', 'address', 'revenue'],
    ['name', 'address', 'revenue']
)

save_features(features=features, path=str(Path(__file__).parent / 'features.pkl'))
```

This will create a file called 'features.pkl' in the directory where your Python script lives on your master node.

#### load_features(path)

```python
def load_features(
    path: str                  # Path to the saved features file
) -> List[Callable]
```

This function loads a list of feature objects from disk using pickle deserialization.

**Usage Example:** If you use Pandas or Spark on a local machine to load the features:

```python
from MatchFlow import load_features

features = load_features(path='./features.pkl')
```

This will load in the features list from the 'features.pkl' file in the directory where your Python script lives on your local machine.

If you use Spark on a cluster to load the features:

```python
from MatchFlow import load_features
from pathlib import Path

features = load_features(path=str(Path(__file__).parent / 'features.pkl'))
```

This will load in the features list from the 'features.pkl' file in the directory where your Python script lives on your master node.

#### save_dataframe(dataframe, path)

```python
def save_dataframe(
    dataframe: Union[pd.DataFrame, pyspark.sql.DataFrame]   # DataFrame to save
    path: str                                               # Path where to save the DataFrame
) -> None
```

Save a dataframe to disk as a parquet file. Automatically detecting if it is a Pandas or Spark dataframe.

**Usage Examples:** If you save a Pandas dataframe on a local machine, or if you run Spark on a local machine and save a Spark dataframe:

```python
from MatchFlow import save_dataframe

feature_vectors_df = featurize(
    features,
    customers_df,
    prospects_df,
    candidates_df,
    output_col='feature_vectors'
)

save_dataframe(dataframe=feature_vectors_df, path='./feature_vectors_df.parquet')
```

This will create a file called 'feature_vectors_df.parquet' in the directory where your Python script lives on your local machine.

If you use Spark on a cluster and save a Spark dataframe:

```python
from MatchFlow import save_dataframe
from pathlib import Path

feature_vectors_df = featurize(
    features,
    customers_df,
    prospects_df,
    candidates_df,
    output_col='feature_vectors'
)

save_dataframe(dataframe=feature_vectors_df,  path=str(Path(__file__).parent / 'feature_vectors_df.parquet'))
```

This will create a file called 'feature_vectors_df.parquet' in the directory where your Python script lives on your master node.

#### load_dataframe(path, df_type)

```python
def load_dataframe(
    path: str                  # Path to the saved DataFrame parquet file
    df_type: str               # Type of DataFrame to load ('pandas' or 'sparkdf')
) -> Union[pd.DataFrame, pyspark.sql.DataFrame]
```

This function loads a dataframe from disk.

**Usage Examples:** If you use Pandas on a local machine and want to load a dataframe:

```python
from MatchFlow import load_dataframe

feature_vectors_df = load_dataframe(path='./feature_vectors_df.parquet', df_type='pandas')
```

This will load in the feature vectors dataframe from the 'feature_vectors_df.parquet' file in the directory where your Python script lives on your local machine.

If you use Spark on a local machine and want to load a dataframe:

```python
from MatchFlow import load_dataframe

feature_vectors_df = load_dataframe(path='./feature_vectors_df.parquet', df_type='sparkdf')
```

This will load in the feature vectors dataframe from the 'feature_vectors_df.parquet' file in the directory where your Python script lives on your local machine.

Finally, if you use Spark on a cluster and want to load a dataframe:

```python
from MatchFlow import load_dataframe

feature_vectors_df = load_dataframe(path=str(Path(__file__).parent / 'feature_vectors_df.parquet'), df_type='sparkdf')
```

This will load in the feature vectors dataframe from the 'feature_vectors_df.parquet' file in the directory where your Python script lives on your master node.

### Built-in Labeler Classes

MatchFlow provides several built-in labeler classes. You can use these directly or extend them. All labelers inherit from the public `Labeler` abstract class.

#### GoldLabeler

Given a pair of record IDs, this labeler consults the set of gold matches. If the pair of IDs exist in the set, then the labeler returns 1.0 (match), otherwise it returns 0.0 (non-match). This labeler can be used during the development and debugging process, or to compute the matching accuracy.

This labeler can be used in all three settings: running Python on a single machine, running Spark on a single machine, or running Spark on a cluster.

To create a labeler of this type, use `GoldLabeler(gold)`, where 'gold' is a DataFrame containing all gold matches with two columns: `id1` refers to the record IDs from Table A, and `id2` refer to the record IDs from Table B that match the corresponding `id1` records.

**Usage Example**:

```python
# Create gold standard data
gold_matches = pd.DataFrame({
    'id1': [1, 3, 5],      # Records from dataset A
    'id2': [101, 103, 105] # Matching records from dataset B
})

# Create labeler
gold_labeler = GoldLabeler(gold_matches)

# Use in active learning
labeled_data = label_data(
    model=model,
    mode='batch',
    labeler=gold_labeler,
    fvs=feature_vectors
)
```

#### CLILabeler

Given a pair of record IDs, this labeler retrieves the corresponding tuples, displays them to the command-line interface, asks the user if the tuples match. The labeler returns 1.0, 0.0, 2.0, and -1.0 if the user clicks the "yes" button, "no" button, "unsure" button, and "stop" button, respectively.

This labeler can be used for running Python or Spark on a single machine. It cannot be used for running Spark on a cluster (for that setting, you need to use the Web labeler, discussed below).

To create a labeler object of this type, use `CLILabeler(a_df, b_df, id_col='_id')`. Here 'a_df' and 'b_df' are two Pandas or Spark dataframes that store the tuples of Tables A and B, respetively. Both dataframes must have ID columns with the name specified in 'id_col' (the default is '\_id').

**Usage Example:**

```python
# Create CLI labeler with custom ID column
cli_labeler = CLILabeler(
    a_df=customers_df,
    b_df=prospects_df,
    id_col='customer_id'  # Custom ID column name
)

# Use in seed creation
seeds = create_seeds(
    fvs=feature_vectors,
    nseeds=50,
    labeler=cli_labeler
)
```

#### WebUILabeler

As discussed at the start of this guide, the Web-baser labeler runs a browser and a Web server. When a MatchFlow function wants you to label a pair of tuples (x,y), it sends this pair to the Web server, which in turn sends it to the browser, where you can label the pair as match, non-match, or unsure. Currently we use a Flask-based Web server and a Streamlit interface to implement this labeler.

This labeler can be used in all three modes: Python on a single machine, Spark on a single machine, and Spark on a cluster. If you run on a local machine, then the Web server will run locally on that machine. If you run Spark on a cluster, then the Web server is likely to run on the master node (assuming that you submit the Python script on this node).

To create a labeler object of this type, use `WebUILabeler(a_df, b_df, id_col='_id', flask_port=5005, streamlit_port=8501, flask_host='127.0.0.1')`:

- `a_df` (DataFrame): a Pandas or Spark dataframe storing the tuples of Table A.
- `b_df` (DataFrame): a Pandas or Spark dataframe storing the tuples of Table B.
- `id_col` (str, default='\_id'): Both a_df and b_df must have a column with this name, and this column refers to the tuple IDs.
- `flask_port` (int, default=5005): Port for the Flask backend API.
  - When to change: If port 5005 is already in use by another application
  - Example: Use `flask_port=5006` if 5005 is occupied
- `streamlit_port` (int, default=8501): Port for the Streamlit frontend
  - When to change: If port 8501 is already in use by another application
  - Example: Use `streamlit_port=8502` if 8501 is occupied
- `flask_host` (str, default='127.0.0.1'): Host address for the Flask server
  - When to change: If you need to access the interface from other machines on the network
  - Example: Use `flask_host='0.0.0.0'` to allow external access

**Example for Running Python on a Single Machine:**

```python
# Create web labeler with custom configuration
web_labeler = WebUILabeler(
    a_df=customers_df,
    b_df=prospects_df,
    id_col='customer_id',
    flask_port=5006,      # Custom port if 5005 is busy
    streamlit_port=8502,  # Custom port if 8501 is busy
    flask_host='0.0.0.0'  # Allow other processes to hit the endpoints (desired behavior)
)

# Use in active learning
labeled_data = label_data(
    model=model,
    mode='continuous',
    labeler=web_labeler,
    fvs=feature_vectors,
    parquet_file_path='./web-labeling-data.parquet'
)
```

To access the WebUI labeler, you will visit: 127.0.0.1:8502 on your local machine. We are also saving the labeled data to 'web-labeling-data.parquet'. This file will be saved in the directory where your Python script lives on your local machine.

**Example for Running Spark on a Single Machine:**

```python
# Create web labeler with custom configuration
web_labeler = WebUILabeler(
    a_df=customers_df,
    b_df=prospects_df,
    id_col='customer_id',
    flask_port=5006,      # Custom port if 5005 is busy
    streamlit_port=8502,  # Custom port if 8501 is busy
    flask_host='0.0.0.0'  # Allow other processes to hit the endpoints (desired behavior)
)

# Use in active learning
labeled_data = label_data(
    model=model,
    mode='continuous',
    labeler=web_labeler,
    fvs=feature_vectors,
    parquet_file_path='web-labeling-data.parquet'
)
```

To access the WebUI labeler, you will visit: 127.0.0.1:8502 on your local machine. We are also saving the labeled data to 'web-labeling-data.parquet'. This file will be saved in the directory where your Python script lives on your local machine.

**Example for Running Spark on a Cluster:**

```python
from pathlib import Path
# Create web labeler with custom configuration
web_labeler = WebUILabeler(
    a_df=customers_df,
    b_df=prospects_df,
    id_col='customer_id',
    flask_port=5006,      # Custom port if 5005 is busy
    streamlit_port=8502,  # Custom port if 8501 is busy
    flask_host='0.0.0.0'  # Allow other processes to hit the endpoints (desired behavior)
)

parquet_file_path = Path(__file__).resolve()
# Use in active learning
labeled_data = label_data(
    model=model,
    mode='continuous',
    labeler=web_labeler,
    fvs=feature_vectors,
    parquet_file_path=str(parquet_file_path / 'web-labeling-data.parquet')
)
```

To access the WebUI labeler, you will visit: {public ip address of your master node}:8502 from your local machine. We are also saving the labeled data to 'web-labeling-data.parquet'. This file will be saved in the directory where your Python script lives on your master node.

#### CustomLabeler

You can implement your own labeler by subclassing `CustomLabeler` and implementing the `label_pair(row1, row2)` method. This method must return 1.0, 0.0, 2.0, and -1.0 for the cases where the user clicks the button match, non-match, unsure, and stop, respectively.

The constructor should use a_df, b_df, and id_col, as described for the built-in labelers.

**Example:**

```python
from MatchFlow import CustomLabeler

class EmailDomainLabeler(CustomLabeler):
    """Labeler that matches records based on email domain and name similarity"""

    def label_pair(self, row1, row2):
        # Extract email domains
        email1 = row1.get('email', '')
        email2 = row2.get('email', '')

        if email1 and email2:
            domain1 = email1.split('@')[-1].lower()
            domain2 = email2.split('@')[-1].lower()

            # Check if same domain
            if domain1 == domain2:
                # Check name similarity
                name1 = row1.get('name', '').lower()
                name2 = row2.get('name', '').lower()

                # Simple word overlap
                words1 = set(name1.split())
                words2 = set(name2.split())
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)

                return 1.0 if overlap >= 0.5 else 0.0

        return 0.0

# Create custom labeler
custom_labeler = EmailDomainLabeler(customers_df, prospects_df)

# Use in seed creation
seeds = create_seeds(
    fvs=feature_vectors,
    nseeds=100,
    labeler=custom_labeler
)
```

### Training Data Persistence

MatchFlow automatically saves and loads training data during active learning and seed creation processes to ensure that you don't lose your labeling progress. This is especially important for long labeling sessions or when working with large datasets.

#### Automatic Training Data Saving

When using `label_data()` or `create_seeds()`, MatchFlow automatically:

1. **Saves Progress**: The system saves the training data after a batch (for batch active learning), or after an example is labeled (for continuous active learning) to a Parquet file (default: `active-matcher-training-data.parquet`).

2. **Resumes from Previous Session**: If you restart the labeling process, MatchFlow automatically detects and loads previously labeled data.

3. **Incremental Updates**: New labeled pairs are appended to existing training data without overwriting previous work.

#### Configuration Options

You can customize where the training data is saved:

```python
# Custom file path for training data
label_data(
    model=model_object,
    mode='batch',
    labeler=labeler_object,
    fvs=feature_vectors,
    parquet_file_path='my-custom-training-data.parquet'  # Custom path
)
```

#### File Format

Training data is saved in Parquet format with the following schema:

- `_id`: Unique identifier for each record pair
- `id1`: Record ID from Table A
- `id2`: Record ID from Table B
- `features`: Feature vector (array of floats)
- `label`: Ground truth label (1.0 for match, 0.0 for non-match)

### Custom Feature Development

If the built-in features are not adequate for your use case, you can create your own feature, as illustrated by the following example:

```python
class PhoneNumberFeature(Featurizer):
    def __init__(self, a_column, b_column):
        self.a_column = a_column  # Column name in dataset A
        self.b_column = b_column  # Column name in dataset B

    def __str__(self):
        # Must return unique identifier for this feature
        return f"phone_similarity({self.a_column}, {self.b_column})"

    def __call__(self, record_b, records_a):
        # record_b: one record from dataset B
        # records_a: list of records from dataset A to compare against

        phone_b = self.normalize_phone(record_b[self.b_column])
        similarities = []

        for record_a in records_a:
            phone_a = self.normalize_phone(record_a[self.a_column])
            similarity = self.compare_phones(phone_a, phone_b)
            similarities.append(similarity)

        return similarities

    def normalize_phone(self, phone):
        # Remove formatting: "(555) 123-4567" → "5551234567"
        if phone is None or pd.isna(phone):
            return ""
        return ''.join(c for c in str(phone) if c.isdigit())

    def compare_phones(self, phone1, phone2):
        if phone1 == phone2:
            return 1.0  # Exact match
        elif len(phone1) >= 7 and len(phone2) >= 7 and phone1[-7:] == phone2[-7:]:
            return 0.8  # Same last 7 digits (same local number, different area code)
        elif len(phone1) >= 4 and len(phone2) >= 4 and phone1[-4:] == phone2[-4:]:
            return 0.3  # Same last 4 digits (might be related)
        else:
            return 0.0  # Different numbers
```

### Best Practices

We end the guide by sharing some simple tips for starting a new mathing project:

1. **Understand Your Data**

   ```python
   # Always start by exploring your data
   print("Dataset A shape:", df_a.shape)
   print("Dataset B shape:", df_b.shape)
   print("Dataset A columns:", df_a.columns.tolist())
   print("Dataset B columns:", df_b.columns.tolist())

   # Check data quality
   print("Missing values in A:", df_a.isnull().sum())
   print("Missing values in B:", df_b.isnull().sum())

   # Ensure _id columns exist and are unique
   print("Dataset A _id unique:", df_a['_id'].nunique() == len(df_a))
   print("Dataset B _id unique:", df_b['_id'].nunique() == len(df_b))
   ```

2. **Start Small**

   ```python
   # Use a small sample first to test your pipeline
   sample_a = df_a.sample(1000)
   sample_b = df_b.sample(1000)

   # Develop and test with samples
   features = create_features(sample_a, sample_b, ...)
   ```

3. **Validate Each Step**

   ```python
   # Check your feature creation
   features = create_features(...)
   print(f"Created {len(features)} features")

   # Check your input candidates (from external blocking system)
   print(f"Received {len(candidates)} candidate pairs to analyze")
   print(f"Candidates cover {candidates['id2'].nunique()} records from dataset B")

   # Check your feature vectors
   feature_vectors = featurize(...)
   print(f"Feature vector shape: {feature_vectors['feature_vectors'].iloc[0].shape}")
   ```
