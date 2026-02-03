from typing import List, Optional, Callable, Any, Union
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
import pickle
from .storage import MemmapDataFrame
from threading import Lock
from .utils import get_logger, repartition_df
from time import time
from .tokenizer import (
    AlphaNumericTokenizer,
    NumericTokenizer,
    QGramTokenizer,
    StrippedQGramTokenizer,
    WhiteSpaceTokenizer,
    StrippedWhiteSpaceTokenizer,
    ShingleTokenizer,
)

from .feature import (
    ExactMatchFeature,
    EditDistanceFeature,
    SmithWatermanFeature,
    NeedlemanWunschFeature,
    RelDiffFeature,
    JaccardFeature, 
    OverlapCoeffFeature, 
    CosineFeature,
    MongeElkanFeature,
    TFIDFFeature, 
    SIFFeature
)


TOKENIZERS = [
        StrippedWhiteSpaceTokenizer(),
        NumericTokenizer(),
        QGramTokenizer(3),
]

EXTRA_TOKENIZERS = [
        AlphaNumericTokenizer(),
        QGramTokenizer(5),
        StrippedQGramTokenizer(3),
        StrippedQGramTokenizer(5),
]

SIM_FUNCTIONS = [
    TFIDFFeature,
    JaccardFeature, 
    SIFFeature,
    OverlapCoeffFeature, 
    CosineFeature,
]

log = get_logger(__name__)

class BuildCache:
    def __init__(self):
        self._cache = []
        self._lock = Lock()

    def add_or_get(self, builder):
        with self._lock:
            try:
                builder = self._cache[self._cache.index(builder)]
            except ValueError:
                self._cache.append(builder)

        return builder

    def clear(self):
        with self._lock:
            self._cache.clear()


def get_base_sim_functions():
    """
    get the base similarity functions

    Returns
    -------
    list
        a list of similarity functions, currently includes:
        - TFIDFFeature
        - JaccardFeature
        - SIFFeature
        - OverlapCoeffFeature
        - CosineFeature
    """
    return SIM_FUNCTIONS


def get_base_tokenizers():
    """
    get the base tokenizers

    Returns
    -------
    list
        a list of tokenizers, currently includes:
        - StrippedWhiteSpaceTokenizer
        - NumericTokenizer
        - QGramTokenizer(3)
    """
    return TOKENIZERS


def get_extra_tokenizers():
    """
    get the extra tokenizers

    Returns
    -------
    list
        a list of extratokenizers, currently includes:
        - AlphaNumericTokenizer
        - QGramTokenizer(5)
        - StrippedQGramTokenizer(3)
        - StrippedQGramTokenizer(5)
    """
    return EXTRA_TOKENIZERS


def _tokenize_and_count(df_itr, token_col_map):

    for df in df_itr:
        yield pd.DataFrame({
                col : df[t[1]].apply(t[0].tokenize) for col, t in token_col_map.items()
                }).map(lambda x : len(x) if x is not None else None)


def _drop_nulls(df, threshold):
    numeric_cols = {f.name for f in df.schema
                    if isinstance(f.dataType, (T.DoubleType, T.FloatType))}
    checks = [
        (
            (df[c].isNull() | F.isnan(df[c]))
            if c in numeric_cols
            else df[c].isNull()
        )
        .cast('int').alias(c)
        for c in df.columns
    ]

    null_percent = (
        df
        .select(*checks)
        .agg(*[F.mean(c).alias(c) for c in df.columns])
        .toPandas()
        .iloc[0]
    )

    cols = null_percent.index[null_percent.lt(threshold)]
    return df.select(*cols)


def create_features(
    A: Union[pd.DataFrame, SparkDataFrame],
    B: Union[pd.DataFrame, SparkDataFrame],
    a_cols: List[str],
    b_cols: List[str],
    sim_functions: Optional[List[Callable[..., Any]]] = None,
    tokenizers: Optional[List[Callable[..., Any]]] = None,
    null_threshold: float = .5
) -> List[Callable]:
    """
    creates the features which will be used to featurize your tuple pairs

    Parameters
    ----------

    A : Union[pd.DataFrame, SparkDataFrame]
        the records of table A
    B : Union[pd.DataFrame, SparkDataFrame]
        the records of table B
    a_cols : list
        The names of the columns for DataFrame A that should have features generated
    b_cols : list
        The names of the columns for DataFrame B that should have features generated
    sim_functions : list of callables, optional
        similarity functions to apply (default: None)
    tokenizers : list of callables, optional
        tokenizers to use (default: None)
    null_threshold : float
        the portion of values that must be null in order for the column to be dropped and
        not considered for feature generation

    Returns
    -------
    List[Callable]
        a list containing initialized feature objects for columns in A, B
    """
    if sim_functions is None:
        sim_functions = SIM_FUNCTIONS
    if tokenizers is None:
        tokenizers = TOKENIZERS
    if isinstance(A, SparkDataFrame):
        # only keep a_cols and b_cols (if B is not None)
        df = A.select(a_cols)
        if B is not None:
            df = df.unionAll(B.select(b_cols))
        
        df = repartition_df(df, 5000, [])
        df = _drop_nulls(df, null_threshold)
        cols_to_keep = df.columns

        numeric_cols = [c.name for c in df.schema if c.dataType in {T.IntegerType(), T.LongType(), T.FloatType(), T.DoubleType()}]
        # cast everything to a string
        df = df.select(*[F.col(c).cast('string') for c in df.columns])

        token_cols = {}
        for t in tokenizers:
            for c in df.columns:
                cname = t.out_col_name(c)
                token_cols[cname] = (t, c)

        schema = T.StructType([T.StructField(c, T.IntegerType()) for c in token_cols])
        df = df.mapInPandas(lambda x : _tokenize_and_count(x, token_cols), schema=schema)
        #record_count = df.count()

        avg_counts = df.agg(*[F.mean(c).alias(c) for c in token_cols])\
                        .toPandas().iloc[0]
    elif isinstance(A, pd.DataFrame):
        # only keep a_cols and b_cols (if B is not None)
        df = A[a_cols]
        if B is not None:
            df = pd.concat([df, B[b_cols]])

        # drop null columns
        null_frac = df.isnull().mean()
        cols_to_keep = null_frac[null_frac < null_threshold].index.tolist()
        df = df[cols_to_keep]

        # find numeric columns, and then cast everything to a string
        numeric_cols = df.select_dtypes(include=[np.integer, np.floating]).columns.tolist()
        df = df.astype(str)

        # create token columns map
        token_cols = {}
        for t in tokenizers:
            for c in df.columns:
                cname = t.out_col_name(c)
                token_cols[cname] = (t, c)

        # get the average number of tokens for each tokenizer, column
        results = {}
        for new_col, (tokenizer, orig_col) in token_cols.items():
            tokens = df[orig_col].apply(lambda x: tokenizer.tokenize(x) if pd.notnull(x) else None)
            counts = tokens.apply(lambda x: len(x) if x is not None else np.nan)
            results[new_col] = counts
        counts_df = pd.DataFrame(results)
        avg_counts = counts_df.mean()

    # add features to features list
    features = []
    for c in cols_to_keep:
        features.append(ExactMatchFeature(c, c))

    for c in numeric_cols:
        features.append(RelDiffFeature(c, c))

    for token_col_name, p in token_cols.items():
        tokenizer, column_name = p
        avg_count = avg_counts[token_col_name]

        if avg_count >= 3:
            features += [f(column_name, column_name, tokenizer=tokenizer) for f in sim_functions]

        if str(tokenizer) == AlphaNumericTokenizer.NAME:
            if avg_count <= 10:
                features.append(MongeElkanFeature(column_name, column_name, tokenizer=tokenizer))
                features.append(EditDistanceFeature(column_name, column_name))
                features.append(SmithWatermanFeature(column_name, column_name))
    return features


def featurize(
    features: List[Callable],
    A,
    B,
    candidates,
    output_col: str = 'feature_vectors',
    fill_na: float = 0.0,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    applies the featurizer to the record pairs in candidates

    Parameters
    ----------
    features : List[Callable]
        a list containing initialized feature objects for columns in A, B
    A : Union[pd.DataFrame, SparkDataFrame]
        the records of table A
    B : Union[pd.DataFrame, SparkDataFrame]
        the records of table B
    candidates : Union[pd.DataFrame, SparkDataFrame]
        blocked candidates with required columns:
        - `id2`: id from table B
        - `id1_list`: list of candidate ids from table A.
        If your candidates were produced by Sparkly/Delex, you must rename columns
        to this format before calling `featurize()`:
        - rename the column that contains the table B id -> `id2`
        - rename `ids` -> `id1_list`
    output_col : str
        the name of the column for the resulting feature vectors, default `feature_vectors`
    fill_na : float
        value to fill in for missing data, default 0.0
    Returns
    -------
    Union[pd.DataFrame, SparkDataFrame]
        DataFrame with feature vectors created with the following schema:
        (`id2`, `id1`, `output_col`, other columns from candidates).
        Returns pandas DataFrame if inputs A and B are pandas DataFrames,
        otherwise returns Spark DataFrame.
    """
    return_pandas = False
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return_pandas = isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame)
    if isinstance(A, pd.DataFrame):
        A = spark.createDataFrame(A)
    if isinstance(B, pd.DataFrame):
        B = spark.createDataFrame(B)
    if isinstance(candidates, pd.DataFrame):
        candidates = spark.createDataFrame(candidates)
    table_a_preproc, table_b_preproc = _build(A, B, features)
    fvs = _gen_fvs(candidates, table_a_preproc, table_b_preproc, output_col, fill_na, features)
    log.info('scoring feature vectors')
    positively_correlated = _get_pos_cor_features(features)

    fvs = _score_fvs(fvs, output_col, positively_correlated)    
    log.info(f'scored feature vectors')
    
    return fvs.toPandas() if return_pandas else fvs


def _get_pos_cor_features(features):
    positively_correlated_features = {
        'exact_match',
        'needleman_wunch',
        'smith_waterman',
        'jaccard',
        'overlap_coeff',
        'cosine',
        'monge_elkan_jw',
        'tf_idf',
        'sif'
    }
    return [1 if any(str(f).startswith(prefix) for prefix in positively_correlated_features) else 0
            for f in features]


def _score_fvs(fvs, output_col, positively_correlated):
    pos_cor_array = F.array(*[F.lit(x) for x in positively_correlated])

    return (fvs.withColumn("score", F.aggregate(
        F.zip_with(output_col, pos_cor_array, 
                    lambda x, y: F.nanvl(x, F.lit(0.0)) * y),
                    F.lit(0.0), 
                    lambda acc, x: acc + x)
                    )
            )


def _build(A, B, features):
    A = _prepreprocess_table(A).persist()

    if B is not None:
        B = _prepreprocess_table(B).persist()
    cache = BuildCache()
    
    # Use sequential processing to avoid multiprocessing deadlocks with Spark
    # Spark already provides its own multiprocessing, so we don't need joblib's
    for f in features:
        f.build(A, B, cache)
    cache.clear()
    
    if B is not None:
        table_a_preproc = _create_sqlite_df(A, True, B is None, features)
        table_b_preproc = _create_sqlite_df(B, False, True, features)
        table_b_preproc.to_spark()
    else:
        table_a_preproc = _create_sqlite_df(A, True, B is None, features)
        table_b_preproc = table_a_preproc

    table_a_preproc.to_spark()
    A.unpersist()
    if B is not None:
        B.unpersist()
    return table_a_preproc, table_b_preproc


def _prepreprocess_table(df):
    part_size = 5000
    df = repartition_df(df, part_size, '_id')\
        .select('_id', *[F.col(c).cast('string') for c in df.columns if c != '_id'])
    return df


def _create_sqlite_df(df, pp_for_a, pp_for_b, features):
    if not pp_for_a and not pp_for_b:
        raise RuntimeError('preprocessing must be done for a and/or b')

    schema = T.StructType([
        df.schema['_id'],
        T.StructField('pickle',  T.BinaryType())
    ])

    # project out unused columns
    #df = df.select('_id', *self._projected_columns)
    cols = _get_processing_columns(df, pp_for_a, pp_for_b, features)
    df = df.mapInPandas(lambda x : _preprocess(x, pp_for_a, pp_for_b, features), schema)

    sqlite_df = MemmapDataFrame.from_spark_df(df, 'pickle', cols)

    return sqlite_df


def _get_processing_columns(df, pp_for_a, pp_for_b, features):
    data = df.limit(5).toPandas().set_index('_id')
    data = _preprocess_data(data, pp_for_a, pp_for_b, features)
    # Exclude _id from processing columns since it's the index, not a feature
    return [col for col in data.columns if col != '_id']


def _preprocess_data(data, pp_for_a, pp_for_b, features):
    if pp_for_a:
        for f in features:
            data = f.preprocess(data, True)
    if pp_for_b:
        for f in features:
            data = f.preprocess(data, False)
    return data


def _preprocess(df_itr, pp_for_a, pp_for_b, features):
    preprocess_chunk_size = 100
    for dataframe in df_itr:
        for start in range(0, len(dataframe), preprocess_chunk_size):
            if start >= len(dataframe):
                break
            end = min(start + preprocess_chunk_size, len(dataframe))
            df = dataframe.iloc[start:end].set_index('_id')
            df = _preprocess_data(df, pp_for_a, pp_for_b, features)

            df = df.apply(lambda x : MemmapDataFrame.compress(pickle.dumps(x.values)), axis=1)\
                    .to_frame(name='pickle')\
                    .reset_index(drop=False)

            yield df


def _gen_fvs(pairs, table_a_preproc, table_b_preproc, output_col, fill_na, features):
    if table_a_preproc is None:
        raise RuntimeError('FVGenerator must be built before generating feature vectors')

    fields = pairs.drop('id1_list').schema.fields
    for i, f in enumerate(fields):
        # is an array field
        if hasattr(f.dataType, 'elementType'):
            fields[i] = T.StructField(f.name, f.dataType.elementType)

    schema = T.StructType(fields)\
        .add('id1', 'long')\
        .add('fv', T.ArrayType(T.FloatType()))\

    pairs = repartition_df(pairs, 50, 'id2')

    def generate_feature_vectors_udf(table_a_preproc, table_b_preproc, fill_na, features):
        def _udf(df_itr):
            for df in df_itr:
                yield from _generate_feature_vectors(df, table_a_preproc, table_b_preproc, fill_na, features)
        return _udf

    fvs = pairs.mapInPandas(generate_feature_vectors_udf(table_a_preproc, table_b_preproc, fill_na, features), schema=schema)\
        .withColumn('_id', F.monotonically_increasing_id())\
        .withColumnRenamed('fv', output_col)

    return fvs


def _generate_feature_vectors(df, table_a_preproc, table_b_preproc, fill_na, features):
    table_a = table_a_preproc
    table_b = table_b_preproc
    table_a.init()
    table_b.init()

    b_recs = table_b.fetch(df['id2'].values)

    for idx, row in df.iterrows():
        b_rec = b_recs.loc[row.id2]
        # for high arity data memory can be a issue
        # fetch records lazily without caching to reduce memory pressure
        a_recs = table_a.fetch(row.id1_list)
        f_mat = _generate_feature_vectors_inner(b_rec, a_recs, fill_na, features)

        row['fv'] = list(f_mat)
        row.rename(index={'id1_list' : 'id1'}, inplace=True)
        yield pd.DataFrame(row.to_dict())


def _generate_feature_vectors_inner(rec, recs, fill_na, features):
    f_cols = [f(rec, recs) for f in features]
    f_mat = np.stack(f_cols, axis=-1).astype(np.float32)

    if fill_na is not None:
        f_mat = np.nan_to_num(f_mat, copy=False, nan=fill_na)
    else:
        f_mat = np.nan_to_num(f_mat, copy=False)
    return f_mat


def score(
        fvs: pd.DataFrame,
        features: pd.DataFrame,
) -> pd.DataFrame:
    """
    computes a score by summing up the positively correlated features score in the feature vectors

    Parameters
    ----------
    fvs : pandas DataFrame
        DataFrame with feature vectors created with the following schema:
        (`id2`, `id1`, `fv`, other columns from candidates)
    features : pandas DataFrame
        a DataFrame containing initialized feature objects for columns in A, B                      
    """
    pass
