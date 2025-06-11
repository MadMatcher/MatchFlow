from typing import List, Optional, Callable, Any
import pandas as pd
import numpy as np
from time import time
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
import pickle
from storage import MemmapDataFrame
from threading import Lock
from joblib import Parallel, delayed
from utils import repartition_df
from tokenizer import (
    AlphaNumericTokenizer,
    NumericTokenizer,
    QGramTokenizer,
    StrippedQGramTokenizer,
    WhiteSpaceTokenizer,
    StrippedWhiteSpaceTokenizer,
    ShingleTokenizer,
)

from feature import (
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

def create_features(
    A: pd.DataFrame,
    B: pd.DataFrame,
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

    A : pandas DataFrame
        the records of table A
    B : pandas DataFrame
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
    pandas DataFrame
        a DataFrame containing initialized feature objects for columns in A, B
    """
    if sim_functions is None:
        sim_functions = SIM_FUNCTIONS
    if tokenizers is None:
        tokenizers = TOKENIZERS

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
    output_col: str = 'features',
    fill_na: object = None,
) -> pd.DataFrame:
    """
    applies the featurizer to the record pairs in candidates

    Parameters
    ----------
    features : List[Callable]
        a DataFrame containing initialized feature objects for columns in A, B
    A : pandas DataFrame
        the records of table A
    B : pandas DataFrame
        the records of table B
    candidates : pandas DataFrame
        id pairs of A and B that are potential matches
    output_col : str
        the name of the column for the resulting feature vectors, default `fvs`
    fill_na : 
        value to fill in for missing data, default None
    Returns
    -------
    pandas DataFrame
        DataFrame with feature vectors created with the following schema:
        (`id2`, `id1`, `fv`, other columns from candidates)
    """
    spark = SparkSession.builder.getOrCreate()
    if isinstance(A, pd.DataFrame):
        A = spark.createDataFrame(A)
    if isinstance(B, pd.DataFrame):
        B = spark.createDataFrame(B)
    if isinstance(candidates, pd.DataFrame):
        candidates = spark.createDataFrame(candidates)
    start = time()
    table_a_preproc, table_b_preproc = _build(A, B, features)
    end = time()
    print(f"{end-start} seconds to preprocess tables")
    start = time()
    fvs = _gen_fvs(candidates, table_a_preproc, table_b_preproc, output_col, fill_na, features)
    end = time()
    print(f"{end-start} seconds to generate fvs")
    fvs = fvs.toPandas()
    end = time()
    print(f"{end-start} seconds to convert to pandas")
    return fvs


def _build(A, B, features):
    A = _prepreprocess_table(A).persist()

    if B is not None:
        B = _prepreprocess_table(B).persist()
    cache = BuildCache()
    pool = Parallel(n_jobs=-1, backend='threading')
    pool(delayed(f.build)(A, B, cache) for f in features)
    cache.clear()
    if B is not None:
        delayed_build = delayed(_create_sqlite_df)
        table_a_preproc, table_b_preproc = pool([delayed_build(A, True, B is None, features), delayed_build(B, False, True, features)])
        table_b_preproc.to_spark()
    else:
        table_a_preproc = _create_sqlite_df(A, True, B is None)
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
    return data.columns


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
        for start in range(0, len(dataframe), preprocess_chunk_size):  # TODO: 100 used to be self._preprocess_chunk_size, may want to make a variable
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
