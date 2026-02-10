import zlib
import pyspark
import pandas as pd
from pyspark import SparkContext
import numpy as np
import numba as nb
from contextlib import contextmanager
from pyspark import StorageLevel
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, explode, size
from random import randint
import mmh3
import sys
import logging
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union
from pathlib import Path
import os
from math import floor
# compression for storage
compress = zlib.compress
decompress = zlib.decompress

logging.basicConfig(
        stream=sys.stderr,
        format='[%(filename)s:%(lineno)s - %(funcName)s() ] %(asctime)-15s : %(message)s',
)


def type_check(var, var_name, expected):
    """
    type checking utility, throw a type error if the var isn't the expected type
    """
    if not isinstance(var, expected):
        raise TypeError(f'{var_name} must be type {expected} (got {type(var)})')

def type_check_iterable(var, var_name, expected_var_type, expected_element_type):
    """
    type checking utility for iterables, throw a type error if the var isn't the expected type
    or any of the elements are not the expected type
    """
    type_check(var, var_name, expected_var_type)
    for e in var:
        if not isinstance(e, expected_element_type):
            raise TypeError(f'all elements of {var_name} must be type{expected_element_type} (got {type(var)})')

def is_null(o):
    """
    check if the object is null, note that this is here to 
    get rid of the weird behavior of np.isnan and pd.isnull
    """
    r = pd.isnull(o)
    return r if isinstance(r, bool) else False

@contextmanager
def persisted(df, storage_level=StorageLevel.MEMORY_AND_DISK):
    """
    context manager for presisting a dataframe in a with statement.
    This automatically unpersists the dataframe at the end of the context
    """
    if df is not None:
        df = df.persist(storage_level) 
    try:
        yield df
    finally:
        if df is not None:
            df.unpersist()

def is_persisted(df):
    """
    check if the pyspark dataframe is persist
    """
    sl = df.storageLevel
    return sl.useMemory or sl.useDisk

def get_logger(name, level=logging.DEBUG):
    """
    Get the logger for a module

    Returns
    -------
    Logger

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger

def repartition_df(df, part_size, by=None):
    """
    repartition the dataframe into chunk of size 'part_size'
    by column 'by'
    """
    cnt = df.count()
    n = max(cnt // part_size, SparkContext.getOrCreate().defaultParallelism * 4)
    n = min(n, cnt)
    if by is not None:
        return df.repartition(n, by)
    else:
        return df.repartition(n)


class SparseVec: 
    def __init__(self, size, indexes, values):    
        self._size = size    
        self._indexes = indexes.astype(np.int32)
        self._values = values.astype(np.float32)

    @property
    def indexes(self):
        return self._indexes

    @property
    def values(self):
        return self._values
    
    def dot(self, other):    
        return _sparse_dot(self._indexes, self._values, other._indexes, other._values)

@nb.njit('float32(int32[::1], float32[::1],int32[::1], float32[::1])')
def _sparse_dot(l_ind, l_val, r_ind, r_val):
    l = 0
    r = 0
    s = 0.0

    while l < l_ind.size and r < r_ind.size:
        li = l_ind[l]
        ri = r_ind[r]
        if li == ri:
            s += l_val[l] * r_val[r]
            l += 1
            r += 1
        elif li < ri:
            l += 1
        else:
            r += 1

    return s

class PerfectHashFunction:

    def __init__(self, seed=None):
        self._seed = seed if seed is not None else randint(0, 2**31)
    

    @classmethod
    def create_for_keys(cls, keys):
        if len(set(keys)) != len(keys):
            raise ValueError('keys must be unique')
        # used because it is ordered
        hashes = {}

        MAX_RETRIES = 10
        for i in range(MAX_RETRIES):
            hashes.clear()
            hash_func = cls()
            hash_vals = map(hash_func.hash, keys)
            for h, k in zip(hash_vals, keys):
                if h in hashes:
                    break
                hashes[h] = k
            if len(hashes) == len(keys):
                break
        else:
            raise RuntimeError(f'max retries ({MAX_RETRIES}) exceeded')

        return hash_func, np.fromiter(hashes.keys(), dtype=np.int64, count=len(hashes))
    
    def hash(self, s):
        return mmh3.hash64(s, self._seed)[0]

# Training Data Persistence Utilities
def save_training_data_streaming(new_batch, parquet_file_path, logger=None):
    """Save training data using streaming writes for efficiency
    
    This method appends new labeled pairs to an existing parquet file without
    loading the entire dataset into memory. Each labeled pair is saved immediately.
    
    Parameters
    ----------
    new_batch : pandas.DataFrame
        New batch of labeled data to append (columns: _id, id1, id2, label)
    parquet_file_path : str
        Path to save the parquet file
    logger : logging.Logger, optional
        Logger instance for logging messages
    """
    if logger is None:
        logger = get_logger(__name__)
        
    # Ensure we only save the essential columns in consistent order
    required_columns = ['_id', 'id1', 'id2', 'feature_vectors', 'label']
    new_batch_clean = new_batch[required_columns].copy().reset_index(drop=True)
    def to_float64_list(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float64).tolist()
        elif isinstance(x, list):
            return [np.float64(i) for i in x]
        else:
            return [np.float64(i) for i in list(x)]
    
    new_batch_clean['feature_vectors'] = new_batch_clean['feature_vectors'].apply(to_float64_list)
        
    try:
        table = pa.Table.from_pandas(new_batch_clean)
        
        if os.path.exists(parquet_file_path):
            # Read existing data and append
            existing_table = pq.read_table(parquet_file_path)
            # Ensure existing data has same columns
            existing_df = existing_table.to_pandas()
            existing_df_clean = existing_df[required_columns].copy().reset_index(drop=True)
            existing_table_clean = pa.Table.from_pandas(existing_df_clean)
            
            combined_table = pa.concat_tables([existing_table_clean, table])
            pq.write_table(combined_table, parquet_file_path)
            logger.info(f'Appended {len(new_batch)} labeled pairs to '
                       f'{parquet_file_path}')
        else:
            # Create new file
            pq.write_table(table, parquet_file_path)
            logger.info(f'Created new training data file: {parquet_file_path}')
            

    except Exception as e:
        logger.warning(f'Streaming save failed: {e}, falling back to pandas save')
        _save_with_pandas(new_batch_clean, parquet_file_path, logger)


def load_training_data_streaming(parquet_file_path, logger=None):
    """Load training data from a parquet file.
    
    Parameters
    ----------
    parquet_file_path : str
        Path to the parquet file
    logger : logging.Logger, optional
        Logger instance for logging messages
        
    Returns
    -------
    pandas.DataFrame or None
        Training data if file exists, None otherwise
    """
    if logger is None:
        logger = get_logger(__name__)
        
    try:
        if os.path.exists(parquet_file_path):
            # Read all data efficiently
            table = pq.read_table(parquet_file_path)
            training_data = table.to_pandas()

            # Type check and convert columns
            required_columns = ['_id', 'id1', 'id2', 'feature_vectors', 'label']
            for col in required_columns:
                if col not in training_data.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Ensure integer columns are int64
            for col in ['_id', 'id1', 'id2']:
                if training_data[col].dtype != 'int64':
                    training_data[col] = training_data[col].astype('int64')

            # Ensure label is float
            if training_data['label'].dtype != 'float64':
                training_data['label'] = training_data['label'].astype('float64')

            # Ensure features is a list of floats (array<float>)
            def to_float_list(x):
                if isinstance(x, np.ndarray):
                    return x.astype(float).tolist()
                elif isinstance(x, list):
                    return [float(i) for i in x]
                else:
                    return list(x)  # fallback, may error if not iterable
            training_data['feature_vectors'] = training_data['feature_vectors'].apply(to_float_list)

            logger.info(f'Loaded {len(training_data)} labeled pairs from '
                       f'{parquet_file_path}')
            return training_data
        return None
        
    except Exception as e:
        logger.warning(f'Streaming load failed: {e}, falling back to pandas read')
        return _load_with_pandas(parquet_file_path, logger)


def _save_with_pandas(training_data, parquet_file_path, logger):
    """Fallback save method using pandas"""
    try:
        if os.path.exists(parquet_file_path):
            # Read existing data and append
            existing_data = pd.read_parquet(parquet_file_path)
            combined_data = pd.concat([existing_data, training_data], 
                                    ignore_index=True)
            combined_data.to_parquet(parquet_file_path, index=False)
        else:
            training_data.to_parquet(parquet_file_path, index=False)
        logger.info(f'Saved {len(training_data)} labeled pairs to '
                   f'{parquet_file_path}')
    except Exception as e:
        logger.warning(f'Pandas save failed: {e}')


def _load_with_pandas(parquet_file_path, logger):
    """Fallback load method using pandas"""
    try:
        if os.path.exists(parquet_file_path):
            training_data = pd.read_parquet(parquet_file_path)
            logger.info(f'Loaded {len(training_data)} labeled pairs from '
                       f'{parquet_file_path}')
            return training_data
        return None
    except Exception as e:
        logger.warning(f'Pandas load failed: {e}')
        return None


def adjust_iterations_for_existing_data(existing_data_size, n_fvs, batch_size, max_iter):
    """Calculate remaining iterations based on existing data and constraints
    
    This function is designed for batch active learning where iterations
    correspond to discrete batches of labeled examples.
    
    Parameters
    ----------
    existing_data_size : int
        Number of existing labeled examples
    n_fvs : int
        Total number of feature vectors
    batch_size : int
        Number of examples per batch
    max_iter : int
        Maximum number of iterations
        
    Returns
    -------
    int
        Adjusted number of iterations
    """
    completed_iterations = floor(existing_data_size / batch_size)
    remaining_iterations = max_iter - completed_iterations    
    return max(0, remaining_iterations)


def adjust_labeled_examples_for_existing_data(existing_data_size, max_labeled):
    """Calculate remaining labeled examples for continuous active learning
    
    This function is designed for continuous active learning where we track
    the total number of labeled examples rather than iterations.
    
    Parameters
    ----------
    existing_data_size : int
        Number of existing labeled examples
    max_labeled : int
        Maximum number of examples to label
        
    Returns
    -------
    int
        Remaining number of examples to label
    """
    if existing_data_size >= max_labeled:
        return 0
    return max_labeled - existing_data_size


def convert_arrays_for_spark(df):
    """
    Convert numpy arrays and lists in a pandas DataFrame to PySpark-compatible format.
    
    This function detects columns containing numpy arrays or lists and converts them
    to Python lists, which PySpark can properly infer and handle.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that may contain numpy arrays or lists
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with arrays converted to lists for PySpark compatibility
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df
    
    df = df.copy()
    
    # Check each column for arrays/lists
    for col in df.columns:
        sample_value = df[col].iloc[0]
        
        # Check if it's a numpy array or list that needs conversion
        if hasattr(sample_value, 'tolist') and hasattr(sample_value, 'dtype'):
            # This is a numpy array, convert to list
            df[col] = df[col].apply(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x
            )
        elif isinstance(sample_value, list):
            # This is already a list, but check if it contains numpy arrays
            if len(sample_value) > 0 and hasattr(sample_value[0], 'tolist'):
                # List contains numpy arrays, convert them
                df[col] = df[col].apply(
                    lambda x: [item.tolist() if hasattr(item, 'tolist') else item for item in x] if isinstance(x, list) else x
                )

    return df


def check_tables(table_a: Union[pd.DataFrame, SparkDataFrame], table_b: Union[pd.DataFrame, SparkDataFrame]):
    """
    Check that both table_a and table_b have the column '_id'.
    Check that both id columns are unique.
    """
    logger = logging.getLogger(__name__)

    if isinstance(table_a, pd.DataFrame):
        if isinstance(table_b, pd.DataFrame):
            if '_id' not in table_a.columns:
                raise ValueError(f"table_a must have the column '_id'.  Available columns: {table_a.columns}")
            if '_id' not in table_b.columns:
                raise ValueError(f"table_b must have the column '_id'.  Available columns: {table_b.columns}")
            if table_a['_id'].nunique() != len(table_a):
                raise ValueError(f"table_a '_id' column must be unique.")
            if table_b['_id'].nunique() != len(table_b):
                raise ValueError(f"table_b '_id' column must be unique.")
        else:
            raise ValueError("table_a and table_b must both be either pandas DataFrames or Spark DataFrames")
    elif isinstance(table_a, SparkDataFrame):
        if isinstance(table_b, SparkDataFrame):
            if '_id' not in table_a.columns:
                raise ValueError(f"table_a must have the column '_id'.  Available columns: {table_a.columns}")
            if '_id' not in table_b.columns:
                raise ValueError(f"table_b must have the column '_id'.  Available columns: {table_b.columns}")
            if table_a.select('_id').distinct().count() != table_a.count():
                raise ValueError("table_a '_id' column must be unique")
            if table_b.select('_id').distinct().count() != table_b.count():
                raise ValueError("table_b '_id' column must be unique")
        else:
            raise ValueError("table_a and table_b must both be either pandas DataFrames or Spark DataFrames")
    else:
        raise ValueError("table_a and table_b must both be either pandas DataFrames or Spark DataFrames")

    logger.warning("check_tables: table_a and table_b formats are correct")
    

def check_candidates(candidates, table_a, table_b):
    """
    Check that the candidates have the column 'id2' and 'id1_list'.
    Check that the id2 column is unique.
    Check that the id1_list column is a list of ids.
    Check that the ids in the id1_list column are present in the table_a id column.
    Check that the ids in the id2 column are present in the table_b id column.
    """
    logger = logging.getLogger(__name__)

    if isinstance(candidates, pd.DataFrame):
        if 'id2' not in candidates.columns:
            raise ValueError(f"candidates must have the column 'id2'.  Available columns: {candidates.columns}")
        if 'id1_list' not in candidates.columns:
            raise ValueError(f"candidates must have the column 'id1_list'.  Available columns: {candidates.columns}")
        if candidates['id2'].nunique() != len(candidates):
            raise ValueError("candidates 'id2' column must be unique")
        if not candidates['id1_list'].apply(lambda x: isinstance(x, (list, np.ndarray))).all():
            raise ValueError("candidates 'id1_list' column must be a list of ids")
        if not candidates['id1_list'].apply(lambda x: all(i in table_a['_id'].tolist() for i in x)).all():
            raise ValueError("candidates 'id1_list' column must only contain ids that are present in the table_a '_id' column")
        if not candidates['id2'].apply(lambda x: x in table_b['_id'].tolist()).all():
            raise ValueError("candidates 'id2' column must only contain ids that are present in the table_b '_id' column")
    elif isinstance(candidates, SparkDataFrame):
        if 'id2' not in candidates.columns:
            raise ValueError(f"candidates must have the column 'id2'.  Available columns: {candidates.columns}")
        if 'id1_list' not in candidates.columns:
            raise ValueError(f"candidates must have the column 'id1_list'.  Available columns: {candidates.columns}")
        if candidates.select('id2').distinct().count() != candidates.count():
            raise ValueError("candidates 'id2' column must be unique")
        
        # Check that id1_list is an array type
        id1_list_schema = dict(candidates.dtypes)['id1_list']
        if not id1_list_schema.startswith('array'):
            raise ValueError("candidates 'id1_list' column must be a list of ids")
        
        # Check id2 validity using anti-join
        table_b_ids_df = table_b.select('_id').distinct().withColumnRenamed('_id', 'id2')
        invalid_id2 = candidates.join(
            table_b_ids_df, on='id2', how='left_anti'
        )
        if invalid_id2.count() > 0:
            raise ValueError("candidates 'id2' column must only contain ids that are present in the table_b '_id' column")
        
        # Check id1_list validity by exploding and joining
        table_a_ids_df = table_a.select('_id').distinct().withColumnRenamed('_id', 'id1')
        candidates_exploded = candidates.select(
            col('id2'),
            explode(col('id1_list')).alias('id1')
        )
        invalid_id1 = candidates_exploded.join(
            table_a_ids_df, on='id1', how='left_anti'
        )
        if invalid_id1.count() > 0:
            raise ValueError("candidates 'id1_list' column must only contain ids that are present in the table_a '_id' column")
    else:
        raise ValueError("candidates must be a pandas DataFrame or Spark DataFrame")

    logger.warning("check_candidates: candidates formats are correct")


def check_labeled_data(labeled_data, table_a, table_b, label_column_name):
    """
    Check that the labeled_data have the column 'id2', 'id1_list', and label_column_name.
    Check that the label_column_name column is a list of floats.
    Check that the id2 column is unique.
    Check that the id1_list column is a list of ids.
    Check that the ids in the id1_list column are present in the table_a id column.
    Check that the ids in the id2 column are present in the table_b id column.
    Check that the label_column list is the same length as the id1_list column.
    """
    logger = logging.getLogger(__name__)

    if isinstance(labeled_data, pd.DataFrame):
        if 'id2' not in labeled_data.columns:
            raise ValueError(f"candidates must have the column 'id2'.  Available columns: {labeled_data.columns}")
        if 'id1_list' not in labeled_data.columns:
            raise ValueError(f"candidates must have the column 'id1_list'.  Available columns: {labeled_data.columns}")
        if labeled_data['id2'].nunique() != len(labeled_data):
            raise ValueError("candidates 'id2' column must be unique")
        if not labeled_data['id1_list'].apply(lambda x: isinstance(x, (list, np.ndarray))).all():
            raise ValueError("candidates 'id1_list' column must be a list of ids")
        if not labeled_data['id1_list'].apply(lambda x: all(i in table_a['_id'].tolist() for i in x)).all():
            raise ValueError("candidates 'id1_list' column must only contain ids that are present in the table_a '_id' column")
        if not labeled_data['id2'].apply(lambda x: x in table_b['_id'].tolist()).all():
            raise ValueError("candidates 'id2' column must only contain ids that are present in the table_b '_id' column")
        if not labeled_data.apply(lambda row: len(row[label_column_name]) == len(row['id1_list']), axis=1).all():
            raise ValueError(f"labeled_data '{label_column_name}' column must be a list the same length as its corresponding 'id1_list' column")
    elif isinstance(labeled_data, SparkDataFrame):
        if 'id2' not in labeled_data.columns:
            raise ValueError(f"candidates must have the column 'id2'.  Available columns: {labeled_data.columns}")
        if 'id1_list' not in labeled_data.columns:
            raise ValueError(f"candidates must have the column 'id1_list'.  Available columns: {labeled_data.columns}")
        if label_column_name not in labeled_data.columns:
            raise ValueError(f"labeled_data must have the column '{label_column_name}'.  Available columns: {labeled_data.columns}")
        if labeled_data.select('id2').distinct().count() != labeled_data.count():
            raise ValueError("candidates 'id2' column must be unique")
        
        # Check that id1_list is an array type
        id1_list_schema = dict(labeled_data.dtypes)['id1_list']
        if not id1_list_schema.startswith('array'):
            raise ValueError("candidates 'id1_list' column must be a list of ids")
        
        # Check id2 validity using anti-join
        table_b_ids_df = table_b.select('_id').distinct().withColumnRenamed('_id', 'id2')
        invalid_id2 = labeled_data.join(
            table_b_ids_df, on='id2', how='left_anti'
        )
        if invalid_id2.count() > 0:
            raise ValueError("candidates 'id2' column must only contain ids that are present in the table_b '_id' column")
        
        # Check id1_list validity by exploding and joining
        table_a_ids_df = table_a.select('_id').distinct().withColumnRenamed('_id', 'id1')
        labeled_exploded = labeled_data.select(
            col('id2'),
            explode(col('id1_list')).alias('id1')
        )
        invalid_id1 = labeled_exploded.join(
            table_a_ids_df, on='id1', how='left_anti'
        )
        if invalid_id1.count() > 0:
            raise ValueError("candidates 'id1_list' column must only contain ids that are present in the table_a '_id' column")
        
        # Check label column length matches id1_list length using Spark SQL
        invalid_length = labeled_data.filter(
            size(col(label_column_name)) != size(col('id1_list'))
        )
        if invalid_length.count() > 0:
            raise ValueError(f"labeled_data '{label_column_name}' column must be a list the same length as its corresponding 'id1_list' column")
    else:
        raise ValueError("candidates must be a pandas DataFrame or Spark DataFrame")

    logger.warning("check_labeled_data: labeled_data formats are correct")

def check_gold_data(gold_data, table_a, table_b):
    """
    Gold data must have the columns 'id1' and 'id2'.
    Check that the ids in the id1 column are present in the table_a '_id' column.
    Check that the ids in the id2 column are present in the table_b '_id' column.
    """
    logger = logging.getLogger(__name__)

    if isinstance(gold_data, pd.DataFrame):
        if 'id1' not in gold_data.columns:
            raise ValueError(f"gold_data must have the column 'id1'.  Available columns: {gold_data.columns}")
        if 'id2' not in gold_data.columns:
            raise ValueError(f"gold_data must have the column 'id2'.  Available columns: {gold_data.columns}")
        if not gold_data['id1'].apply(lambda x: x in table_a['_id'].tolist()).all():
            raise ValueError("gold_data 'id1' column must only contain ids that are present in the table_a '_id' column")
        if not gold_data['id2'].apply(lambda x: x in table_b['_id'].tolist()).all():
            raise ValueError("gold_data 'id2' column must only contain ids that are present in the table_b '_id' column")
    elif isinstance(gold_data, SparkDataFrame):
        if 'id1' not in gold_data.columns:
            raise ValueError(f"gold_data must have the column 'id1'.  Available columns: {gold_data.columns}")
        if 'id2' not in gold_data.columns:
            raise ValueError(f"gold_data must have the column 'id2'.  Available columns: {gold_data.columns}")
        
        # Check id1 validity
        table_a_ids_df = table_a.select('_id').distinct().withColumnRenamed('_id', 'id1')
        invalid_id1 = gold_data.join(
            table_a_ids_df, on='id1', how='left_anti'
        )
        if invalid_id1.count() > 0:
            raise ValueError("gold_data 'id1' column must only contain ids that are present in the table_a '_id' column")
        
        # Check id2 validity
        table_b_ids_df = table_b.select('_id').distinct().withColumnRenamed('_id', 'id2')
        invalid_id2 = gold_data.join(
            table_b_ids_df, on='id2', how='left_anti'
        )
        if invalid_id2.count() > 0:
            raise ValueError("gold_data 'id2' column must only contain ids that are present in the table_b '_id' column")
    else:
        raise ValueError("gold_data must be a pandas DataFrame or Spark DataFrame")

    logger.warning("check_gold_data: gold_data formats are correct")
