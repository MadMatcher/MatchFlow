import zlib
import pyspark
import pandas as pd
from pyspark import SparkContext
import numpy as np
import numba as nb
from contextlib import contextmanager
from pyspark import StorageLevel
from random import randint
import mmh3
import sys
import logging

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
