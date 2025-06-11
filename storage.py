import sqlite3
import numpy as np
from pyspark import SparkFiles, SparkContext, StorageLevel
import numba as nb
import pyspark.sql.functions as F
import pandas as pd
import pickle
from timeit import timeit
from random import choices
from tempfile import mkstemp
from pathlib import Path
import os
from itertools import islice
from utils import get_logger
from utils import decompress, compress
from joblib import Parallel, delayed

log = get_logger(__name__)



class MemmapArray:

    def __init__(self, arr):
        self._dtype = arr.dtype
        self._shape = arr.shape
        self._local_mmap_file = Path(mkstemp(suffix='.mmap_arr')[1])
        mmap = np.memmap(self._local_mmap_file, shape=self._shape, dtype=self._dtype, mode='w+')
        mmap[:] = arr[:]
        mmap.flush()
        self._mmap_arr = None

    @property
    def values(self):
        return self._mmap_arr

    @property
    def shape(self):
        return self._shape
    
    def __len__(self):
        return self._shape[0]

    def init(self):
        if self._mmap_arr is None:
            f = self._local_mmap_file
            if self._on_spark:
                f = SparkFiles.get(f.name)
                if not os.path.exists(f):
                    raise RuntimeError('cannot find database file at {f}')
            self._mmap_arr = np.memmap(f, mode='r', shape=self._shape, dtype=self._dtype)
    
    def delete(self):
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()

    def to_spark(self):
        SparkContext.getOrCreate().addFile(str(self._local_mmap_file))
        self._on_spark = True


class MemmapDataFrame:

    def __init__(self):
        self._local_mmap_file = Path(mkstemp(suffix='.df.db')[1])
        self._id_to_offset_map = None
        self._offset_arr = None
        self._mmap_arr = None
        self._map_arr_shape = None
        self._on_spark = False
        self._conn = None
        self._columns = None

    
    @staticmethod
    def compress(o):
        return compress(o)

    @staticmethod
    def decompress(o):
        return decompress(o)
    
    def write_chunk(self, fd, _id, pic):
        idx = np.searchsorted(self._index_arr, _id)
        start = self._offset_arr[idx]
        os.pwrite(fd, memoryview(pic), start)


    @classmethod
    def from_spark_df(cls, spark_df, pickle_column, stored_columns, id_col='_id'):
        obj = cls()
        obj._columns = stored_columns

        spark_df = spark_df.select(
                        F.col(id_col).alias('_id'),
                        F.col(pickle_column).alias('pickle'), 
                        F.length(pickle_column).alias('sz')
                    )

        size_arrs = []
        id_arrs = []
        seq_col = 'pickle'
        itr = spark_to_pandas_stream(spark_df, 10000)

        local_mmap_file = obj._local_mmap_file 
        # buffer 256MB at a time
        with open(local_mmap_file, 'wb', buffering=2**20 * 256) as ofs:
            for part in itr:
                id_arrs.append(part[id_col].to_numpy(dtype=np.int64))
                size_arrs.append(part['sz'].to_numpy(dtype=np.uint64))
                for seq in part[seq_col]:
                    ofs.write(memoryview(seq))

        id_arr = np.concatenate(id_arrs, dtype=np.int64)
        size_arr = np.concatenate(size_arrs) 

        obj._id_to_offset_map = LongIntHashMap.build(
                id_arr+1,
                np.arange(len(id_arr), dtype=np.int32)
        )
        offset_arr = np.cumsum(np.concatenate([np.zeros(1, dtype=np.uint64), size_arr]))
        obj._offset_arr = MemmapArray(offset_arr)
        total_bytes = offset_arr[-1]
        obj._mmap_arr_shape = total_bytes

        return obj
 
    def init(self):
        if self._mmap_arr is None:
            f = self._local_mmap_file
            if self._on_spark:
                f = SparkFiles.get(f.name)
                if not os.path.exists(f):
                    raise RuntimeError('cannot find database file at {f}')
            self._mmap_arr = np.memmap(f, mode='r', shape=self._mmap_arr_shape)

        self._id_to_offset_map.init()
        self._offset_arr.init()
    
    def delete(self):
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()

    def to_spark(self):
        self._id_to_offset_map.to_spark()
        self._offset_arr.to_spark()
        SparkContext.getOrCreate().addFile(str(self._local_mmap_file))
        self._on_spark = True
    

    def fetch(self, ids):
        self.init()
        ids = np.array(ids) 
        idxes = self._id_to_offset_map[ids+1]

        if np.any(idxes < 0):
            raise ValueError('unknown id')  

        starts = self._offset_arr.values[idxes]
        ends = self._offset_arr.values[idxes+1]

        rows = np.array([pickle.loads(self.decompress(self._mmap_arr[start:end])) for start, end in zip(starts, ends)], dtype=object)
        df = pd.DataFrame(rows, index=ids, columns=self._columns, dtype=object)
        return df


class SqliteDataFrame:

    def __init__(self):
        self._local_tmp_file = Path(mkstemp(suffix='.df.db')[1])
        self._on_spark = False
        self._conn = None
        self._columns = None


    @classmethod
    def from_spark_df(cls, spark_df, pickle_column, stored_columns, id_col='_id'):
        chunk_size = 250
        obj = cls()
        obj._columns = stored_columns
        obj._init_db()

        spark_df = spark_df.select(
                        F.col(id_col).alias('_id'),
                        F.col(pickle_column).alias('pickle')
                    ).persist(StorageLevel.DISK_ONLY)
        spark_df.count()

        log.info('building sqlite dataframe')
        conn = obj._get_conn()       
        itr = spark_df.toLocalIterator(True)
        df = pd.DataFrame.from_records(list(islice(itr, chunk_size)), columns=['_id', 'pickle'])
        while len(df) > 0:
            df.to_sql('dataframe', conn, if_exists='append', index=False, method='multi')
            df = pd.DataFrame.from_records(list(islice(itr, chunk_size)), columns=['_id', 'pickle'])
        conn.commit()
        conn.close()

        spark_df.unpersist()
        return obj
 
    def init(self):
        if self._conn is None:
            self._conn = self._get_conn()

    def to_spark(self):
        SparkContext.getOrCreate().addFile(str(self._local_tmp_file))
        self._on_spark = True
    
    def _init_db(self):
        conn = self._get_conn()
        conn.execute('DROP TABLE IF EXISTS dataframe;')
        conn.execute('''CREATE TABLE dataframe (
                            _id,
                            pickle,
                            PRIMARY KEY (_id)
                    ) WITHOUT ROWID;
                '''
                )

    def _get_conn(self):
        f = self._local_tmp_file
        if self._on_spark:
            f = SparkFiles.get(f.name)
            if not os.path.exists(f):
                raise RuntimeError('cannot find database file at {f}')
        f = str(f)
        conn = sqlite3.connect(f)
        conn.execute('PRAGMA synchronous=OFF')
        conn.execute('PRAGMA journal_mode=OFF')
        sz = os.path.getsize(f)
        if sz > 0:
            conn.execute(f'PRAGMA mmap_size = {sz};')
        conn.commit()
        return conn


    def fetch(self, ids):
        self.init()
        # TODO add caching for queries?
        query = f'''
        SELECT _id, pickle
        FROM dataframe
        WHERE _id in (?{",?" * (len(ids)-1)});
        '''

        res = self._conn.execute(query, ids).fetchall()
        index = [t[0] for t in res]
        if len(res) != len(ids):
            missings_ids = set(ids) - set(index) 
            raise RuntimeError(f'not all ids found : {len(res)} of {len(ids)} returned, missing {missings_ids}')

        rows = [pickle.loads(t[1]) for t in res]
        df = pd.DataFrame(rows, index=index, columns=self._columns, dtype=object)
        return df


class SqliteDict:

    def __init__(self):
        self._local_tmp_file = Path(mkstemp(suffix='.df.db')[1])
        self._on_spark = False
        self._conn = None


    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj._init_db()

        log.info('building sqlite dict')
        conn = obj._get_conn()       

        df = pd.DataFrame.from_records(list(d.items()), columns=['key', 'val'])
        chunk_size = 10000
        for i in range(0, len(df), chunk_size):
            df.iloc[i : min(i+chunk_size, len(df))].to_sql('dataframe', conn, if_exists='append', index=False, method='multi')

        if i != len(df):
            df.iloc[i:].to_sql('dataframe', conn, if_exists='append', index=False, method='multi')

        conn.commit()
        conn.close()

        return obj
 
    
    def init(self):
        if self._conn is None:
            self._conn = self._get_conn()

    def deinit(self):
        self._conn = None

    def to_spark(self):
        SparkContext.getOrCreate().addFile(str(self._local_tmp_file))
        self._on_spark = True
    
    def _init_db(self):
        conn = self._get_conn()
        conn.execute('DROP TABLE IF EXISTS dict;')
        conn.execute('''CREATE TABLE dict (
                            key,
                            val,
                            PRIMARY KEY (key, val)
                    ) WITHOUT ROWID;
                '''
                )

    def _get_conn(self):
        f = self._local_tmp_file
        if self._on_spark:
            f = SparkFiles.get(f.name)
            if not os.path.exists(f):
                raise RuntimeError('cannot find database file at {f}')
        f = str(f)
        conn = sqlite3.connect(f)
        conn.execute('PRAGMA synchronous=OFF')
        conn.execute('PRAGMA journal_mode=OFF')
        sz = os.path.getsize(f)
        if sz > 0:
            conn.execute(f'PRAGMA mmap_size = {sz};')
        conn.commit()
        return conn


    def __getitem__(self, keys):
        self.init()
        keys = list(keys)

        if len(keys) == 0:
            return []

        # TODO add caching for queries?
        query = f'''
        SELECT key, val
        FROM dataframe
        WHERE key in (?{",?" * (len(keys)-1)});
        '''

        res = self._conn.execute(query, keys).fetchall()
        res_dict = dict(res)

        missing = {k for k in keys if k not in res_dict}
        if len(missing) != 0:
            raise KeyError(f'not all ids found : {len(res)} of {len(keys)} returned, missing {missing}')

        return [res_dict[k] for k in keys]



map_entry_t = np.dtype([('hash', np.uint64), ('val', np.int32)])
numba_map_entry_t = nb.from_dtype(map_entry_t)

njit_kwargs = {
        'cache' : False, 
        'parallel' : False
}


#@nb.njit(nb.void(numba_map_entry_t[:], nb.uint64, nb.int32), cache=True, parallel=False)
@nb.njit(**njit_kwargs)
def hash_map_insert_key(arr, key, val):
    if key == 0:
        raise ValueError('keys must be non zero')

    i = key % len(arr)
    while True:
        if arr[i].hash == 0 or arr[i].hash == key:
            arr[i].hash = key
            arr[i].val = val
            return 
        else:
            i += 1
            if i == len(arr):
                i = 0

#@nb.njit(nb.void(numba_map_entry_t[:], nb.uint64[:], nb.int32[:]), cache=True, parallel=False)
@nb.njit(**njit_kwargs)
def hash_map_insert_keys(arr, keys, vals):
    for i in range(len(keys)):
        hash_map_insert_key(arr, keys[i], vals[i])


sigs = [nb.int32(nb.types.Array(numba_map_entry_t, 1, 'C', readonly=r), nb.uint64) for r in [True, False]]
#@nb.njit(sigs, cache=True, parallel=False)
@nb.njit(**njit_kwargs)
def hash_map_get_key(arr, key):
    i = key % len(arr)
    while True:
        if arr[i].hash == key:
            # hash found, return value at position
            return arr[i].val 
        elif arr[i].hash == 0:
            # hash not found, return -1
            return -1
        else:
            i += 1
            if i == len(arr):
                i = 0

sigs = [nb.int32[:](nb.types.Array(numba_map_entry_t, 1, 'C', readonly=r), nb.uint64[:]) for r in [True, False]]
#@nb.njit(sigs, cache=True, parallel=False)
@nb.njit(**njit_kwargs)
def hash_map_get_keys(arr, keys):
    out = np.empty(len(keys), dtype=np.int32)
    for i in range(len(keys)):
        out[i] = hash_map_get_key(arr, keys[i])
    return out


class DistributableHashMap:

    def __init__(self, arr):
        self._memmap_arr = MemmapArray(arr)

    @property
    def _arr(self):
        return self._memmap_arr.values

    @property
    def on_spark(self):
        return self._memmap_arr.on_spark

    def init(self):
        self._memmap_arr.init()

    def to_spark(self):
        self._memmap_arr.to_spark()



class LongIntHashMap(DistributableHashMap):

    def __init__(self, arr):
        super().__init__(arr)

    @classmethod
    def build(cls, longs, ints, load_factor=.75):
        map_size = int(len(longs) / load_factor)
        arr = np.zeros(map_size, dtype=map_entry_t)
        hash_map_insert_keys(arr, longs, ints)

        return cls(arr)

    def __getitem__(self, keys):
        if isinstance(keys, (np.uint64, np.int64, int)):
            return hash_map_get_key(self._arr, np.uint64(keys))
        elif isinstance(keys, np.ndarray):
            return hash_map_get_keys(self._arr, keys)
        else:
            raise TypeError(f'unknown type {type(keys)}')

def spark_to_pandas_stream(df, chunk_size):
    df = df.repartition(max(1, df.count() // chunk_size), '_id')\
            .rdd\
            .mapPartitions(lambda x : iter([pd.DataFrame([e.asDict(True) for e in x]).convert_dtypes()]) )\
            .persist(StorageLevel.DISK_ONLY)
    # trigger read
    df.count()
    for batch in df.toLocalIterator(True):
        yield batch

    df.unpersist()
