import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip, safe_sparse_dot
from utils import is_null, get_logger
from utils import SparseVec
import gc
import numba as nb
from copy import deepcopy
from utils import PerfectHashFunction
from storage import MemmapArray
#import spacy 


log = get_logger(__name__)




class TFIDFVectorizer:

    def __init__(self):
        self._N = None
        self._hash_func = None
        self._hashes = None
        self._idfs = None

    def build_from_doc_freqs(self, doc_freqs):
        self._idfs = MemmapArray(_doc_freq_to_idf(doc_freqs.doc_freqs_, doc_freqs.corpus_size_))
        self._idfs.to_spark()

        self._hashes = MemmapArray(doc_freqs.hashes_)
        self._hashes.to_spark()

        self._hash_func = deepcopy(doc_freqs.hash_func_)
        self._N = len(self._idfs)

    def _hash(self, s):
        return self._hash_func.hash(s)
    
    def out_col_name(self, base):
        return f'term_vec({base})'
    
    def init(self):
        self._idfs.init()
        self._hashes.init()

    def vectorize(self, tokens):
        if is_null(tokens):
            return None
        hashes = np.fromiter((self._hash(t) for t in tokens), dtype=np.int64, count=len(tokens))
        hashes, values = np.unique(hashes, return_counts=True)
        return SparseVec(self._N, *_vectorize_tfidf(self._hashes.values, self._idfs.values, hashes, values))

@nb.njit('float32[:](int64[::1], int64)')
def _doc_freq_to_idf(doc_freq, corpus_size):
    return (np.log((corpus_size +1 ) / (doc_freq + 1)) + 1).astype(np.float32)




@nb.njit(
    nb.types.Tuple((
            nb.types.int32[:],
            nb.types.float32[:]
        ))(
            nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
            nb.types.Array(nb.types.float32, 1, 'C', readonly=True),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False)
    )
)
def _vectorize_tfidf(hash_idx, idfs, hashes, values):
        # sorted hashes implies idxes are sorted because 
        # self._hashes is also sorted
        idxes = np.searchsorted(hash_idx, hashes)
        if np.any(hash_idx[idxes] != hashes):
            #missing_hashes = hashes[hash_idx[idxes] != hashes]
            raise ValueError('unknown hash')

        idf = idfs[idxes]
        values = (np.log(values).astype(np.float32) + 1) * idf
        values /= np.linalg.norm(values, 2)


        return (idxes.astype(np.int32), values)


class SIFVectorizer:

    def __init__(self):
        self._a_param = 0.001
        self._N = None
        self._sifs = None
        self._hash_func = None
        self._hashes = None

    def build_from_doc_freqs(self, doc_freqs):
        self._sifs = MemmapArray(_doc_freq_to_sif(doc_freqs.doc_freqs_, doc_freqs.corpus_size_, self._a_param))
        self._sifs.to_spark()

        self._hashes = MemmapArray(doc_freqs.hashes_.copy())
        self._hashes.to_spark()

        self._hash_func = deepcopy(doc_freqs.hash_func_)
        self._N = len(self._sifs)
    
    def _hash(self, s):
        return self._hash_func.hash(s)
    
    def out_col_name(self, base):
        return f'sif_vec({base})'
    
    def init(self):
        self._sifs.init()
        self._hashes.init()
        pass

    def vectorize(self, tokens):
        if is_null(tokens):
            return None
        hashes = np.fromiter((self._hash(t) for t in tokens), dtype=np.int64, count=len(tokens))
        hashes, values = np.unique(hashes, return_counts=True)

        return SparseVec(self._N, *_vectorize_sif(self._hashes.values, self._sifs.values, hashes, values))


@nb.njit('float32[:](int64[::1], int64, float32)')
def _doc_freq_to_sif(doc_freq, corpus_size, a_param):
    return ((a_param / ((doc_freq / corpus_size) + a_param))).astype(np.float32)

@nb.njit(
    nb.types.Tuple((
            nb.types.int32[:],
            nb.types.float32[:]
        ))(
            nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
            nb.types.Array(nb.types.float32, 1, 'C', readonly=True),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False),
            nb.types.Array(nb.types.int64, 1, 'C', readonly=False)
    )
)
def _vectorize_sif(hash_idx, sifs, hashes, values):
        # sorted hashes implies idxes are sorted because 
        # self._hashes is also sorted
        idxes = np.searchsorted(hash_idx, hashes)
        if np.any(hash_idx[idxes] != hashes):
            raise ValueError('unknown hash')

        sif = sifs[idxes]
        values = values.astype(np.float32) * sif
        values /= np.linalg.norm(values, 2)

        return (idxes.astype(np.int32), values)
