import pandas as pd
from tokenizer.vectorizer import TFIDFVectorizer, SIFVectorizer
from feature.token_feature import TokenFeature
import pyspark.sql.functions as F
import numpy as np
from utils import get_logger
from utils import PerfectHashFunction
from threading import Lock

log = get_logger(__name__)

class DocFreqBuilder:

    def __init__(self, a_attr, b_attr, tokenizer):
        self.tokenizer = tokenizer
        self.a_attr = a_attr
        self.b_attr = b_attr 
        self.doc_freqs_ = None
        self.hashes_ = None
        self.hash_func_ = None
        self.corpus_size_ = None
        self._built = False
        self._lock = Lock()
    
    def __eq__(self, o):
        return isinstance(o, DocFreqBuilder) and\
                self.a_attr == o.a_attr and\
                self.b_attr == o.b_attr and\
                self.tokenizer == o.tokenizer
            

    def build(self, A, B):
        with self._lock:
            if not self._built:
                df = A.select(F.col(self.a_attr).alias('str'))
                self.corpus_size_ = A.count()
                if B is not None:
                    df = df.unionAll(B.select(F.col(self.b_attr).alias('str')))
                    self.corpus_size_ += B.count()
                doc_freqs = df.select(self.tokenizer.tokenize_spark('str').alias('tokens'))\
                                 .filter(F.col('tokens').isNotNull())\
                                    .select(F.explode(F.array_distinct('tokens')).alias('tok'))\
                                    .groupby('tok')\
                                    .count()\
                                    .toPandas()
                self.hash_func_, hashes = PerfectHashFunction.create_for_keys(doc_freqs['tok'].values)
                srt = hashes.argsort()
                self.hashes_ = hashes[srt]
                self.doc_freqs_ = doc_freqs['count'].values[srt]
                self._built = True

class TFIDFFeature(TokenFeature):

    def __init__(self, a_attr, b_attr, tokenizer):
        super().__init__(a_attr, b_attr, tokenizer)
        self.vectorizer = TFIDFVectorizer()
        self._a_vec_col = self._get_vector_column(self.a_attr)
        self._b_vec_col = self._get_vector_column(self.b_attr)
    

    def sim_func(self, x,y):
        pass
    
    def __str__(self):
        return f'tf_idf_{str(self._tokenizer)}({self.a_attr}, {self.b_attr})'
    
    def _get_vector_column(self, input_col):
        return self.vectorizer.out_col_name(self._get_token_column(input_col))

    def _preprocess_output_column(self, attr):
        return self._get_vector_column(attr)
    

    def build(self, A, B, cache):
        doc_freqs = DocFreqBuilder(self.a_attr, self.b_attr, self._tokenizer)
        doc_freqs = cache.add_or_get(doc_freqs)
        doc_freqs.build(A, B)
        self.vectorizer.build_from_doc_freqs(doc_freqs)

    def _preprocess(self, data, input_col):
        self.vectorizer.init()
        # call tokenizer to produce tokens
        toks = data[input_col].apply(self._tokenizer.tokenize)
        vecs = toks.apply(self.vectorizer.vectorize)
        vecs.name = self._get_vector_column(input_col)

        return vecs

    def __call__(self, rec, recs):
        vec = rec[self._b_vec_col]
        vecs = recs[self._a_vec_col]
        if vec is None:
            return pd.Series(np.nan, index=vecs.index)
        return vecs.apply(lambda x : vec.dot(x) if x is not None else np.nan).astype(np.float64)

class SIFFeature(TFIDFFeature):
    def __init__(self, a_attr, b_attr, tokenizer):
        super().__init__(a_attr, b_attr, tokenizer)
        self.vectorizer = SIFVectorizer()
        self._a_vec_col = self._get_vector_column(self.a_attr)
        self._b_vec_col = self._get_vector_column(self.b_attr)
    
    def __str__(self):
        return f'sif_{str(self._tokenizer)}({self.a_attr}, {self.b_attr})'

