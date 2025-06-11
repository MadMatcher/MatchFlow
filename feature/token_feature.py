import pandas as pd
import numpy as np
from feature.feature import Feature
from tokenizer.tokenizer import Tokenizer
from utils import is_null, get_logger, type_check
from py_stringmatching import MongeElkan
from abc import abstractmethod
import math

log = get_logger(__name__)

class TokenFeature(Feature):
    def __init__(self, a_attr, b_attr, tokenizer):
        super().__init__(a_attr, b_attr)
        type_check(tokenizer, 'tokenizer', Tokenizer)       
        self._tokenizer = tokenizer
        self._a_toks_col = self._get_token_column(a_attr)
        self._b_toks_col = self._get_token_column(b_attr)

    @abstractmethod
    def sim_func(self, x, y):
        """
        function that takes in two sets of tokens and outputs a float 
        """
        pass

    def _get_input_column(self, base):
        return base 

    def _get_token_column(self, base):
        return self._tokenizer.out_col_name(base)

    def _preprocess_output_column(self, attr):
        return self._get_token_column(attr)

    def _preprocess(self, df, input_col):
        toks = df[input_col].apply(self._tokenizer.tokenize_set)
        toks.name = self._tokenizer.out_col_name(input_col)
        return toks

    def __call__(self, rec, recs):
        s = rec[self._b_toks_col]
        if is_null(s):
            return pd.Series(np.nan, index=recs.index)
        sets = recs[self._a_toks_col]
        return sets.apply(lambda x : self.sim_func(s, x)).astype(np.float64)


def _overlap(x,y):
    if is_null(x) or is_null(y):
        return np.nan
    elif len(x) == 0 or len(y) == 0:
        return 0.0
    elif len(x) < len(y):
        return sum((e in y) for e in x)
    else:
        return sum((e in x) for e in y)

def _jaccard(x,y):
    olap = _overlap(x,y)
    if np.isnan(olap):
        return np.nan
    elif olap == 0:
        return 0.0
    else:
        return olap / (len(x) + len(y) - olap)

def _overlap_coeff(x,y):
    olap = _overlap(x,y)
    if np.isnan(olap):
        return np.nan
    elif olap == 0:
        return 0.0
    else:
        return olap / min(len(x), len(y))


class JaccardFeature(TokenFeature):
    
    def sim_func(self, x, y):
        return _jaccard(x,y)

    def __str__(self):
        return f'jaccard({self._a_toks_col}, {self._b_toks_col})'

class OverlapCoeffFeature(TokenFeature):

    def sim_func(self, x, y):
        return _overlap_coeff(x,y)

    def __str__(self):
        return f'overlap_coeff({self._a_toks_col}, {self._b_toks_col})'


class CosineFeature(TokenFeature):
    def sim_func(self, x, y):
        olap = _overlap(x,y)
        if np.isnan(olap):
            return np.nan
        elif olap == 0:
            return 0.0
        else:
            return olap / math.sqrt(len(x) * len(y))

    def __str__(self):
        return f'cosine({self._a_toks_col}, {self._b_toks_col})'

class MongeElkanFeature(TokenFeature):
    """
    MongeElkan with jaro winkler as the inner sim func
    """
    # default uses jaro
    _me = MongeElkan().get_raw_score
    
    def sim_func(self, x, y):
        if not is_null(y):
            return self._me(x,y)
        else:
            return np.nan

    def __str__(self):
        return f'monge_elkan_jw({self.a_attr}, {self.b_attr})'
