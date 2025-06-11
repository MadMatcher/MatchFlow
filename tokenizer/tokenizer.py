from abc import abstractmethod, ABC
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Column
from typing import Iterator
import re
import numpy as np



class Tokenizer(ABC):

    def __str__(self):
        return self.NAME
    
    def tokenize_spark(self, input_col : Column):
        '''
        return a column expression that gives the same output 
        as the tokenize method. required for effeciency when building metadata for 
        certain methods
        '''
        # spark treats whitespace differently than str.split
        # so make a udf to keep tokenization consistent
        @F.pandas_udf(T.ArrayType(T.StringType()))
        def t(itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
            for s in itr:
                yield s.apply(self.tokenize)

        return t(input_col)
    
    @abstractmethod
    def tokenize(self, s):
        '''
        convert the string into a BAG of tokens (tokens should not be deduped)
        '''
        pass

    def out_col_name(self, input_col):
        '''
        the name of the output column from the tokenizer 
        e.g. for a 3gram tokenizer, the tokens from the name columns could 
        be "3gram(name)"
        '''
        return f'{str(self)}({input_col})'
    
    def tokenize_set(self, s):
        '''
        tokenize the string and return a set or None if the tokenize returns None
        '''
        r = self.tokenize(s)
        return set(r) if r is not None else None

    def __eq__(self, o):
        return isinstance(o, type(self)) and self.NAME == o.NAME

class StrippedWhiteSpaceTokenizer(Tokenizer):
    WHITESPACE_NORM = re.compile('[\s]+')
    RE = re.compile('[^a-z0-9 ]+')
    NAME='stripped_whitespace_tokens'
    def __init__(self):
        pass

    def tokenize(self, s):
        if isinstance(s, str):
            s = self.WHITESPACE_NORM.sub(' ', s).lower()
            s = self.RE.sub('', s)
            return s.split()
        else:
            return None

class ShingleTokenizer(Tokenizer):
    base_tokenize = StrippedWhiteSpaceTokenizer().tokenize

    def __init__(self, n):
        self._n = n
        self.NAME = f'{self._n}shingle_tokens'
    
    def tokenize(self, s : str) -> list:
        single_toks = self.base_tokenize(s)
        if single_toks is None:
            return None

        if len(single_toks) < self._n:
            return []

        offsets = [0] + np.cumsum(list(map(len, single_toks))).tolist()
        slices = zip(offsets[:len(single_toks) - self._n], offsets[self._n:])
        combined = ''.join(single_toks)
        return [combined[s:e] for s,e in slices]

class WhiteSpaceTokenizer(Tokenizer):
    NAME='whitespace_tokens'
    def __init__(self):
        pass

    def tokenize(self, s):
        return s.lower().split() if isinstance(s, str) else None


class NumericTokenizer(Tokenizer):

    NAME = 'num_tokens'
    def __init__(self):
        self._re = re.compile('[0-9]+')

    def tokenize(self, s):
        return self._re.findall(s) if isinstance(s, str) else None

class AlphaNumericTokenizer(Tokenizer):
    # TODO drop short tokens and stop words?
    # stopword removal didn't improve accuracy
    #STOP_WORDS = set(stopwords.words('english'))
    NAME = 'alnum_tokens'
    def __init__(self):
        self._re = re.compile('[a-z0-9]+')


    def tokenize(self, s):
        if not isinstance(s, str):
            return None
        else:
            return self._re.findall(s.lower())


class QGramTokenizer(Tokenizer):

    def __init__(self, n):
        self._q = n
        self.NAME = f'{self._q}gram_tokens'

    def tokenize(self, s : str) -> list:
        if not isinstance(s, str):
            return None
        if len(s) < self._q:
            return []
        s = s.lower()
        # TODO can this be optimized?
        return [s[i:i+self._q] for i in range(len(s) - self._q + 1)]

class StrippedQGramTokenizer(Tokenizer):
    
    RE = re.compile('\\W+')
    def __init__(self, n):
        self._q = n
        self.NAME = f'stripped_{self._q}gram_tokens'
    
    def _preproc(self, s : str) -> str:
        # strip all non-word chars
        return self.RE.sub('', s)

    def tokenize(self, s : str) -> list:
        if not isinstance(s, str):
            return None

        s = self._preproc(s).lower()
        if len(s) < self._q:
            return []
        # TODO can this be optimized?
        return [s[i:i+self._q] for i in range(len(s) - self._q + 1)]

