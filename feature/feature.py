from abc import abstractmethod, ABC
from functools import partial
import pandas as pd
import numpy as np
from utils import get_logger
from py_stringmatching import Levenshtein, NeedlemanWunsch, SmithWaterman
from utils import type_check
#from active_matcher.fv_generator import BuildCache

log = get_logger(__name__)

class Feature(ABC):
    def __init__(self, a_attr : str, b_attr : str):
        if not isinstance(a_attr, str) or not isinstance(b_attr, str):
            raise TypeError(f'a_attr and b_attr must be strings not {type(a_attr), type(b_attr)}')

        self._a_attr = a_attr
        self._b_attr = b_attr

    def build(self, A, B, cache):
        '''
        Guarenteed to be called before the features preprocessing is done. 
        this method should generate and store all of the metadata required to 
        compute the features over A and B, NOTE B may be None
        '''
        pass

    @classmethod
    def template(cls, **kwargs):
        return partial(cls, **kwargs)

    @property
    def a_attr(self):
        """
        the name of the attribute from table a used to generate this feature
        """
        return self._a_attr

    @property
    def b_attr(self):
        """
        the name of the attribute from table a used to generate this feature
        """
        return self._b_attr

    @abstractmethod
    def __str__(self):
        """
        return a string representation of this feature, this should uniquely identify the feature
        """
        pass

    @abstractmethod
    def __call__(self, A : dict, B : pd.DataFrame) -> pd.Series:
        """
        compute the feature with A for each row in B, both A and B are preprocessed
        """
        pass

    @abstractmethod
    def _preprocess(self, data : pd.DataFrame, input_col : str) -> pd.Series:
        '''
        this method should perform preprocessing for the input_col and 
        return a series with the preprocessing data with name _preprocess_output_column(input_col)
        '''
        pass

    @abstractmethod
    def _preprocess_output_column(self):
        '''
        the name of the column that will be output for preprocessing 
        this features. Return None if there is no preprocessing that needs to be 
        done for this feature. This column name + a row id must unique identify 
        an object in the preprocessing output. For example, jaccard_3gram(a_name, b_name)
        would probably output 3gram_tokens(a_name) for preprocessing table A. Note that 
        If these are name collisions, preprocessing will sliently skip processing data and 
        lead to strange behavior.
        '''
        pass

    def preprocess_output_column(self, for_table_a : bool):
        """
        get the name of the preprocessing output column for table A or B
        """
        if for_table_a:
            return self._preprocess_output_column(self.a_attr)
        else:
            return self._preprocess_output_column(self.b_attr)

    def preprocess(self, data, is_table_a):
        """
        preprocess the data, adding the output column to data
        """
        out_col = self.preprocess_output_column(is_table_a)
        if out_col is None:
            return data

        if is_table_a: 
            if self.preprocess_output_column(is_table_a) not in data:
                c = self._preprocess(data, self.a_attr)
                data[c.name] = c
        else:
            if self.preprocess_output_column(False) not in data:
                c = self._preprocess(data, self.b_attr)
                data[c.name] = c

        return data



class ExactMatchFeature(Feature):
    """
    Case insensitive exact string match
    """
    
    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data

    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        strings = strings.apply(lambda x : x.lower() if pd.notnull(x) else None)
        return strings.eq(s.lower()).astype(np.float64)

    def __str__(self):
        return f'exact_match({self.a_attr}, {self.b_attr})'


class RelDiffFeature(Feature):
    """
    relative difference between two values
    """
        
    def __init__(self, a_attr, b_attr):
        super().__init__(a_attr, b_attr)
        self._a_float_col = self._preprocess_output_column(a_attr)
        self._b_float_col = self._preprocess_output_column(b_attr)

    def _preprocess_output_column(self, attr):
        return f'float({attr})'

    def _preprocess(self, data, input_col):
        floats = data[input_col].apply(lambda x : float(x) if x is not None else None)
        floats.name = self._preprocess_output_column(input_col)
        return floats

    def __call__(self, rec, recs):
        f = rec[self._b_float_col]
        floats = recs[self._a_float_col]

        if pd.isnull(f):
            return pd.Series(np.nan, index=floats.index)

        vals = floats.values.astype(np.float32)

        return pd.Series(np.abs(vals - f) / np.maximum(np.abs(vals), np.abs(f)), index=floats.index).astype(np.float64)

    def __str__(self):
        return f'rel_diff({self.a_attr}, {self.b_attr})'

class EditDistanceFeature(Feature):
    """
    edit distance between two strings, case insensitive
    """
    
    _func = Levenshtein().get_sim_score

    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data

    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        strings = strings.apply(lambda x : str(x).lower() if pd.notnull(x) else None)
        return strings.apply(lambda x : self._func(s, x) if x is not None else np.nan).astype(np.float64)

    def __str__(self):
        return f'edit_distance({self.a_attr}, {self.b_attr})'

class NeedlemanWunschFeature(Feature):
    """
    needleman_wunch between two strings, case insensitive
    """
    
    _func = NeedlemanWunsch().get_raw_score

    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data
    
    def _sim_func(self, x, y):
        div = max(len(x), len(y))
        if div != 0:
            return self._func(x,y) 
        else:
            return 0.0

    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        strings = strings.apply(lambda x : str(x).lower() if pd.notnull(x) else None)
        return strings.apply(lambda x : self._sim_func(s, x) if x is not None else np.nan).astype(np.float64)

    def __str__(self):
        return f'needleman_wunch({self.a_attr}, {self.b_attr})'



class SmithWatermanFeature(Feature):
    """
    smith waterman between two strings, case insensitive
    """
    
    _func = SmithWaterman().get_raw_score

    def _preprocess_output_column(self, for_table_a):
        return None

    def _preprocess(self, data, is_table_a):
        return data

    def _sim_func(self, x, y):
        div = max(len(x), len(y))
        if div != 0:
            return self._func(x,y) 
        else:
            return 0.0


    def __call__(self, rec, recs):
        s = rec[self.b_attr]
        strings = recs[self.a_attr]

        if not isinstance(s, str):
            return pd.Series(np.nan, index=strings.index)

        strings = strings.apply(lambda x : str(x).lower() if pd.notnull(x) else None)
        return strings.apply(lambda x : self._sim_func(s, x) if x is not None else np.nan).astype(np.float64)

    def __str__(self):
        return f'smith_waterman({self.a_attr}, {self.b_attr})'
