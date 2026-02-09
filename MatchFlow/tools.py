"""
Public API functions for MatchFlow.

This module provides the main functions that users will interact with.
Implementation details are hidden in the _internal package.
"""

from typing import List, Optional, Callable, Any, Union, Literal, Dict, Type
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import StructType, StructField, DoubleType
from sklearn.base import BaseEstimator
import xxhash
import numpy as np
import pickle
from pathlib import Path
import logging
import sys

from ._internal.ml_model import MLModel, SKLearnModel, SparkMLModel
from ._internal.utils import get_logger, convert_arrays_for_spark, save_training_data_streaming, load_training_data_streaming, check_tables, check_candidates, check_labeled_data, check_gold_data
from ._internal.labeler import Labeler, CLILabeler, GoldLabeler
from ._internal.active_learning.ent_active_learner import EntropyActiveLearner
from ._internal.active_learning.cont_entropy_active_learner import ContinuousEntropyActiveLearner
from ._internal.featurization import (
    create_features,
    get_base_sim_functions,
    get_base_tokenizers,
    get_extra_tokenizers,
    featurize
)

logger = get_logger(__name__)

# Re-export the public functions
__all__ = [
    'create_features',
    'get_base_sim_functions',
    'get_base_tokenizers',
    'get_extra_tokenizers',
    'featurize',
    'down_sample',
    'create_seeds',
    'train_matcher',
    'apply_matcher',
    'label_data',
    'label_pairs',
    'save_features',
    'load_features',
    'save_dataframe',
    'load_dataframe',
    'check_tables',
    'check_candidates',
    'check_labeled_data',
    'check_gold_data'
]


def down_sample(
    fvs: Union[pd.DataFrame, SparkDataFrame],
    percent: float,
    search_id_column: str,
    score_column: str = 'score',
    bucket_size: int = 1_000,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    down sample by score_column to produce percent * fvs.count() rows

    Parameters
    ----------
    fvs : Union[pd.DataFrame, SparkDataFrame]
        the feature vectors to be downsampled
    percent : float
        the portion of the vectors to be output, (0.0, 1.0]
    search_id_column: str
        the name of the column containing unique identifiers for each record
    score_column : str
        the column that scored the vectors, should be positively correlated with the probability of the pair being a match
    bucket_size : int = 1000
        the size of the buckets for partitioning, default 1000
        
    Returns
    -------
    Union[pd.DataFrame, SparkDataFrame]
        the down sampled dataset with percent * fvs.count() rows with the same schema as fvs
    """
    if isinstance(fvs, pd.DataFrame):
        if percent <= 0 or percent > 1.0:
            raise ValueError('percent must be in the range (0.0, 1.0]')
        fvs = fvs.copy()
        total = len(fvs)
        nparts = max(total // bucket_size, 1)
        fvs["_hash_bucket"] = fvs[search_id_column].astype(str).apply(lambda val: xxhash.xxh64(val).intdigest() % nparts)
        fvs["_rank_desc"] = fvs.groupby("_hash_bucket")[score_column] \
            .rank(method="min", ascending=False)

        bucket_sizes = fvs["_hash_bucket"].value_counts().to_dict()
        fvs["_bucket_size"] = fvs["_hash_bucket"].map(bucket_sizes)

        fvs["_percent_rank"] = fvs.apply(
            lambda row: 0.0
            if row["_bucket_size"] == 1
            else (row["_rank_desc"] - 1) / (row["_bucket_size"] - 1),
            axis=1,
        )
        fvs = fvs[fvs["_percent_rank"] <= percent].copy()
        fvs = fvs.drop(columns=["_hash_bucket", "_rank_desc", "_bucket_size", "_percent_rank"])

    elif isinstance(fvs, SparkDataFrame):
        if percent <= 0 or percent > 1.0:
            raise ValueError('percent must be in the range (0.0, 1.0]')

        if bucket_size < 1000:
            raise ValueError('bucket_size must be >= 1000')

        if isinstance(score_column, str):
            score_column = F.col(score_column)

        # temp columns for sampling
        percentile_col = '_PERCENTILE'
        hash_col = '_HASH'

        window = Window().partitionBy(hash_col).orderBy(score_column.desc())
        nparts = max(fvs.count() // bucket_size, 1)
        fvs = fvs.withColumn(hash_col, F.xxhash64(search_id_column) % nparts)\
                        .select('*', F.percent_rank().over(window).alias(percentile_col))\
                        .filter(F.col(percentile_col) <= percent)\
                        .drop(percentile_col, hash_col)
    return fvs


def create_seeds(
    fvs: Union[pd.DataFrame, SparkDataFrame],
    nseeds: int,
    labeler: Labeler,
    score_column: str = 'score',
    parquet_file_path: str = 'active-matcher-training-data.parquet'
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Create labeled seed examples for active learning.
    
    Parameters
    ----------
    fvs : Union[pd.DataFrame, SparkDataFrame]
        DataFrame containing feature vectors with scores
    nseeds : int
        the number of seeds you want to use to train an initial model
    labeler : Labeler
        the labeler object you want to use to assign labels to rows
    score_column : str, default='score'
        the name of the score column in your fvs DataFrame
    parquet_file_path : str, default='active-matcher-training-data.parquet'
        The path to save the labeled data to
    Returns
    -------
    Union[pd.DataFrame, SparkDataFrame]
        A DataFrame with labeled seeds, schema is (previous schema of fvs, `label`) where the values in 
        label are either 0.0 or 1.0
    """
    if nseeds == 0:
        raise ValueError("no seeds would be created")
    
    return_pandas = isinstance(fvs, pd.DataFrame)
    spark = None if return_pandas else SparkSession.builder.getOrCreate()
    
    existing_training_data = load_training_data_streaming(parquet_file_path, logger)

    if existing_training_data is not None:
        logger.info(f'Found {len(existing_training_data)} existing labeled examples')
        # Check if we already have enough seeds
        if len(existing_training_data) >= nseeds:
            logger.info(f'Already have {len(existing_training_data)} labeled examples, returning existing data')
            result_df = existing_training_data.head(nseeds)
            # Convert to Spark DataFrame if input was Spark DataFrame
            if not return_pandas:
                result_df = convert_arrays_for_spark(result_df)
                return spark.createDataFrame(result_df)
            return result_df
        else:
            logger.info(f'Using {len(existing_training_data)} existing examples, need {nseeds - len(existing_training_data)} more')
            seeds = existing_training_data.to_dict('records')
            pos_count = existing_training_data['label'].sum()
            neg_count = len(existing_training_data) - pos_count
    else:
        logger.info(f'No existing labeled examples found, creating new seeds')
        seeds = []
        pos_count = 0
        neg_count = 0

    existing_ids = set(existing_training_data['_id'].tolist()) if existing_training_data is not None else set()
    # gold labeler overrides the score column with 1.0 for gold pairs and 0.0 for non-gold pairs
    if isinstance(labeler, GoldLabeler):
        gold_pairs = labeler._gold
        if isinstance(fvs, pd.DataFrame) and (score_column not in fvs.columns or fvs[score_column].isna().all()):
            is_gold_mask = [tuple(x) in gold_pairs for x in fvs[['id1', 'id2']].to_numpy()]
            fvs[score_column] = np.where(is_gold_mask, 1.0, 0.0)
        elif isinstance(fvs, SparkDataFrame) and score_column not in [col.name for col in fvs.schema.fields]:
            fvs = fvs.withColumn(score_column, 
                F.when(F.struct('id1', 'id2').isin([F.struct(F.lit(p[0]), F.lit(p[1])) for p in gold_pairs]), 1.0)
                .otherwise(0.0))

    if isinstance(fvs, pd.DataFrame):
        fvs = fvs[fvs[score_column].notna()]
        fvs_length = len(fvs)
        if nseeds > len(fvs):
            raise ValueError("number of seeds would exceed the size of the fvs DataFrame")
        maybe_pos = fvs.nlargest(fvs_length//2, score_column).iterrows()
        maybe_neg = fvs.nsmallest(fvs_length//2, score_column).iterrows()
    elif isinstance(fvs, SparkDataFrame):
        if isinstance(score_column, str):
            score_column = F.col(score_column)
        fvs = fvs.filter((~F.isnan(score_column)) & (score_column.isNotNull()))
        fvs_length = fvs.count()
        if nseeds > fvs_length:
            raise ValueError("number of seeds would exceed the size of the fvs DataFrame")
        # lowest scoring vectors
        maybe_neg = fvs.sort(score_column, ascending=True)\
                        .limit(fvs_length//2)\
                        .toPandas()\
                        .iterrows()
        # highest scoring vectors
        maybe_pos = fvs.sort(score_column, ascending=False)\
                        .limit(fvs_length//2)\
                        .toPandas()\
                        .iterrows()
    i = 0
    while pos_count + neg_count < nseeds and i < fvs_length:
        try:
            _, ex = next(maybe_pos) if pos_count <= neg_count else next(maybe_neg)
            # Skip if this _id already exists in seeds
            if ex['_id'] in existing_ids:
                logger.debug(f'Skipping _id={ex["_id"]}, already exists in seeds')
                i += 1
                continue
            label = float(labeler(ex['id1'], ex['id2']))
            if label == -1.0:  # User requested to stop
                break
            elif label == 2.0:  # User marked as unsure
                i += 1
                continue
            elif label == 1.0:  # Positive match
                pos_count += 1
            else:  # label == 0.0, Negative match
                neg_count += 1
            ex['label'] = label
            seeds.append(ex)
            new_seed_df = pd.DataFrame([ex])
            save_training_data_streaming(new_seed_df, parquet_file_path, logger)
            existing_ids.add(ex['_id'])
        except StopIteration:
            print("Ran out of examples before reaching nseeds")
            break
        i += 1
    if not seeds:
        raise RuntimeError("No seeds were labeled before stopping")
    
    print(f"seeds: pos_count = {pos_count} neg_count = {neg_count}")
    if return_pandas:
        return pd.DataFrame(seeds)
    else:
        seeds_clean = []
        for seed in seeds:
            seed_clean = {k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.number) else v for k, v in seed.items()}
            seeds_clean.append(seed_clean)
        return spark.createDataFrame(seeds_clean)


def train_matcher(
    model: MLModel,
    labeled_data: Union[pd.DataFrame, SparkDataFrame],
    feature_col: str = "feature_vectors",
    label_col: str = "label",
) -> MLModel:
    """Train a matcher model on labeled data.
    
    Parameters
    ----------
    model : MLModel
        An MLModel instance to train
    labeled_data : pandas DataFrame
        DataFrame containing the labeled data
    feature_col : str, default="feature_vectors"
        Name of the column containing feature vectors
    label_col : str, default="label"
        Name of the column containing labels
        
    Returns
    -------
    MLModel
        The trained model
    """
    # the users choices for models: they should either give us a pre-trained model, their own custom MLModel
    # or, they should specify the necessary params. We should be returning to them the trained_model. 
    # on apply, we should expect to get a trained model. 
    return model.train(labeled_data, feature_col, label_col)


def apply_matcher(
    model: MLModel,
    df: Union[pd.DataFrame, SparkDataFrame],
    feature_col: str,
    prediction_col: str,
    confidence_col: Optional[str] = None,
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Apply a trained model to make predictions.
    
    Parameters
    ----------
    model : MLModel
        A trained MLModel instance
    df : pandas DataFrame
        The DataFrame to make predictions on
    feature_col : str
        Name of the column containing feature vectors
    prediction_col : str
        Name of the column to store predictions in
    confidence_col : str, optional
        Name of the column to store confidence scores in. If provided, both predictions
        and confidence scores will be computed efficiently in a single pass.
        
    Returns
    -------
    Union[pd.DataFrame, SparkDataFrame]
        The input DataFrame with predictions added (and confidence scores if requested)
    """
    if confidence_col is not None:
        return model.predict_with_confidence(df, feature_col, prediction_col, confidence_col)
    else:
        return model.predict(df, feature_col, prediction_col)


def label_data(
    model: MLModel,
    mode: Literal["batch", "continuous"],
    labeler: Labeler,
    fvs: Union[pd.DataFrame, SparkDataFrame],
    seeds: Optional[Union[pd.DataFrame, SparkDataFrame]] = None,
    parquet_file_path: str = 'active-matcher-training-data.parquet',
    **learner_kwargs
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Generate labeled data using active learning.
    
    Parameters
    ----------
    model : MLModel
        An MLModel instance
    mode : Literal["batch", "continuous"]
        Whether to use batch or continuous active learning
    labeler : Labeler
        A Labeler instance
    fvs : pandas DataFrame
        The data that needs to be labeled
    seeds : Union[pandas DataFrame, SparkDataFrame], optional
        Initial labeled examples to start with
    parquet_file_path : str, default='active-matcher-training-data.parquet'
        The path to save the labeled data to
    **learner_kwargs :
        Additional keyword arguments to pass to the active learner constructor. For batch mode, see EntropyActiveLearner (e.g. batch_size, max_iter). For continuous mode, see ContinuousEntropyActiveLearner (e.g. queue_size, max_labeled, on_demand_stop).
        
    Returns
    -------
    Union[pd.DataFrame, SparkDataFrame]
        DataFrame with ids of potential matches and the corresponding label
    """
    spark = SparkSession.builder.getOrCreate()
        
    # Create model and labeler
    return_pandas = isinstance(fvs, pd.DataFrame)
    if isinstance(fvs, pd.DataFrame):
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        fvs = spark.createDataFrame(fvs)

    if seeds is None:
        seeds = create_seeds(fvs=fvs, nseeds=min(10, fvs.count()), labeler=labeler, score_column='score', parquet_file_path=parquet_file_path if mode == "batch" else None)
    if isinstance(seeds, SparkDataFrame):
        seeds = seeds.toPandas()
    if mode == "batch":
        learner = EntropyActiveLearner(model, labeler, parquet_file_path=parquet_file_path, **learner_kwargs)
    elif mode == "continuous":
        learner = ContinuousEntropyActiveLearner(model, labeler, parquet_file_path=parquet_file_path, **learner_kwargs)
    else:
        raise ValueError(f"mode must be either 'batch' or 'continuous', not {mode}")
    labeled_data = learner.train(fvs, seeds)
    if return_pandas:
        return labeled_data
    else:
        # Convert arrays to PySpark-compatible format
        labeled_data = convert_arrays_for_spark(labeled_data)
        return spark.createDataFrame(labeled_data)


def label_pairs(
        labeler: Labeler,
        pairs: Union[pd.DataFrame, SparkDataFrame]
) -> Union[pd.DataFrame, SparkDataFrame]:
    """Label pairs without active learning.

    Parameters
    ----------
    labeler : Labeler
        A Labeler instance
    pairs : Union[pd.DataFrame, SparkDataFrame]
        The pairs to label

    Returns
    -------
    Union[pd.DataFrame, SparkDataFrame]
        DataFrame with labeled pairs
    """
    if isinstance(pairs, pd.DataFrame):
        label = 0
        labeled_pairs = []
        id1_col, id2_col = pairs.columns[0], pairs.columns[1]
        for _, row in pairs.iterrows():
            label = labeler(row[id1_col], row[id2_col])
            if label == -1.0:   # -1.0 means the user wants to stop labeling
                break
            labeled_pairs.append({
                id1_col: row[id1_col],
                id2_col: row[id2_col],
                'label': label
            })
        labeled_pairs = pd.DataFrame(labeled_pairs)
    elif isinstance(pairs, SparkDataFrame):
        spark = SparkSession.builder.getOrCreate()
        labeled_pairs = []
        id1_col, id2_col = pairs.columns[0], pairs.columns[1]
        for row in pairs.collect():
            label = labeler(row[id1_col], row[id2_col])
            if label == -1.0:   # -1.0 means the user wants to stop labeling
                break
            labeled_pairs.append({
                id1_col: row[id1_col],
                id2_col: row[id2_col],
                'label': float(label)
            })
        if labeled_pairs:
            labeled_pairs = spark.createDataFrame(labeled_pairs)
        else:
            # Create empty DF with the correct schema
            id1_type = pairs.schema[id1_col].dataType
            id2_type = pairs.schema[id2_col].dataType
            schema = StructType([
                StructField(id1_col, id1_type, True),
                StructField(id2_col, id2_type, True),
                StructField('label', DoubleType(), True)
            ])
            labeled_pairs = spark.createDataFrame([], schema)

    return labeled_pairs

def save_features(features, path):
    """
    Save a list of feature objects to disk using pickle serialization.
    
    Parameters
    ----------
    features : List[Callable]
        List of feature objects to save
    path : str
        Path where to save the features file
        
    Returns
    -------
    None
    """
    path = Path(path)
    
    logger.info(f"Saving {len(features)} features to {path}")
    with open(path, 'wb') as f:
        pickle.dump(features, f)
    logger.info(f"Successfully saved features to {path}")


def load_features(path):
    """
    Load a list of feature objects from disk using pickle deserialization.
    
    Parameters
    ----------
    path : str
        Path to the saved features file
        
    Returns
    -------
    List[Callable]
        List of loaded feature objects
    """
    path = Path(path)
    logger.info(f"Loading features from {path}")
    with open(path, 'rb') as f:
        features = pickle.load(f)
    logger.info(f"Successfully loaded {len(features)} features from {path}")
    return features


def save_dataframe(dataframe, path):
    """
    Save a DataFrame to disk, automatically detecting whether it's a pandas or Spark DataFrame.
    
    Parameters
    ----------
    dataframe : Union[pd.DataFrame, pyspark.sql.DataFrame]
        DataFrame to save (pandas or Spark)
    path : str
        Path where to save the DataFrame
        
    Returns
    -------
    None
    """
    path = Path(path)
    if isinstance(dataframe, pd.DataFrame):
        logger.info(f"Saving pandas DataFrame with shape {dataframe.shape} to {path}")
        dataframe.to_parquet(path)
        logger.info(f"Successfully saved pandas DataFrame to {path}")
    elif isinstance(dataframe, SparkDataFrame):
        spark = SparkSession.builder.getOrCreate()
        master = spark.sparkContext.master
        path_str = str(path)
        # if we are on a cluster but without a distributed file system, save as pandas
        if not master.startswith("local") and "://" not in path_str:
            pdf = dataframe.toPandas() 
            logger.info(f"Saving pandas DataFrame with shape {pdf.shape} to {path_str}")
            pdf.to_parquet(path_str)
            logger.info(f"Successfully saved pandas DataFrame to {path_str}")
            return
        logger.info(f"Saving Spark DataFrame to {path}")
        dataframe.write.mode('overwrite').parquet(str(path))
        logger.info(f"Successfully saved Spark DataFrame to {path}")
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(dataframe)}. "
                       f"Expected pandas.DataFrame or pyspark.sql.DataFrame")


def load_dataframe(path, df_type):
    """
    Load a DataFrame from disk based on the specified type.
    
    Parameters
    ----------
    path : str
        Path to the saved DataFrame file
    df_type : str
        Type of DataFrame to load ('pandas' or 'sparkdf')
        
    Returns
    -------
    Union[pd.DataFrame, pyspark.sql.DataFrame]
        Loaded DataFrame
    """    
    path = Path(path)
    logger.info(f"Loading DataFrame from {path} as {df_type}")
    if df_type.lower() == 'pandas':
        dataframe = pd.read_parquet(path)
        logger.info(f"Successfully loaded pandas DataFrame with shape {dataframe.shape} from {path}")
        return dataframe
    elif df_type.lower() == 'sparkdf':
        spark = SparkSession.builder.getOrCreate()
        master = spark.sparkContext.master
        path_str = str(path)
        # if we are on a cluster but without a distributed file system, load as pandas
        if not master.startswith("local") and "://" not in path_str:
            pdf = pd.read_parquet(path_str)
            logger.info(f"Successfully loaded pandas DataFrame with shape {pdf.shape} from {path_str}")
            return spark.createDataFrame(pdf)
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        dataframe = spark.read.parquet(str(path))
        logger.info(f"Successfully loaded Spark DataFrame from {path}")
        return dataframe
    else:
        raise ValueError(f"Unsupported DataFrame type: {df_type}. "
                        f"Expected 'pandas' or 'sparkdf'")
