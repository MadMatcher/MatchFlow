from typing import Literal, Optional, Union
import pandas as pd
import labeler
from active_learning import EntropyActiveLearner, ContinuousEntropyActiveLearner
from ml_model import MLModel, SKLearnModel, SparkMLModel
import xxhash
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator


def down_sample(
    fvs: pd.DataFrame,
    percent: float,
    search_id_column: str,
    score_column: str = 'score',
    bucket_size: int = 1_000,
) -> pd.DataFrame:
    """
    down sample by score_column to produce percent * fvs.count() rows

    Parameters
    ----------
    fvs : pandas DataFrame
        the feature vectors to be downsampled
    percent : float
        the portion fo the vectors to be output, (0.0, 1.0]
    score_column : str
        the column that scored the vectors, should be positively correlated with the probability of the pair being a match
    CHECK: should not need the search_id_column or bucket size because we don't need to hash into buckets for non-spark

    Returns
    -------
    pandas DataFrame
        the down sampled dataset with percent * fvs.count() rows with the same schema as fvs
    """
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

    return fvs


def select_seeds(
    fvs: pd.DataFrame,
    nseeds: int,
    labeler: labeler,
    score_column: str = 'score'
) -> pd.DataFrame:
    """
    select seeds to train a model

    Parameters
    ----------
    fvs : pandas DataFrame
        the DataFrame with feature vectors that is your training data
    nseeds : int
        the number of seeds you want to use to train an initial model
    labeler : Labeler
        the labeler object you want to use to assign labels to rows
    score_column : str
        the name of the score column in your fvs DataFrame

    Returns
    -------
    pandas DataFrame
        A DataFrame with labeled seeds, schema is (previous schema of fvs, `label`) where the values in label are either 0.0 or 1.0 
    """
    fvs = fvs[fvs[score_column].notna()].copy()
    maybe_pos = fvs.nlargest(nseeds, score_column).iterrows()
    maybe_neg = fvs.nsmallest(nseeds, score_column).iterrows()

    pos_count = 0
    neg_count = 0
    seeds = []
    i = 0
    while pos_count + neg_count < nseeds and i < nseeds * 2:
        try:
            _, ex = next(maybe_pos) if pos_count <= neg_count else next(maybe_neg)
            label = float(labeler(ex['id1'], ex['id2']))
            
            if label == -1.0:  # User requested to stop
                break
            elif label == 2.0:  # User marked as unsure
                continue
            elif label == 1.0:  # Positive match
                pos_count += 1
            else:  # label == 0.0, Negative match
                neg_count += 1

            ex['label'] = label
            seeds.append(ex)
        except StopIteration:
            print("Ran out of examples before reaching nseeds")
            break
        i += 1
    if not seeds:
        raise RuntimeError("No seeds were labeled before stopping")
    
    print(f"seeds: pos_count = {pos_count} neg_count = {neg_count}")
    return pd.DataFrame(seeds)


def train_matcher(
    model: MLModel,
    df: pd.DataFrame,
    feature_col: str,
    label_col: str
) -> MLModel:
    """
    train a matcher with labeled data
    
    Parameters
    ----------
    model : MLModel
        the model object that you want to train
    df : pandas DataFrame
        the DataFrame which will be used to train the data with at least a feature column and a label column
    feature_col: str
        the name of the feature column in df
    label_col: str
        the name of the label column in df

    Returns
    -------
    MLModel
        the trained MLModel
    """
    trained_matcher = model.train(df, vector_col=feature_col, label_column=label_col)
    return trained_matcher


def apply_matcher(
    model: Union[MLModel, BaseEstimator, object],  # Any trained model (MLModel, sklearn, or spark)
    df: pd.DataFrame,
    feature_col: str,
    output_col: str
) -> pd.DataFrame:
    """
    applies a trained model on unlabeled data

    Parameters
    ----------
    model : MLModel, sklearn model, or spark model
        The trained model to apply to the data. Can be:
        - An MLModel instance (SKLearnModel or SparkMLModel)
        - A trained sklearn model (any BaseEstimator)
        - A trained spark model (any model with transform method)
    df : pandas DataFrame
        the DataFrame that you want to apply the model to
    feature_col : str
        the column that will be used to make a prediction
    output_col : str
        the column that will store the predicted label from the model

    Returns
    -------
    pandas DataFrame
        the DataFrame with a prediction column holding the predictions
    """
    # If model is already an MLModel instance, use it directly
    if isinstance(model, MLModel):
        return model.predict(df, feature_col, output_col)
    
    # Otherwise wrap it in the appropriate MLModel class
    if hasattr(model, 'transform'):  # Spark model
        from ml_model import SparkMLModel
        return SparkMLModel(model).predict(df, feature_col, output_col)
    else:  # sklearn model
        from ml_model import SKLearnModel
        return SKLearnModel(model).predict(df, feature_col, output_col)


def label_data(
    model: MLModel,
    mode: Literal["batch", "continuous"],
    labeler: labeler,
    fvs: pd.DataFrame,
    seeds: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    generate labeled data from unlabeled data using Batch Active Learning or Continuious Active Learning

    Parameters
    ----------
    model : MLModel
        the model object that will be used for training and applying to the data
    mode : Literal["batch", "continuous"]
        batch or continuous to determine which type of active learning to use
    labeler : Labeler
        the labeler object that will label data during active learning
    fvs : pandas DataFrame
        the data that needs to be labeled
        
    Returns
    -------
    pandas DataFrame
        DataFrame with ids of potential matches and the corresponding label
    """
    spark = SparkSession.builder.getOrCreate()
    if seeds is None:
        seeds = select_seeds(fvs=fvs, nseeds=10, labeler=labeler, score_column='score')
    if mode == "batch":
        learner = EntropyActiveLearner(model, labeler)
    elif mode == "continuous":
        learner = ContinuousEntropyActiveLearner(model, labeler)
    
    if isinstance(fvs, pd.DataFrame):
        fvs = spark.createDataFrame(fvs)
    if isinstance(fvs, pd.DataFrame):
        seeds = spark.createDataFrame(seeds)
    labeled_data = learner.train(fvs, seeds)
    return labeled_data
