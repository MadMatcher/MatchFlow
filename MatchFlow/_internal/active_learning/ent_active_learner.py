"""
Entropy-based active learning implementation.

This module is part of the internal implementation and should not be imported directly.
Use the public API in the root package instead.
"""

import pyspark.sql.functions as F    
from pyspark.sql import SparkSession
import pandas as pd
import pyspark.sql.types as T    
from copy import deepcopy, copy
from ..utils import persisted, get_logger, repartition_df, type_check, save_training_data_streaming, load_training_data_streaming, adjust_iterations_for_existing_data
from ..labeler import Labeler, WebUILabeler
from ..ml_model import MLModel, SKLearnModel, convert_to_array, convert_to_vector
from pyspark.ml.functions import vector_to_array, array_to_vector
import pyspark
from math import ceil


log = get_logger(__name__)
    
    
class EntropyActiveLearner:    
    
    def __init__(self, model, labeler, batch_size=10, max_iter=50,  parquet_file_path='active-matcher-training-data.parquet'):    
        """

        Parameters
        ----------
        model : MLModel
            the model to be trained

        labeler : Labeler
            the labeler that will be used to label batches during active learning

        batch_size : int
            the number of examples to select per batch

        max_iter : int 
            the maximum number of iterations of active learning

        parquet_file_path : str
            path to save/load training data parquet file

        """


        self._check_init_args(model, labeler, batch_size, max_iter, parquet_file_path)

        self._batch_size = batch_size
        self._labeler = labeler
        self._model = copy(model)
        self._max_iter = max_iter
        self._parquet_file_path = parquet_file_path
        self.local_training_fvs_ = None
        self._terminate_if_label_everything = False
    
    def _check_init_args(self, model, labeler, batch_size=10, max_iter=50, parquet_file_path='active-matcher-training-data.parquet'):    

        type_check(model, 'model', MLModel)
        type_check(labeler, 'labeler', Labeler)
        type_check(batch_size, 'batch_size', int)
        type_check(max_iter, 'max_iter', int)
        if batch_size <= 0 :
            raise ValueError('batch_size must be > 0')
        if max_iter <= 0 :
            raise ValueError('max_iter must be > 0')
        type_check(parquet_file_path, 'parquet_file_path', str)

    def _select_training_vectors(self, fvs, ids):
        return fvs.filter(F.col('_id').isin(ids))

    
    def _get_pos_negative(self, batch):
        pos = batch['label'].sum()
        neg = len(batch) - pos

        return pos, neg
    
    def _label_everything(self, fvs):
        spark = SparkSession.builder.getOrCreate()
        # Derive the schema from the source feature vectors (extended with the
        # columns we add) so createDataFrame never has to infer the
        # feature_vectors type.
        schema = T.StructType(list(fvs.schema.fields) + [
            T.StructField('label', T.DoubleType()),
            T.StructField('labeled_in_iteration', T.DoubleType()),
        ])
        batch = fvs.toPandas()
        batch['label'] = batch[['id1', 'id2']].apply(lambda x: float(self._labeler(*x.values)), axis=1)
        batch['labeled_in_iteration'] = -2.0
        self.local_training_fvs_ = batch
        training_fvs = spark.createDataFrame(self.local_training_fvs_, schema=schema)

        self._model.train(training_fvs, 'feature_vectors', 'label')
        return self.local_training_fvs_
    
    def _prep_fvs(self, fvs):
        # fvs(_id, id1, id2, features)
        fvs = repartition_df(fvs, 25000, '_id')

        return self._model.prep_fvs(fvs)

    def train(self, fvs, seeds):
        """
        run active learning

        Parameters
        ----------
        fvs : pyspark.sql.DataFrame
            the feature vectors for training, schema must contain (_id Any, id1 long, id2 long, features array<double or float>)

        start_ids : List of _id
            the ids of the seed feature vectors for starting active learning, these must be present in `fvs`
        """
        type_check(fvs, 'fvs', pyspark.sql.DataFrame)

        spark = SparkSession.builder.getOrCreate()

        fvs = self._prep_fvs(fvs)

        with persisted(fvs) as fvs:
            n_fvs = fvs.count()

            # Load existing training data if available
            existing_training_data = load_training_data_streaming(self._parquet_file_path, log)
            
            if existing_training_data is not None:
                # Use existing data instead of seeds
                self.local_training_fvs_ = existing_training_data
                log.info(f'Using {len(self.local_training_fvs_)} existing labeled examples')
            else:
                # Start with seeds
                self.local_training_fvs_ = seeds.copy()
                # Save initial seeds
                save_training_data_streaming(self.local_training_fvs_, self._parquet_file_path, log)
            
            # Calculate adjusted iterations based on existing data
            #max_itr = adjust_iterations_for_existing_data(
            #    len(self.local_training_fvs_), n_fvs, self._batch_size, self._max_iter)
            max_itr = self._max_iter
            # just label everything and return 
            if n_fvs <= len(self.local_training_fvs_) + (self._batch_size * max_itr):
                if self._terminate_if_label_everything:
                    log.info('running al to completion would label everything, labeling all fvs and returning')
                    return self._label_everything(fvs)
                else:
                    log.info('running al to completion would label everything, but self._terminate_if_label_everything is False so AL will still run')


            # Train initial model with existing data
            if existing_training_data is not None:
                log.info('Training initial model with existing data')
                initial_training_fvs = spark.createDataFrame(self.local_training_fvs_)\
                                           .repartition(len(self.local_training_fvs_) // 100 + 1, '_id')\
                                           .persist()
                self._model.train(initial_training_fvs, 'feature_vectors', 'label')
                initial_training_fvs.unpersist()
                log.info('Initial model training complete')
            else:
                self.local_training_fvs_ = seeds.copy()
                
            # seed feature vectors
            self.local_training_fvs_['labeled_in_iteration'] = -1
            schema = spark.createDataFrame(self.local_training_fvs_).schema

            total_pos, total_neg = self._get_pos_negative(self.local_training_fvs_)
            if total_pos == 0 or total_neg == 0:
                log.error(f'total positive = {total_pos} negative = {total_neg}')
                raise RuntimeError('both postive and negative vectors are required for training')

            log.info(f'max iter = {max_itr}')
            i = 0
            label = None
            if isinstance(self._labeler, WebUILabeler):
                log.warning("Records are almost ready to be labeled.")

            while i < max_itr and label != -1:
                log.info(f'starting iteration {i}')
                # train model
                log.info('training model')
                training_fvs = spark.createDataFrame(self.local_training_fvs_, schema=schema)\
                                    .repartition(len(self.local_training_fvs_) // 100 + 1, '_id')\
                                    .persist()

                self._model.train(training_fvs, 'feature_vectors', 'label')

                cand_fvs = fvs.join(training_fvs, on='_id', how='left_anti')
                log.info('selecting and labeling new examples')
                # get next labeled batch
                # sort by ids to make training consistent 
                new_labeled_batch = self._model.entropy(cand_fvs, 'feature_vectors', 'entropy')\
                                        .sort(['entropy', '_id'], ascending=False)\
                                        .limit(self._batch_size)\
                                        .drop('entropy')\
                                        .toPandas()

                training_fvs.unpersist()

                labels = []
                labeled_indices = []
                user_stopped = False

                for idx, row in new_labeled_batch.iterrows():
                    label = float(self._labeler(row['id1'], row['id2']))
                    if label == -1.0:
                        user_stopped = True
                        break
                    labels.append(label)
                    labeled_indices.append(idx)
                
                # Only keep the rows that were actually labeled as match or non-match
                new_labeled_batch = new_labeled_batch.loc[labeled_indices].copy()
                new_labeled_batch['label'] = labels
                new_labeled_batch = new_labeled_batch[(new_labeled_batch['label'] == 0.0) | (new_labeled_batch['label'] == 1.0)]
                new_labeled_batch['labeled_in_iteration'] = i

                pos, neg = self._get_pos_negative(new_labeled_batch)
                total_pos += pos
                total_neg += neg

                save_training_data_streaming(new_labeled_batch, self._parquet_file_path, log)

                self.local_training_fvs_ = pd.concat([self.local_training_fvs_, new_labeled_batch], ignore_index=True)
                log.info(f'new batch positive = {pos} negative = {neg}, total positive = {total_pos} negative = {total_neg}')
                
                if n_fvs <= len(self.local_training_fvs_):
                    log.info('all fvs labeled, terminating active learning')
                    break
                if user_stopped:
                    log.info('user stopped labeling, terminating active learning')
                    break
                i += 1

        labeled_data = self.local_training_fvs_.dropna(subset=['label']).copy()
        if isinstance(self._labeler, WebUILabeler):
            log.warning("Active learning is complete.")
        return labeled_data
