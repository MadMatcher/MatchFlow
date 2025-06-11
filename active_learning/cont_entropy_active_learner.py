import pyspark.sql.functions as F    
from tqdm import tqdm
import traceback
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import pyspark.sql.types as T    
from copy import deepcopy, copy
from utils import persisted, get_logger, repartition_df, type_check
from labeler import Labeler
from ml_model import MLModel, SKLearnModel, convert_to_array, convert_to_vector
from pyspark.ml.functions import vector_to_array, array_to_vector
from queue import PriorityQueue, Queue, Empty
from threading import Thread, Event
import pyspark
from math import ceil
import time
from dataclasses import dataclass, field
from typing import Any

log = get_logger(__name__)
    
    
class ContinuousEntropyActiveLearner:    
    
    def __init__(self, model, labeler, queue_size=50, max_labeled=100000, on_demand_stop=True):    
        """

        Parameters
        ----------
        model : MLModel
            the model to be trained

        labeler : Labeler
            the labeler that will be used to label batches during active learning

        queue_size : int
            the number of examples to queue during active learning

        max_labeled : int 
            the maximum number of examples that will be labeled, only necessary if on_demand_stop is False

        on_demand_stop : bool
            if True, the active learning will stop when the user stops labeling

        """

        self._check_init_args(model, labeler, queue_size, max_labeled, on_demand_stop)

        self._queue_size = queue_size
        self._max_labeled = max_labeled
        self._on_demand_stop = on_demand_stop
        self._min_batch_size = 3
        self._labeler = labeler
        self._model = copy(model)
        self.local_training_fvs_ = None
        self._terminate_if_label_everything = False
    
    def _check_init_args(self, model, labeler, queue_size, max_labeled, on_demand_stop):

        type_check(model, 'model', MLModel)
        type_check(labeler, 'labeler', Labeler)
        type_check(queue_size, 'queue_size', int)
        type_check(max_labeled, 'max_labeled', int)
        if queue_size <= 0 :
            raise ValueError('queue_size must be > 0')
        if max_labeled <= 0 :
            raise ValueError('max_labeled must be > 0')
        type_check(on_demand_stop, 'on_demand_stop', bool)

    def _select_training_vectors(self, fvs, ids):
        return fvs.filter(F.col('_id').isin(ids))

    
    def _get_pos_negative(self, batch):
        pos = batch['label'].sum()
        neg = len(batch) - pos

        return pos, neg
    
    def _label_everything(self, fvs):
        spark = SparkSession.builder.getOrCreate()
        batch = fvs.toPandas()
        batch['label'] = batch[['id1', 'id2']].apply(lambda x: float(self._labeler(*x.values)), axis=1)
        batch['labeled_in_iteration'] = -2
        self.local_training_fvs_ = batch
        training_fvs = spark.createDataFrame(self.local_training_fvs_)

        self._model.train(training_fvs, 'features', 'label')
        return copy(self._model)
    
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

        to_be_label_queue = PriorityQueue(self._queue_size)
        labeled_queue = Queue(self._queue_size * 5)

        stop_training = Event()

        training_thread = Thread(target=self._training_loop, args=(to_be_label_queue, labeled_queue, stop_training, fvs, seeds))
        training_thread.start()
        # run labeler in main thread
        nlabeled = len(seeds)
        #log.info('running training')
        terminate = False
        pos, neg = self._get_pos_negative(seeds.dropna(subset=['label']))
        # if on_demand_stop is True, the active learning will stop when the user stops labeling
        # if on_demand_stop is False, the active learning will run until the max_labeled examples are labeled
        while (not terminate and self._on_demand_stop) or (nlabeled < self._max_labeled and not self._on_demand_stop):
            try:
                example = to_be_label_queue.get(timeout=10).item
            except Empty:
                if not training_thread.is_alive():
                    raise RuntimeError('label queue is empty by training thread is dead, likely due to an exception during training')
            else:
                # example gotten, label and send back to training thread
                example['label'] = float(self._labeler(example['id1'], example['id2']))
                if example['label'] == -1.0:
                    log.info('user stopped labeling, terminating active learning')
                    terminate = True
                elif example['label'] != 2.0:
                    labeled_queue.put(example)
                    pos += 1 if example['label'] == 1.0 else 0
                    neg += 1 if example['label'] == 0.0 else 0
                    log.info(f'total positive = {pos} negative = {neg}')
                    nlabeled += 1
        # signal training thread to stop and wait for termination
        stop_training.set()
        training_thread.join()

        labeled_data = self.local_training_fvs_.dropna(subset=['label']).copy()
        return labeled_data


    def _training_loop(self, to_be_label_queue, labeled_queue, stop_event, fvs, seeds):
        try:
            spark = SparkSession.builder.config("spark.ui.showConsoleProgress", "false").getOrCreate()

            fvs = self._prep_fvs(fvs)

            with persisted(fvs) as fvs:
                n_fvs = fvs.count()
                # just label everything and return 
                if n_fvs <= len(seeds) + self._max_labeled and not self._on_demand_stop:
                    if self._terminate_if_label_everything:
                        log.info('running al to completion would label everything, labeling all fvs and returning')
                        return self._label_everything(fvs)
                    else:
                        log.info('running al to completion would label everything, but self._terminate_if_label_everything is False so AL will still run')

                self.local_training_fvs_ = seeds.copy()
                self.local_training_fvs_.set_index('_id', drop=False, inplace=True)
                # seed feature vectors
                self.local_training_fvs_['labeled_in_iteration'] = -1.0
                schema = spark.createDataFrame(self.local_training_fvs_).schema

                total_pos, total_neg = self._get_pos_negative(self.local_training_fvs_)
                if total_pos == 0 or total_neg == 0:
                    log.error(f'total positive = {total_pos} negative = {total_neg}')
                    raise RuntimeError('both postive and negative vectors are required for training')

                
                i = -1
                while not stop_event.is_set():
                    i += 1
                    # wait for more stuff to be labeled, but check stop event
                    while (self._queue_size - to_be_label_queue.qsize() < self._min_batch_size 
                           and not stop_event.is_set()):
                        time.sleep(1)
                    
                    # If stop event is set, break out of the main loop
                    if stop_event.is_set():
                        break
                    
                    # get new labeled examples
                    new_examples = []
                    while not labeled_queue.empty():
                        new_examples.append(labeled_queue.get())
                    # add new examples if there are any
                    if len(new_examples) != 0:
                        df = pd.DataFrame(new_examples)

                        df['labeled_in_iteration'] = float(i)
                        self.local_training_fvs_.loc[df.index, 'label'] = df['label']
                        self.local_training_fvs_.loc[df.index, 'labeled_in_iteration'] = df['labeled_in_iteration']
                    
                    # train model
                    #log.info('training model')
                    training_fvs = spark.createDataFrame(self.local_training_fvs_, schema=schema)\
                                        .repartition(len(self.local_training_fvs_) // 100 + 1, '_id')\
                                        .persist()

                    self._model.train(
                            training_fvs.dropna(subset=['label']),
                            'features',
                            'label'
                    )

                    cand_fvs = fvs.join(training_fvs, on='_id', how='left_anti')
                    #log.info('selecting and labeling new examples')
                    # number of examples to replenish in the queue
                    batch_size = self._queue_size - to_be_label_queue.qsize()
                    # get next labeled batch
                    # sort by ids to make training consistent 
                    new_labeled_batch = self._model.entropy(cand_fvs, 'features', 'entropy')\
                                            .sort(['entropy', '_id'], ascending=False)\
                                            .limit(batch_size)\
                                            .toPandas()\
                                            .set_index('_id', drop=False)

                    new_labeled_batch['label'] = np.nan
                    self.local_training_fvs_ = pd.concat([self.local_training_fvs_, new_labeled_batch])

                    training_fvs.unpersist()
                    
                    # push examples to the queue to be labeled
                    for i, r in new_labeled_batch.iterrows():
                        ent = r.pop('entropy')
                        to_be_label_queue.put(PQueueItem(ent, r))

                    #total_pos, total_neg = self._get_pos_negative(self.local_training_fvs_.dropna(subset=['label']))

                    #log.info(f'total positive = {total_pos} negative = {total_neg}')
                    
                    if n_fvs <= len(self.local_training_fvs_):
                        log.info('all fvs labeled, terminating active learning')
                        break

                self.local_training_fvs_ = self.local_training_fvs_.dropna(subset=['label'])
                training_fvs = spark.createDataFrame(self.local_training_fvs_)
                # final train model
                self._model.train(training_fvs, 'features', 'label')
        except Exception as e:
            log.error(traceback.format_exc())


        return copy(self._model)


@dataclass(order=True)
class PQueueItem:
    entropy : float
    item : Any=field(compare=False)
    

