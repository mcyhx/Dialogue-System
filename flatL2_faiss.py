#!/usr/bin/env python
# -*- coding: utf-8 -*-
 

import logging
import sys
import os
import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import faiss

sys.path.append('..')
import config
from preprocessor import clean


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def wam(sentence, w2v_model):
    '''
    @description:word average model to get sentences vectors
    @param {type}
    sentence: split by space
    w2v_model: word2vec model
    @return:
    '''
    arr = []
    for s in clean(sentence).split():
        if s not in w2v_model.wv.vocab.keys():
            arr.append(np.random.randn(1, 300))
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr), axis=0).reshape(1, -1)


class FlatL2(object):
    def __init__(self,
                 w2v_path,
                 model_path=None,
                 data_path=None,
                ):
        self.w2v_model = KeyedVectors.load(w2v_path)
        self.data = self.load_data(data_path)
        if model_path and os.path.exists(model_path):
            # 加载
            self.index = self.load_flatL2(model_path)
        elif data_path:
            # 训练
            self.index = self.build_flatL2(model_path)
        else:
            logging.error('No existing model and no building data provided.')

    def load_data(self, data_path):
        '''
        @description: read data and get sentences vectors
        @param {type}
        data_path：pairs of data(Q and A) path
        @return: sentences vectors
        '''
        data = pd.read_csv(
            data_path)
        data['custom_vec'] = data['custom'].apply(
            lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(
            lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data

    def evaluate(self, vecs):
        '''
        @description: model evaluation of recall at 1 (return itself)
        @param {type} text: The query.
        @return {type} None
        '''
        logging.info('Evaluating.')
        nq, d = vecs.shape
        t0 = time.time()
        D, I = self.index.search(vecs, 1)
        t1 = time.time()

        missing_rate = (I == -1).sum() / float(nq)
        recall_at_1 = (I == np.arange(nq)).sum() / float(nq)
        print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
            (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))

    def build_flatL2(self, to_file):
        '''
        @description: model training (flatL2 index)
        @param {type}
        to_file： model saved path
        @return:
        '''
        logging.info('Building hnsw index.')
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1, 300)
        vecs = vecs.astype('float32')
        dim = self.w2v_model.vector_size
        
       

        #     Declaring index
 
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0

        flat_config = [cfg]
        resources = [faiss.StandardGpuResources()]
        index = faiss.GpuIndexFlatL2(resources[0], dim, flat_config[0])


        print("add")
        index.verbose = True  # to see progress
        print('xb: ', vecs.shape)

        print('dtype: ', vecs.dtype)
        index.add(vecs)  # add vectors to the index
        print("total: ", index.ntotal)

        print('save index')
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu,to_file)
        
        
         
        return index

    def load_flatL2(self, model_path):
        '''
        @description: load flatL2 model
        @param {type}
        model_path：model saved path
        @return: flatL2 model
        '''
        logging.info(f'Loading flatL2 index from {model_path}.')
        flatL2 = faiss.read_index(model_path)
        return flatL2

    def search(self, text, k=5):
        '''
        @description: search cloest meaning sentences for the specified sentence by flatL2 index
        @param {type}
        text: the specified sentenc
        k: number of cloest meaning sentence return
        @return: DataFrame contianing the customer input, assistance response
                and the distance to the query.
        '''
        logging.info(f'Searching for {text}.')
        test_vec = wam(clean(text), self.w2v_model)
        test_vec = test_vec.astype('float32')
        # vecs is a n2-by-d matrix with query vectors
        #k = 4                          # we want 4 similar vectors
        D, I = self.index.search(test_vec, k)
        print(I)

        return pd.concat(
            (self.data.iloc[I[0]]['custom'].reset_index(),
             self.data.iloc[I[0]]['assistance'].reset_index(drop=True),
             pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])),
            axis=1)


if __name__ == "__main__":
    flatL2 = FlatL2(config.w2v_path,
                config.flatL2_path,
                config.train_path)
    test = '我要转人工'
    print(flatL2.search(test, k=10))
    eval_vecs = np.stack(flatL2.data['custom_vec'].values).reshape(-1, 300)
    eval_vecs = eval_vecs.astype('float32')
    flatL2.evaluate(eval_vecs[:10])
