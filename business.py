#!/usr/bin/env python
# -*- coding: utf-8 -*-
 

import logging
import sys
import os

import fasttext
import jieba.posseg as pseg
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from config import root_path
from preprocessor import clean, filter_content
import config

tqdm.pandas()

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class Intention(object):
    def __init__(self,
                 data_path=config.train_path,  # Original data path.
                 sku_path=config.ware_path,  # Sku file path.
                 model_path=None,  # Saved model path.
                 kw_path=None,  # Key word file path.
                 model_train_file=config.business_train,  # Path to save training data for intention.
                 model_test_file=config.business_test):  # Path to save test data for intention.
        self.model_path = model_path
        self.data = pd.read_csv(data_path)
        
        if model_path and os.path.exists(model_path):
            self.fast = fasttext.load_model(model_path)
        else:
            self.kw = self.build_keyword(sku_path, to_file=kw_path)
            self.data_process(model_train_file)  # Create
            self.fast = self.train(model_train_file, model_test_file)
            

    def build_keyword(self, sku_path, to_file):
        '''
        @description:build key word set that related business
        @param {type}
        sku_path： JD sku file path
        to_file： key word set saved file
        @return: 关键词list
        '''
        
        logging.info('Building keywords.')
        tokens = []
        # Filtering words according to POS tags.

        tokens = self.data['custom'].dropna().apply(lambda x: [token for token, pos in pseg.cut(x) if pos in ['n', 'vn', 'nz']])
        
        key_words = set([tk for idx, sample in tokens.iteritems() for tk in sample if len(tk) > 1])
        logging.info('Key words built.')
        sku = []
        with open(sku_path, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split('\t')
                sku.extend(line[-1].split('/'))
        key_words |= set(sku)
        logging.info('Sku words merged.')
        if to_file is not None:
            with open(to_file, 'w') as f:
                for i in key_words:
                    f.write(i + '\n')
        return key_words

    def data_process(self, model_data_file):
        '''
        @description: if the sentences has key word,label sentences as "1", else label sentences as "0"
                      process the sentences to the format fasttext require
        @param {type}
        model_data_file： model file saved path
        @return:
        '''
        logging.info('Processing data.')
        self.data['is_business'] = self.data['custom'].progress_apply(
            lambda x: 1 if any(kw in x for kw in self.kw) else 0)
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(self.data.iterrows(),
                                   total=self.data.shape[0]):
                outline = clean(row['custom']) + "\t__label__" + str(
                    int(row['is_business'])) + "\n"
                f.write(outline)

    def train(self, model_data_file, model_test_file):
        '''
        @description: train model
        @param {type}
        model_data_file： train dataset file path
        model_test_file： test dataset file path
        @return: fasttext model
        '''
        logging.info('Training classifier.')
        classifier = fasttext.train_supervised(model_data_file,
                                               label="__label__",
                                               dim=100,
                                               epoch=5,
                                               lr=0.1,
                                               wordNgrams=2,
                                               loss='softmax',
                                               thread=5,
                                               verbose=True)
        self.test(classifier, model_test_file)
        classifier.save_model(self.model_path)
        logging.info('Model saved.')
        return classifier

    def test(self, classifier, model_test_file):
        '''
        @description: test model
        @param {type}
        classifier： model
        model_test_file： test dataset file path
        @return:
        '''
        logging.info('Testing trained model.')
        test = pd.read_csv(os.path.join(root_path, 'data/test.csv')).fillna('')
        test['is_business'] = test['custom'].progress_apply(
            lambda x: 1 if any(kw in x for kw in self.kw) else 0)

        with open(model_test_file, 'w') as f:
            for index, row in tqdm(test.iterrows(), total=test.shape[0]):
                outline = clean(row['custom']) + "\t__label__" + str(
                    int(row['is_business'])) + "\n"
                f.write(outline)
        result = classifier.test(model_test_file)
        # F1 score
        print(result[1] * result[2] * 2 / (result[2] + result[1]))

    def predict(self, text):
        '''
        @description: predict intent
        @param {type}
        text： sentences
        @return:
        '''
        logging.info('Predicting.')
        label, score = self.fast.predict(clean(filter_content(text)))
        return label[0]


if __name__ == "__main__":
    it = Intention(config.train_path,
                   config.ware_path,
                   model_path=config.ft_path,
                   kw_path=config.keyword_path)
    print(it.predict('怎么申请价保呢？'))
    print(it.predict('你好'))