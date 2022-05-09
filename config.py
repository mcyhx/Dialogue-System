#!/usr/bin/env python
# -*- coding: utf-8 -*-
 

import torch
import os
root_path = os.path.abspath(os.path.dirname(__file__))

train_raw = os.path.join(root_path, 'data/chat.txt')
dev_raw = os.path.join(root_path, 'data/开发集.txt')
test_raw = os.path.join(root_path, 'data/测试集.txt')
ware_path = os.path.join(root_path, 'data/ware.txt')

sep = '[SEP]'

''' Data '''
# main
train_path = os.path.join(root_path, 'data/train_no_blank.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')
# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')


''' Intention '''
# fasttext
ft_path = os.path.join(root_path, "model/intention/fastext")

''' Retrival '''
# Embedding
w2v_path = os.path.join(root_path, "model/retrieval/word2vec")

# flatL2 parameters
 
flatL2_path = os.path.join(root_path, 'model/retrieval/flatL2_index')

# 通用配置
is_cuda = True
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


 




