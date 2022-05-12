#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import pandas as pd
import torch
from torch.utils.data import Dataset
import csv


class DataPrecessForSentence(Dataset):
     
    def __init__(self, bert_tokenizer, file, max_char_len=103):
        
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels \
            = self.get_input(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[
            idx], self.labels[idx]

     
    def get_input(self, file):
        """
        preprocess data into the format that can be feed to Bert Model
        inputs:
            dataset     : pandas datafram which has three columns:question1, question2, and label
                          label shows whether the question1 and the question2 have the same meaning. 
                          "1" stands for same meaning and "0" stands for different meaning
            max_seq_len : maxmium sequence length of processed sequence,an integer that is samller or equal to 512
            
        outputs:
            seq         : seq added 'CLS' and'SEP', and padded if length of processed sequence is smaller than 
                          max_seq_len
            seq_mask    : "0" for real tokens and "1" for padded token
                           
            seq_segment : O for the first sentence, 1 for the second sentence if there are two sentences
            labels      : 1 for the positive sample and 0 for negative sample
        """
        df = pd.read_csv(file,
                         sep='\t',
                         header=None,
                         names=['question1', 'question2', 'label'],
                         quoting=csv.QUOTE_NONE
                        )
 

        df["question1"] = df["question1"].apply(lambda x: "".join(x.split()))
        df["question2"] = df["question2"].apply(lambda x: "".join(x.split()))
        labels = pd.to_numeric(df['label'].values,errors = 'coerce').astype('int8')

        
        tokens_seq_1 = list(
            map(self.bert_tokenizer.tokenize, df['question1'].values))
        tokens_seq_2 = list(
            map(self.bert_tokenizer.tokenize, df['question2'].values))
       
        result = list(map(self.trucate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.tensor(seqs).type(
            torch.long), torch.tensor(seq_masks).type(
                torch.long), torch.tensor(seq_segments).type(
                    torch.long), torch.from_numpy(labels).type(torch.int64)

    def trucate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        process sequence into ['CLS',seq,'SEP'] format，pad the sequence if the length of sequence
        is smaller than max_seq_len
        inputs:
            seq_1       : input sentence : question1
            seq_2       : input sentence : question2
            max_seq_len : maxmium sequence length of processed sequence

        outputs:
            seq         : processed sequence into ['CLS',seq,'SEP'] format with padded if the length of sequence
                          is smaller than max_seq_len。
            seq_mask    : "1" for real tokens and "0" for padded token
            seq_segment :  O for the first sentence, 1 for the second sentence if there are two sentences

        """
     
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3) // 2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3) // 2]
        # add 'CLS' and 'SEP'
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) +
                             2) + [1] * (len(tokens_seq_2) + 1)
        # convert tokens into ids
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # pad sequence 
        padding = [0] * (self.max_seq_len - len(seq))
        # create seq_mask
        seq_mask = [1] * len(seq) + padding
        # create seq_segment
        seq_segment = seq_segment + padding
        # concate sequence
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment
