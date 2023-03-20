import random
import re
import numpy as np
import pandas as pd

from tqdm import tqdm 
from copy import deepcopy
from typing import List, Dict, Tuple, TypedDict
from datasets import load_dataset
from functools import lru_cache

class Seq2seqText(TypedDict):
    """Output example formatting (only here for documentation)"""
    input_text : str
    label_text : int

#== Main data loading function =========================================================================# 
def load_bionlp_data(fold_num:int=0)->Tuple[List[Seq2seqText], List[Seq2seqText], List[Seq2seqText]]:
    # load the csv bionlp data
    path = '/home/alta/summary/BioNLP2023/bionlp2023-1a-train-og.csv'
    df = pd.read_csv(path)
    
    # process data from csv to list of dictionaries
    data = []
    for k, (index, row) in enumerate(df.iterrows()):
        row_info = {
            'ex_id': str(k), 
            'file_id': row['File ID'],
            'objective': row['Objective Sections'],
            'subjective': row['Subjective Sections'],
            'assessment': row['Assessment'],
            'summary': row['Summary'],
            'label_text': row['Summary']
        }
        data.append(row_info)
    
    print('Strange Bug? Fix it when you can')
    
    # split data into train and dev split (based on fold)
    if fold_num is not None:
        train_ids, test_ids = load_fold_ids(fold_num)
        train = [data[i] for i in train_ids if data[i]['summary'] is not np.nan]
        test  = [data[i] for i in test_ids if data[i]['summary'] is not np.nan]
        dev   = test
    else:
        data = [i for i in data if i['summary'] is not np.nan]
        train, dev, test = data, data, data
        
    return train, dev, test

def load_fold_ids(fold_num:int=3):
    # load the saved ids
    splits_ids_path = '/home/alta/relevance/vr311/bionlp/split.npy'
    fold_groups = np.load(splits_ids_path)
        
    # set the test split to the fold given, and the rest as train
    test_ids = fold_groups[fold_num]         
    train_ids = [i for group in fold_groups for i in group if i not in test_ids]
    return train_ids, test_ids

#== Util functions for processing data sets =======================================================#
def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random_seeded = random.Random(1)
    random_seeded.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

def _rename_keys(train:list, dev:list, test:list, old_key:str, new_key:str):
    train = [_rename_key(ex, old_key, new_key) for ex in train]
    dev   = [_rename_key(ex, old_key, new_key) for ex in dev]
    test  = [_rename_key(ex, old_key, new_key) for ex in test]
    return train, dev, test
