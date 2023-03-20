import random
import os
import re

from typing import List
from types import SimpleNamespace
from tqdm import tqdm
from copy import deepcopy
from functools import lru_cache

from .load_bionlp import load_bionlp_data
from ..models.pre_trained_trans import load_tokenizer
from ..utils.general import get_base_dir, save_pickle, load_pickle

BASE_PATH = get_base_dir()
BASE_CACHE_PATH = f"{BASE_PATH}/tokenize-cache/"

#== Main DataHandler class ========================================================================#
class DataHandler:
    def __init__(self, trans_name:str, formatting:str='{A}'):
        self.trans_name = trans_name
        self.tokenizer = load_tokenizer(trans_name)
        self.formatting = formatting
        
    #== Data processing (i.e. tokenizing text) ====================================================#
    @lru_cache (maxsize=10)
    def prep_split(self, data_name:str, mode:str, lim=None):
        split = self.load_split(data_name=data_name, mode=mode, lim=lim)
        data = self._prep_ids(split)
        return data

    @lru_cache(maxsize=10)
    def prep_data(self, data_name:str, lim=None):
        train, dev, test = self.load_data(data_name=data_name, lim=lim)
        train, dev, test = [self._prep_ids(split) for split in [train, dev, test]]
        return train, dev, test
    
    def _prep_ids(self, split_data:List[SimpleNamespace]):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            input_text = self._prep_text(ex)
            input_ids = self.tokenizer(input_text).input_ids
            label_ids = self.tokenizer(ex.label_text).input_ids
            ex.input_text = input_text
            ex.input_ids = input_ids
            ex.label_ids = label_ids            
        return split_data

    def _prep_text(self, ex):
        template = self.formatting
        template = template.replace('{O}', str(ex.objective))
        template = template.replace('{S}', str(ex.subjective))
        template = template.replace('{A}', str(ex.assessment))
        return template
        
    #== Data loading utils ========================================================================#
    @staticmethod
    @lru_cache(maxsize=10)
    def load_data(data_name:str, lim=None):
        if 'bionlp' in data_name:
            if '-' in data_name:
                _, fold_num = data_name.split('-')
                train, dev, test = load_bionlp_data(fold_num=int(fold_num))
                train, dev, test = to_namespace(train, dev, test)
            else:
                train, dev, test = load_bionlp_data(fold_num=None)
                train, dev, test = to_namespace(train, dev, test)

        else:
            raise ValueError('invalid dataset name given: ', data_name)
            
        if lim:
            train = rand_select(train, lim)
            dev   = rand_select(dev, lim)
            test  = rand_select(test, lim)    
        return train, dev, test
    
    @classmethod
    @lru_cache(maxsize=10)
    def load_split(cls, data_name:str, mode:str, lim=None):
        split_index = {'train':0, 'dev':1, 'test':2}        
        data = cls.load_data(data_name, lim)[split_index[mode]]
        return data

#== Misc utils functions ============================================================================#
def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]

def to_namespace(*args:List):
    def _to_namespace(data:List[dict])->List[SimpleNamespace]:
        if 'ex_id' in data[0]:
            return [SimpleNamespace(**ex) for k, ex in enumerate(data)]
        else:
            return [SimpleNamespace(ex_id=k, **ex) for k, ex in enumerate(data)]

    output = [_to_namespace(split) for split in args]
    return output if len(args)>1 else output[0]

    