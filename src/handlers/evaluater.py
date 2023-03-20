import torch
import pickle
import os
import sacrebleu
import numpy as np
import torch.nn.functional as F

from rouge_score import rouge_scorer
from tqdm import tqdm 
from types import SimpleNamespace

from .trainer import Trainer
from ..data.handler import DataHandler

from ..utils.torch import set_rand_seed

class Evaluater(Trainer):
    """ Evaluator class- inherits Trainer so has all experiment methods
        class takes care of evaluation and automatic caching of results"""

    def __init__(self, path:str, device:str='cuda'):
        self.exp_path = path
        self.device = device

    def setup_helpers(self):
        # load arguments 
        args = self.load_args('model_args.json')

        # set up attributes 
        super().setup_helpers(args)

        # load model weights
        self.load_model()
            
    #== loading and saving generated texts =========================================================#
    def load_pred_texts(self, dataset:str, mode:str='test'):
        if not self.texts_exist(dataset, mode):
            self.setup_helpers()
            texts = self.generate_texts(dataset, mode)
            self.save_cache_texts(texts, dataset, mode)
        texts = self.load_cached_texts(dataset, mode)
        return texts

    def save_cache_texts(self, texts:dict, dataset:str, mode:str):
        text_cache_path = self.get_text_cache_path(dataset, mode)
        with open(text_cache_path, 'wb') as handle:
            pickle.dump(texts, handle)

    def load_cached_texts(self, dataset:str, mode:str)->dict:
        text_cache_path = self.get_text_cache_path(dataset, mode)
        with open(text_cache_path, 'rb') as handle:
            texts = pickle.load(handle)
        return texts

    def texts_exist(self, dataset:str, mode:str)->bool:
        text_cache_path = self.get_text_cache_path(dataset, mode)
        return os.path.isfile(text_cache_path)

    def get_text_cache_path(self, dataset:str, mode:str)->str:
        eval_name = f'{dataset}_{mode}'
        text_cache_path = os.path.join(self.exp_path, 'eval', f'{eval_name}.pk')
        return text_cache_path

    #== Model probability calculation method ======================================================#
    @torch.no_grad()
    def generate_texts(self, dataset:str, mode:str='test', lim:int=None, num_beams:int=10):
        self.model.eval()
        self.to(self.device)
        set_rand_seed(1)

        eval_data = self.data_handler.prep_split(dataset, mode, lim)
        eval_batches = self.batcher(
            data = eval_data,
            bsz = 1,
            shuffle = False
        )
        output = {}
        for batch in tqdm(eval_batches, total=len(eval_data)):
            ex_id = batch.ex_id[0]

            # Generate free running prediction
            free_output = self.model.generate(
                input_ids = batch.input_ids,
                attention_mask = batch.attention_mask,
                max_length = 256,
                num_beams = num_beams,
                length_penalty = 0.6,
                no_repeat_ngram_size = 4,
                num_return_sequences = 1,
                output_scores = True,
                return_dict_in_generate = True,
            )

            beams = free_output.sequences
            texts = self.data_handler.tokenizer.batch_decode(beams, skip_special_tokens=True)
            
            output[ex_id] = texts[0]
        return output

    #== loading specific examples =================================================================#
    def load_ex(self, dataset:str, mode:str, k:int=0)->SimpleNamespace:
        data = self.data_handler.prep_split(dataset, mode, lim=10) #TEMP
        ex = data[k]
        return ex
    
    def tokenize_ex(self, ex:SimpleNamespace):
        ex = self.data_handler._prep_ids([ex])
        batch = next(self.batcher(ex, bsz=1))
        return batch
    
    #== general eval methods ======================================================================#
    @staticmethod
    def load_label_texts(dataset:str, mode:str='test')->dict:
        eval_data = DataHandler.load_split(dataset, mode)
        labels_dict = {}
        for ex in eval_data:
            labels_dict[str(ex.ex_id)] = ex.label_text
        return labels_dict

    @staticmethod
    def load_split(dataset:str, mode:str='test', lim=None)->dict:
        eval_data = DataHandler.load_split(dataset, mode)
        output_dict = {}
        for ex in eval_data:
            output_dict[ex.ex_id] = ex
        return output_dict

    @staticmethod
    def calculate_bleu(pred_texts:dict, label_texts:dict, display:bool=False):
        assert pred_texts.keys() == label_texts.keys()
        
        # format inputs
        ex_ids = label_texts.keys()
        preds_text_list  = [pred_texts[k] for k in ex_ids]
        labels_text_list = [[label_texts[k] for k in ex_ids]]
        
        # calculate bleu score
        bleu_score = sacrebleu.corpus_bleu(
            preds_text_list,
            labels_text_list,
        ).score

        if display:
            print(f'bleu:    {bleu_score:.3f}')

        return bleu_score    

    @staticmethod
    def calculate_rouge(pred_texts:dict, label_texts:dict, display:bool=False):
        assert pred_texts.keys() == label_texts.keys()

        # set up rouge calculation
        rouge1, rouge2, rougeL = [], [], []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # calculate rouge for each text, and average over all dataset
        for ex_id in label_texts.keys():
            text_scores = scorer.score(pred_texts[ex_id], label_texts[ex_id])
            rouge1.append(text_scores['rouge1'][2])
            rouge2.append(text_scores['rouge2'][2])
            rougeL.append(text_scores['rougeL'][2])
    
        rouge_1 = np.mean(rouge1)
        rouge_2 = np.mean(rouge2)
        rouge_L = np.mean(rougeL)

        if display:
            print(f'rouge_1: {rouge_1:.3f}')
            print(f'rouge_2: {rouge_2:.3f}')
            print(f'rouge_L: {rouge_L:.3f}')

        return rouge1, rouge2, rougeL

