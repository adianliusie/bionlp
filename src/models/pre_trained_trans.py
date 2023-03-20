from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from transformers import AutoTokenizer

SEQ2SEQ_TRANS = ['t5-small', 't5-base', 't5-large', 't5-xl']
def load_seq2seq_transformer(system:str)->PreTrainedModel:
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 't5-small': trans_model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
    elif system == 't5-base':  trans_model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    elif system == 't5-large': trans_model = T5ForConditionalGeneration.from_pretrained("t5-large", return_dict=True)
    elif system == 't5-xl':    trans_model = T5ForConditionalGeneration.from_pretrained("t5-3b", return_dict=True) 
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model

LM_TRANS = ['gpt-neo', 'opt-350m']
def load_LM_transformer(system:str)->PreTrainedModel:
    """ downloads and returns the relevant OPT transformer from huggingface """
    if   system == 'gpt-neo':  trans_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    #if   system == 'opt-350m': trans_model = OPTModel.from_pretrained("facebook/opt-350m")
    return trans_model

def load_tokenizer(system:str)->PreTrainedTokenizer:
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 't5-small':   tokenizer = T5TokenizerFast.from_pretrained("t5-small", model_max_length=512)
    elif system == 't5-base':  tokenizer = T5TokenizerFast.from_pretrained("t5-base", model_max_length=512)
    elif system == 't5-large': tokenizer = T5TokenizerFast.from_pretrained("t5-large", model_max_length=512)
    elif system == 't5-xl':    tokenizer = T5TokenizerFast.from_pretrained("t5-3b", model_max_length=512)
    elif system == 'gpt-neo':  tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    elif system == 'opt-350m': tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return tokenizer
