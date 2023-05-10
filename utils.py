from lightning.pytorch.callbacks import Callback
# https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
import json, os, sys
from torchtext.data.utils import get_tokenizer

en_tokenizer = get_tokenizer('spacy', language='en')
de_tokenizer = get_tokenizer('spacy', language='de')

# def English_token():
#     return get_tokenizer('spacy', language='en')
#
# def German_token():
#     return get_tokenizer('spacy', language='de')


def beam_search():
    pass

def target_2_teach(targets):
    pad_mask = targets == 2
    pass

def save_config_json(args, path):
    with open(os.path.join(path, 'config.json'), "w") as outfile:
        json.dump(args, outfile)

def load_config_json(path):
    with open(path, 'r') as infile:
        obj = json.load(infile)
    return obj