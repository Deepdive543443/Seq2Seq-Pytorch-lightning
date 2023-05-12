from lightning.pytorch.callbacks import Callback
# https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
import json, os, sys
from torchtext.data.utils import get_tokenizer
from tensorboard import program
from pytorch_lightning import callbacks




en_tokenizer = get_tokenizer('spacy', language='en')
de_tokenizer = get_tokenizer('spacy', language='de')

class print_example_callback(Callback):
    def __init__(self, trainset, testset):
        super(print_example_callback, self).__init__()
        self.trainset = trainset
        self.testset = testset


    def on_train_epoch_end(self, trainer, pl_module):
        print('\nTraining result')
        input_pair, target_pair = self.trainset.random_pairs()
        translated_sentences, bleu_score = pl_module.translate(input_pair, target_pair)
        print(f"Input: {input_pair}\nTarget: {target_pair}\nTranslate: {translated_sentences}\nBLEU: {bleu_score}\n")

        print('Testing result')
        input_pair, target_pair = self.testset.random_pairs()
        translated_sentences, bleu_score = pl_module.translate(input_pair, target_pair)
        print(f"Input: {input_pair}\nTarget: {target_pair}\nTranslate: {translated_sentences}\nBLEU: {bleu_score}")



def launch_tensorboard(tracking_address):
    # https://stackoverflow.com/a/55708102
    # tb will run in background but it will
    # be stopped once the main process is stopped.
    try:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tracking_address, '--port', '8008'])
        url = tb.launch()
        if url.endswith("/"):
            url = url[:-1]

        return url
    except Exception:
        return None




def beam_search():
    pass


def save_config_json(args, path):
    with open(os.path.join(path, 'config.json'), "w") as outfile:
        json.dump(args, outfile)

def load_config_json(path):
    with open(path, 'r') as infile:
        obj = json.load(infile)
    return obj