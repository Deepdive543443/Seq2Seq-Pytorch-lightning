from lightning.pytorch.callbacks import Callback, ProgressBar, TQDMProgressBar
# https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
import json, os, sys
from torchtext.data.utils import get_tokenizer
from tensorboard import program
output_stream = sys.stdout

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
        print(f"Input: {input_pair}\nTarget: {target_pair}\nTranslate: {translated_sentences}\nBLEU: {bleu_score}\n")

class KstyleBar(ProgressBar):
    def __init__(self, bar_length, args, train_stat = ['v_num'], val_stat = ['v_num'], test_stat = ['v_num']):
        super().__init__()
        self.bar_length = int(bar_length)
        self.epoch_format = args['EPOCHS']

        # Provide the stat you want to track in train, val, and test progress bar
        self.train_stat = train_stat
        self.val_stat = val_stat
        self.test_stat = test_stat

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        #
        epoch_info = 'Epoch[{current_epoch:4.0f}/{max_epoch:4.0f}] '.format(
            current_epoch = trainer.current_epoch, max_epoch = trainer.max_epochs
        )

        # Show the percentage of progress
        percent = (batch_idx / (trainer.num_training_batches - 1))
        percent_info = "{:6.2f}% ".format(percent * 100)

        # Shows the batch index of progress
        batch_idx_info = '{batch_idx:4.0f}/{num_training_batches:4.0f} '.format(
            batch_idx=batch_idx + 1, num_training_batches=trainer.num_training_batches
        )

        bar = list('=' * int(percent * self.bar_length) + '>' + '-' * (self.bar_length - int(percent * self.bar_length)))
        bar[0], bar[-1] = '[', ']'
        bar = ''.join(bar)


        # Loss info
        loss_info = ''
        for stat in self.train_stat:
            try:
                loss_info += '{stat_name}: {stat_value:3.4f}'.format(
                    stat_name=stat, stat_value=self.get_metrics(trainer, pl_module)[stat]
                )
            except:
                None

        print("\r" + epoch_info + percent_info + bar + batch_idx_info + loss_info, end='\r', flush=True)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print()


    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ):
        percent = (batch_idx / (trainer.num_val_batches[0] - 1))
        percent_info = "{:6.2f}% ".format(percent * 100)
        batch_idx_info = '{batch_idx:4.0f}/{num_training_batches:4.0f} '.format(
            batch_idx=batch_idx + 1, num_training_batches=trainer.num_val_batches[0]
        )

        bar = list('=' * int(percent * self.bar_length) + '>' + '-' * (self.bar_length - int(percent * self.bar_length)))
        bar[0], bar[-1] = '[', ']'
        bar = ''.join(bar)

        # Loss info
        loss_info = ''
        for stat in self.val_stat:
            try:
                loss_info += '{stat_name}: {stat_value:3.4f}'.format(
                    stat_name=stat, stat_value=self.get_metrics(trainer, pl_module)[stat]
                )
            except:
                None

        print("\r" +'Evaluating... ' + percent_info + bar + batch_idx_info + loss_info, end='\r', flush=True)


# class MyProgressBar(TQDMProgressBar):
#     def init_validation_tqdm(self):
#         bar = super().init_validation_tqdm()
#         if not sys.stdout.isatty():
#             bar.disable = True
#         return bar
#
#     def init_predict_tqdm(self):
#         bar = super().init_predict_tqdm()
#         if not sys.stdout.isatty():
#             bar.disable = True
#         return bar
#
#     def init_test_tqdm(self):
#         bar = super().init_test_tqdm()
#         if not sys.stdout.isatty():
#             bar.disable = True
#         return bar


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

if __name__ == "__main__":
    for i in range(0, 999999):
        percent = ( i / 999999) * 100
        print(f'\r{percent:.01f}', end='\r', flush=True)
    print(f'{100:.01f} percent complete')
        # output_stream.write(f'{percent:.01f} percent complete', )
        # output_stream.flush()