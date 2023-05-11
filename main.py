from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import *
from models import S2SPL
from torch.utils.data import Dataset, DataLoader
from dataset import en_de_dataset, collate_fn_padding
from config import args
from lightning.pytorch.loggers import TensorBoardLogger

# from torch.utils.tensorboard import program
import argparse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--tensorboard', type=bool, default=True,
                    help='Using tensorboard')
    script_args = vars(ap.parse_args())

    if script_args['tensorboard']:
        url = launch_tensorboard('tb_logs')
        print(f"Tensorflow listening on {url}")

    # Initial Dataset
    trainset = en_de_dataset(split='train', pair=args['PAIR'])
    valset = en_de_dataset(
        split='valid',
        input_vocab=trainset.input_vocab,
        target_vocab=trainset.target_vocab
    )
    testset = en_de_dataset(
        split='test',
        input_vocab=trainset.input_vocab,
        target_vocab=trainset.target_vocab
    )

    # Dataloader
    train_loader = DataLoader(
        dataset=trainset,
        shuffle=True,
        pin_memory=True,
        batch_size=args['BATCH_SIZE'],
        collate_fn=collate_fn_padding,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=valset,
        shuffle=False,
        pin_memory=True,
        batch_size=1,
        collate_fn=collate_fn_padding,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=testset,
        shuffle=False,
        pin_memory=True,
        batch_size=args['BATCH_SIZE'],
        collate_fn=collate_fn_padding,
        drop_last=True
    )

    # Initialize model parameters
    s2s_model = S2SPL(
        vocab_size_encoder=len(trainset.input_vocab),
        vocab_size_decoder=len(trainset.target_vocab),

        input_vocab=trainset.input_vocab,
        target_vocab=trainset.target_vocab,
        input_id_to_word=trainset.id_to_word_input,
        target_id_to_word=trainset.id_to_word_target,
        args=args
    )

    # Transfer model parameters to check point
    args_str = " ".join([str(k)+str(v) for k, v in args.items()])

    #Saving best model on validation
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="valid_loss",
        mode="min",
        dirpath=f"best/{args_str}",
        filename="-{epoch:02d}-{valid_loss:.2f}-validation",
    )
    # Saving best model on training
    checkpoint_callback_train = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        dirpath=f"best/{args_str}",
        filename="-{epoch:02d}-{train_loss:.2f}-training",
    )

    print_progress = print_example_callback(trainset=trainset, testset=testset)

    tb_logger = TensorBoardLogger("tb_logs/", name=f'{args_str}')
    # Saving config as json

    # Training
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", precision=16, max_epochs=args['EPOCHS'], callbacks=[checkpoint_callback, checkpoint_callback_train, print_progress], logger=tb_logger)
    trainer.fit(s2s_model, train_loader, val_loader)

    save_config_json(args, f"best/{args_str}")


