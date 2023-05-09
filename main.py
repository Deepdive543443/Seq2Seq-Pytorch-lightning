from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import *
from models import S2SPL
from torch.utils.data import Dataset, DataLoader
from dataset import en_de_dataset, collate_fn_padding
from config import args
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == "__main__":
    # Initial Dataset
    trainset = en_de_dataset(split='train')
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
        emb_dim=args['EMB_DIM'],
        encoder_layers=args['ENCODER_LAYERS'],
        bidirection=args['BIDIRECTION'],
        vocab_size_decoder=len(trainset.target_vocab),
        decoder_layers=args['DECODER_LAYERS'],
        attn_head=args['ATTENTION_HEAD'],
        learning_rate=args['LEARNING_RATE'],
        encoder_type=args['ENCODER_TYPE'],
        decoder_type=args['DECODER_TYPE'],

        input_vocab=trainset.input_vocab,
        target_vocab=trainset.target_vocab,
        input_id_to_word=trainset.id_to_word_input,
        target_id_to_word=trainset.id_to_word_target,
    )

    # Transfer model parameters to check point
    args_str = " ".join([str(k)+str(v) for k, v in args.items()])
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="valid_loss",
        mode="min",
        dirpath=f"best/{args_str}",
        filename="-{epoch:02d}-{valid_loss:.2f}",
    )

    tb_logger = TensorBoardLogger("tb_logs/", name=f'{args_str}')
    trainer = Trainer(precision=16, max_epochs=args['EPOCHS'], callbacks=[checkpoint_callback], logger=tb_logger)
    trainer.fit(s2s_model, train_loader, val_loader)


