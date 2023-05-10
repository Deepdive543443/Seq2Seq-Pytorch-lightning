import random

import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
import re


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_side, layers, dropout, bidirection = False, rnn_type = 'GRU'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                    input_size=embedding_size,
                    hidden_size=hidden_side,
                    num_layers=layers,
                    bidirectional=bidirection,
                    dropout=dropout
                )
        else:
            self.rnn = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_side,
                num_layers=layers,
                bidirectional=bidirection,
                dropout=dropout
            )

        self.dropout = nn.Dropout(p=dropout)

        self.bidirection = bidirection
        if self.bidirection:
            self.linear = nn.Linear(embedding_size * 2, embedding_size)

    def forward(self, x):
        x = self.dropout(self.emb(x)).permute(1, 0, 2) # [batch, seq, emb] to [seq, batch, emb]

        # Forwarding rnn
        output, hidden_state = self.rnn(x)

        # Map to previous nums of features
        if self.bidirection:
            output = self.linear(output)
        return output, hidden_state



class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_side, layers, num_attn_head, dropout, device, rnn_type = 'GRU'):
        super().__init__()
        self.device = device
        self.emb = nn.Embedding(vocab_size, embedding_size)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_side,
                num_layers=layers,
                dropout=dropout
            )
        else:
            self.rnn = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_side,
                num_layers=layers,
                dropout=dropout
            )

        self.dropout = nn.Dropout(p=dropout)

        self.decode = nn.Linear(hidden_side, vocab_size)

    def forward(self, x, encoder_output, encoder_hidden_state, max_seq_length=20, teaching_rate=0.5):
        outputs_indices = []
        seq_length = 0
        #
        # send in the <start_of_seq>
        start_of_seq = torch.ones(x.shape[0], 1).long()
        start_of_seq = start_of_seq.to(self.device)
        start_of_seq = self.dropout(self.emb(start_of_seq)).permute(1, 0, 2)
        output_features, hidden_state = self.rnn(start_of_seq, encoder_hidden_state)

        # Obtain the indice for next seq
        output_indice = self.decode(output_features)
        outputs_indices.append(output_indice)

        # Recursively
        for i in range(1, max_seq_length):
            # Passing the target token if random value is larger than teaching rate
            input = self.dropout(
                self.emb(x[:,i:i + 1] if random.random() >= teaching_rate else torch.argmax(output_indice, dim=2).T)
            ).permute(1, 0, 2)

            output_features, hidden_state = self.rnn(input, hidden_state)
            output_indice = self.decode(output_features)
            outputs_indices.append(output_indice)

        return torch.cat(outputs_indices, dim=0)



class S2SPL(LightningModule):
    def __init__(
            self,
            vocab_size_encoder,
            vocab_size_decoder,
            input_vocab,
            target_vocab,
            input_id_to_word,
            target_id_to_word,
            args
    ):
        super().__init__()
        self.args = args
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.input_id_to_word = input_id_to_word
        self.target_id_to_word = target_id_to_word

        # Encoder
        self.encoder = RNNEncoder(
            vocab_size=vocab_size_encoder,
            embedding_size=self.args['EMB_DIM'],
            hidden_side=self.args['HIDDEN_ENCODER'],
            layers=self.args['ENCODER_LAYERS'],
            bidirection=self.args['BIDIRECTION'],
            rnn_type=self.args['ENCODER_TYPE'],
            dropout=self.args['DROPOUT_ENCODER']
        )
        # Decoder
        self.decoder = RNNDecoder(
            vocab_size=vocab_size_decoder,
            embedding_size=self.args['EMB_DIM'],
            hidden_side=self.args['HIDDEN_DECODER'],
            layers=self.args['DECODER_LAYERS'],
            num_attn_head=self.args['ATTENTION_HEAD'],
            rnn_type=self.args['DECODER_TYPE'],
            dropout=self.args['DROPOUT_DECODER'],
            device=self.args['DEVICE']
        )
        # Loss function
        self.cross_entrophy = nn.CrossEntropyLoss(reduction='none')

        #Log

        # Check point
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.args['LEARNING_RATE'])
        return optimizer


    def forward(self, inputs, teaching, teaching_rate=0.5):
        output_enc, hidden_state_enc = self.encoder(inputs)
        output_dec = self.decoder(x=teaching, encoder_output=output_enc, encoder_hidden_state=hidden_state_enc, max_seq_length=teaching.shape[1], teaching_rate=teaching_rate)
        return output_dec

    def compute_loss(self, inputs, targets, teaching, teaching_rate=0.5):
        outputs = self.forward(inputs, teaching, teaching_rate=0.5)
        loss = self.cross_entrophy(outputs.permute(1, 2, 0), targets)[targets != 0]
        loss = loss.mean()
        return loss, outputs

    def training_step(self, train_batch, batch_idx):
        inputs, targets, teaching = train_batch
        loss, _ = self.compute_loss(inputs, targets, teaching, teaching_rate=0.5)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        inputs, targets, teaching = valid_batch
        loss, _ = self.compute_loss(inputs, targets, teaching, teaching_rate=0)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        inputs, targets, teaching = test_batch
        loss, _ = self.compute_loss(inputs, targets, teaching, teaching_rate=0)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

    def translate(self, inputs_string):
        pass

    # def training_step(self, train_batch, batch_idx):
    #     inputs, targets, teaching, masks = train_batch
    #     loss, _ = self.compute_loss(inputs, targets, teaching) if random.random() > 0.5 else self.compute_loss_recursive(inputs, targets)
    #     self.log('train_loss', loss, on_step=False ,on_epoch=True)
    #     return loss
    # def forward(self, x, y):
    #     output_enc, hidden_state = self.encoder(x)
    #     output_dec = self.decoder(y, output_enc, hidden_state)
    #     return output_dec
    #
    # def compute_loss(self, inputs, targets, teaching):
    #     outputs = self.forward(inputs, teaching)
    #     loss = self.cross_entrophy(outputs.permute(1, 2, 0), targets)[targets != 0]
    #     loss = loss.mean() # / torch.sum(masks).item()
    #     return loss, outputs
    #
    # def compute_loss_recursive(self, inputs, targets):
    #     # outputs = self.recursive(inputs, max_seq_length=targets.shape[1])
    #     outputs = self.decoder.recursive_forward(
    #         x=inputs,
    #
    #     )
    #     loss = self.cross_entrophy(outputs.permute(1, 2, 0), targets)[targets != 0]
    #     loss = loss.mean()# / torch.sum(masks).item()
    #     return loss, outputs
    #
    #
    #
    # def translate(self, input_sentence, max_seq_length=20):
    #     # Transfer input string to indice
    #     with torch.no_grad():
    #         input_indice = []
    #         for token in re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', input_sentence.lower()):
    #             try:
    #                 input_indice.append(self.input_vocab[token])
    #             except:
    #                 input_indice.append(3)
    #         print(re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', input_sentence.lower()))
    #         print(input_indice)
    #         # Transfer indice to LongTensor with batch dimension shape: [1, seq]
    #         input_tokens = torch.LongTensor(input_indice).unsqueeze(0)
    #
    #         output_feature = self.recursive(input_tokens, max_seq_length=max_seq_length)
    #         output_indice = torch.argmax(output_feature, dim=2).T.squeeze(0)
    #         output_sentence = [self.target_id_to_word[int(i)] for i in output_indice]
    #     return output_sentence
    #     # with torch.no_grad():
    #     #     output_enc, hidden_state = self.encoder(input_tokens)
    #     #     # Recursive Decoder output
    #     #     output_dec = []
    #     #     seq_length = 0
    #     #
    #     #     # send in the <start_of_seq>
    #     #     start_of_seq = torch.LongTensor([1]).unsqueeze(0)
    #     #     start_of_seq = self.decoder.emb(start_of_seq)
    #     #     output_seq, hidden_state = self.decoder.rnn(start_of_seq, hidden_state)
    #     #     output_dec.append(output_seq)
    #     #
    #     #     # Getting remaining output by recursive
    #     #     while seq_length < max_seq_length:
    #     #         output_seq, hidden_state = self.decoder.rnn(output_seq, hidden_state)
    #     #         output_dec.append(output_seq)
    #     #         seq_length += 1
    #     #     output = torch.cat(output_dec, dim=0)
    #     #     output, _ = self.decoder.attn(output, output_enc, output_enc)
    #     #     output = self.decoder.decode(output)
    #     #
    #     #     # decoder to output sentences
    #     #     output = torch.argmax(output, dim=2).T.squeeze(0)
    #     #     output = [self.target_id_to_word[int(i)] for i in output]
    #     #
    #     # return output
    #
    #
    #
    #
    # def validation_step(self, val_batch, batch_idx):
    #     inputs, targets, teaching, masks = val_batch
    #     loss, outputs = self.compute_loss_recursive(inputs, targets)
    #     self.log('valid_loss', loss, on_step=False ,on_epoch=True)
    #
    # def on_validation_end(self):
    #     # if batch_idx == 0:
    #     #     f = open("sample.txt", "a+", encoding='utf-8')
    #     #     f.write(f'Batch: {batch_idx}\n')
    #     #     outputs = torch.argmax(outputs, dim=2).T
    #     #     for idx, (output, target) in enumerate(zip(outputs, targets)):
    #     #
    #     #         output_str = ' '.join([self.target_id_to_word[int(word)] for word in output])
    #     #         target_str = ' '.join([self.target_id_to_word[int(word)] for word in target])
    #     #
    #     #         f.write(f'{output_str}\n{target_str}\n\n')
    #     #         if idx >= 60:
    #     #             break
    #     #     f.close()
    #     pass
    #
    # def test_step(self, test_batch, batch_idx):
    #     inputs, targets, teaching, masks = test_batch
    #     loss, outputs = self.compute_loss_recursive(inputs, targets)
    #     self.log('test_loss', loss, on_step=False, on_epoch=True)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    from config import args
    from dataset import en_de_dataset, collate_fn_padding
    from torch.utils.data import DataLoader
    trainset = en_de_dataset(split='train')
    train_loader = DataLoader(
        dataset=trainset,
        shuffle=False,
        pin_memory=True,
        batch_size=2,
        collate_fn=collate_fn_padding,
        drop_last=True
    )

    s2s_model = S2SPL(
        vocab_size_encoder=len(trainset.input_vocab),
        vocab_size_decoder=len(trainset.target_vocab),

        input_vocab=trainset.input_vocab,
        target_vocab=trainset.target_vocab,
        input_id_to_word=trainset.id_to_word_input,
        target_id_to_word=trainset.id_to_word_target,
        args=args
    )
    critic = nn.CrossEntropyLoss(reduction='none')
    # output = s2s_model.forward(
    #     x = torch.randint(low=0, high=9, size=(3, 20)),
    #     y = torch.randint(low=0, high=9, size=(3, 20))
    # )

    for idx, (input, target, teaching, masks) in enumerate(train_loader):
        print(input.shape, target.shape)
        for idx, (sen1, sen2, sen3, mask) in enumerate(zip(input, target, teaching, masks)):
            sen1 = [trainset.id_to_word_input[int(word)] for word in sen1]
            sen2 = [trainset.id_to_word_target[int(word)] for word in sen2]
            sen3 = [trainset.id_to_word_target[int(word)] for word in sen3]
            mask = [bool(m) for m in mask]
            print(f'{" ".join(sen1)}\n{" ".join(sen2)}\n{" ".join(sen3)}\n{mask}\n\n')

        output = s2s_model.forward(
            x = input,
            y = teaching
        )

        print(output.shape)
        loss = critic(output.permute(1, 2, 0), target)
        print(loss.shape)
        print(loss)
        print()

        # First method
        loss_masked = loss
        loss_masked[~masks] = 0
        print(loss_masked)
        print(loss_masked.sum() / torch.sum(masks))
        print()

        # second
        loss_index = loss[masks]
        print(loss_index)
        print(loss_index.mean())
        print(torch.randn(2,4,6)[:,2:3,:].shape)
        print(torch.cat([torch.randn(2,4,6)[:,:2,:],torch.randn(2,4,6)[:,2:,:]], dim=1).shape)
        break

            # Second method
            # Calculate loss
            # print(output.permute(1, 2, 0).shape)
            # critic = nn.CrossEntropyLoss(reduction='none')
            # loss = critic(output.permute(1, 2, 0), target)
            # # loss = critic(output.permute(1, 0, 2), target)
            #
            # print(output.permute(1, 0, 2).shape, loss.shape, loss.mean(), torch.cat([target, start_of_seq], dim=1).long())





# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
