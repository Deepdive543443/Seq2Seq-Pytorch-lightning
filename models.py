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
        self.bidirection = bidirection
        if self.bidirection:
            self.linear = nn.Linear(embedding_size * 2, embedding_size)

    def forward(self, x):
        x = self.emb(x).permute(1, 0, 2) # [batch, seq, emb] to [seq, batch, emb]

        # Forwarding rnn
        output, hidden_state = self.rnn(x)

        # Map to previous nums of features
        if self.bidirection:
            output = self.linear(output)
        return output, hidden_state



class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_side, layers, num_attn_head, dropout, rnn_type = 'GRU'):
        super().__init__()
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

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_side,
            num_heads=num_attn_head,
        )

        self.decode = nn.Linear(hidden_side, vocab_size)



    def forward(self, x, encoder_output, encoder_hidden_state):
        # Getting word embedding then [batch, seq, emb] to [seq, batch, emb]
        x = self.emb(x).permute(1, 0, 2)

        # forwarding encoder with teacher forching
        output, hidden_state = self.rnn(x, encoder_hidden_state)

        # Apply multihead_attn
        output, weight = self.attn(output, encoder_output, encoder_output)

        # using linear layers mapping the features to vocab index
        output = self.decode(output)
        return output



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
            dropout=self.args['DROPOUT_DECODER']
        )
        # Loss function
        self.cross_entrophy = nn.CrossEntropyLoss()

        #Log

        # Check point
        self.save_hyperparameters()

    def forward(self, x, y):
        output_enc, hidden_state = self.encoder(x)
        output_dec = self.decoder(y, output_enc, hidden_state)
        return output_dec

    def compute_loss(self, inputs, targets, teaching, masks):
        outputs = self.forward(inputs, teaching)
        loss = self.cross_entrophy(outputs.permute(1, 2, 0), targets) * masks
        loss = loss.sum() / torch.sum(masks).item()
        return loss, outputs

    def recursive(self, x, max_seq_length=20):
        self.train(mode=False)

        output_enc, hidden_state = self.encoder(x)
        # Recursive Decoder output
        output_dec = []
        seq_length = 0

        # send in the <start_of_seq>
        start_of_seq = torch.ones(x.shape[0], 1).long()
        start_of_seq = start_of_seq.to(self.device)
        start_of_seq = self.decoder.emb(start_of_seq)
        output_features, hidden_state = self.decoder.rnn(start_of_seq, hidden_state)

        output_indice = self.decoder.decode(output_features)

        output_dec.append(output_indice)

        while len(output_dec) < max_seq_length:
            input_feature = self.decoder.emb(torch.argmax(output_indice, dim=2))
            output_feature, hidden_state = self.decoder.rnn(input_feature, hidden_state)
            output_feature, _ = self.decoder.attn(output_feature, output_enc, output_enc)
            output_indice = self.decoder.decode(output_feature)
            output_dec.append(output_indice)
            seq_length += 1

        output = torch.cat(output_dec, dim=0)
        # output, _ = self.decoder.attn(output, output_enc, output_enc)
        # output = self.decoder.decode(output)

        self.train(mode=True)
        return output

    def compute_loss_recursive(self, inputs, targets, masks):
        with torch.no_grad():
            outputs = self.recursive(inputs, max_seq_length=targets.shape[1])
            # print(f"Max sequence lengthL {targets.shape[1]}   Output shape: {outputs.permute(1, 2, 0).shape}")
            loss = self.cross_entrophy(outputs.permute(1, 2, 0), targets) * masks
            loss = loss.sum() / torch.sum(masks).item()
        return loss, outputs



    def translate(self, input_sentence, max_seq_length=20):
        # Transfer input string to indice
        with torch.no_grad():
            input_indice = []
            for token in re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', input_sentence.lower()):
                try:
                    input_indice.append(self.input_vocab[token])
                except:
                    input_indice.append(3)
            print(re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', input_sentence.lower()))
            print(input_indice)
            # Transfer indice to LongTensor with batch dimension shape: [1, seq]
            input_tokens = torch.LongTensor(input_indice).unsqueeze(0)

            output_feature = self.recursive(input_tokens)
            output_indice = torch.argmax(output_feature, dim=2).T.squeeze(0)
            output_sentence = [self.target_id_to_word[int(i)] for i in output_indice]
        return output_sentence
        # with torch.no_grad():
        #     output_enc, hidden_state = self.encoder(input_tokens)
        #     # Recursive Decoder output
        #     output_dec = []
        #     seq_length = 0
        #
        #     # send in the <start_of_seq>
        #     start_of_seq = torch.LongTensor([1]).unsqueeze(0)
        #     start_of_seq = self.decoder.emb(start_of_seq)
        #     output_seq, hidden_state = self.decoder.rnn(start_of_seq, hidden_state)
        #     output_dec.append(output_seq)
        #
        #     # Getting remaining output by recursive
        #     while seq_length < max_seq_length:
        #         output_seq, hidden_state = self.decoder.rnn(output_seq, hidden_state)
        #         output_dec.append(output_seq)
        #         seq_length += 1
        #     output = torch.cat(output_dec, dim=0)
        #     output, _ = self.decoder.attn(output, output_enc, output_enc)
        #     output = self.decoder.decode(output)
        #
        #     # decoder to output sentences
        #     output = torch.argmax(output, dim=2).T.squeeze(0)
        #     output = [self.target_id_to_word[int(i)] for i in output]
        #
        # return output


    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.args['LEARNING_RATE'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, targets, teaching, masks = train_batch
        loss, _ = self.compute_loss(inputs, targets, teaching, masks)
        self.log('train_loss', loss, on_step=False ,on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets, teaching, masks = val_batch
        loss, outputs = self.compute_loss_recursive(inputs, targets, masks)
        self.log('valid_loss', loss, on_step=False ,on_epoch=True)

    def on_validation_end(self):
        # if batch_idx == 0:
        #     f = open("sample.txt", "a+", encoding='utf-8')
        #     f.write(f'Batch: {batch_idx}\n')
        #     outputs = torch.argmax(outputs, dim=2).T
        #     for idx, (output, target) in enumerate(zip(outputs, targets)):
        #
        #         output_str = ' '.join([self.target_id_to_word[int(word)] for word in output])
        #         target_str = ' '.join([self.target_id_to_word[int(word)] for word in target])
        #
        #         f.write(f'{output_str}\n{target_str}\n\n')
        #         if idx >= 60:
        #             break
        #     f.close()
        pass

    def test_step(self, test_batch, batch_idx):
        inputs, targets, teaching, masks = test_batch
        loss, outputs = self.compute_loss_recursive(inputs, targets, masks)
        self.log('test_loss', loss, on_step=False, on_epoch=True)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    encoder = RNNEncoder(3000, 300, 2, True)
    input = torch.randint(low=0, high=2999, size=(3, 46))

    # obtain the output and hidden state from encoder
    encoder_output, encoder_hidden_state = encoder(input)


    # Define the Decoder
    decoder = RNNDecoder(3000, 300, 4, 3)
    target = torch.randint(low=0, high=2999, size=(3, 37))
    start_of_seq = torch.ones((3, 1)) * 2
    target_onehot = torch.zeros(3, 37, 3000)
    output = decoder(target, encoder_output, encoder_hidden_state)


    # Calculate loss
    print(output.permute(1, 2, 0).shape)
    critic = nn.CrossEntropyLoss(reduction='none')
    loss = critic(output.permute(1, 2, 0), target)
    # loss = critic(output.permute(1, 0, 2), target)

    print(output.permute(1, 0, 2).shape, loss.shape, loss.mean(), torch.cat([target, start_of_seq], dim=1).long())





# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
