import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from lightning.pytorch import LightningModule


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, layers, bidirection = False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=embedding_size,
                num_layers=layers,
                bidirectional=bidirection
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
    def __init__(self, vocab_size, embedding_size, layers, num_attn_head):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=layers
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=300,
            num_heads=num_attn_head,
        )

        self.decode = nn.Sequential(
            nn.Linear(embedding_size, vocab_size)
        )



    def forward(self, x, encoder_output, encoder_hidden_state):
        # Getting word embedding then [batch, seq, emb] to [seq, batch, emb]
        x = self.emb(x).permute(1, 0, 2)

        # forwarding encoder with teacher forching
        query, hidden_state = self.rnn(x, encoder_hidden_state)

        # Apply multihead_attn
        output, weight = self.attn(query, encoder_output, encoder_output)

        # using linear layers mapping the features to vocab index
        output = self.decode(output)
        return output


class S2SPL(LightningModule):
    def __init__(
            self,
            vocab_size_encoder,
            emb_dim,
            encoder_layers,
            bidirection,

            vocab_size_decoder,
            decoder_layers,
            attn_head,

            learning_rate,
            input_id_to_word,
            target_id_to_word
    ):
        super().__init__()
        self.lr = learning_rate
        self.input_id_to_word = input_id_to_word
        self.target_id_to_word = target_id_to_word

        # Encoder
        self.encoder = RNNEncoder(
            vocab_size=vocab_size_encoder,
            embedding_size=emb_dim,
            layers=encoder_layers,
            bidirection=bidirection
        )
        # Decoder
        self.decoder = RNNDecoder(
            vocab_size=vocab_size_decoder,
            embedding_size=emb_dim,
            layers=decoder_layers,
            num_attn_head=attn_head
        )
        # Loss function
        self.cross_entrophy = nn.CrossEntropyLoss()

        #Log

    def forward(self, x, y):
        output_enc, hidden_state = self.encoder(x)

        start_of_seq = torch.ones((y.shape[0],1)).to('cuda')
        output_dec = self.decoder(torch.cat([start_of_seq,y], dim=1).long(), output_enc, hidden_state)
        return output_dec

    def compute_loss(self, inputs, targets, masks):
        outputs = self.forward(inputs, targets)

        end_of_seq = (torch.ones((targets.shape[0], 1)) * 2).to('cuda')
        loss = self.cross_entrophy(outputs.permute(1, 2, 0), torch.cat([targets, end_of_seq], dim=1).long()) * masks
        loss = loss.sum() / torch.sum(masks).item()
        return loss, outputs

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, targets, masks = train_batch
        loss, _ = self.compute_loss(inputs, targets, masks)
        self.log('train_loss', loss, on_step=False ,on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets, masks = val_batch
        loss, outputs = self.compute_loss(inputs, targets, masks)
        if batch_idx == 0:
            f = open("sample.txt", "a+", encoding='utf-8')
            f.write(f'Batch: {batch_idx}\n')
            outputs = torch.argmax(outputs, dim=2).T
            for idx, (output, target) in enumerate(zip(outputs, targets)):

                output_str = ' '.join([self.target_id_to_word[int(word)] for word in output])
                target_str = ' '.join([self.target_id_to_word[int(word)] for word in target])

                f.write(f'{output_str}\n{target_str}\n\n')
                if idx >= 60:
                    break
            f.close()
        self.log('valid_loss', loss, on_step=False ,on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        inputs, targets, masks = test_batch
        loss, output = self.compute_loss(inputs, targets, masks)

        self.log('test_loss', loss)



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
