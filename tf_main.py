import tensorflow as tf
import numpy as np
from config import args

nn = tf.keras.layers


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, args):
        super().__init__()
        self.emb = nn.Embedding(9000, args['EMB_DIM'])
        self.rnn = nn.Bidirectional(
            nn.GRU(args['EMB_DIM'], dropout=args['DROPOUT_ENCODER'], return_sequences=True, return_state=True)
        )

        self.rnn_layers = []
        for i in range(args['ENCODER_LAYERS']):
            self.rnn_layers.append(
                nn.Bidirectional(
                    nn.GRU(300, dropout=args['DROPOUT_ENCODER'], return_sequences=True, return_state=True)
                )
            )
        self.dropout = nn.Dropout(args['DROPOUT_ENCODER'])

    def call(self, x):
        x = self.dropout(self.emb(x))
        hidden_output = []
        for layer in self.rnn_layers:
            x, h1, h2 = layer(x)
            hidden_output.append(h1)
            hidden_output.append(h2)
        hidden_output = tf.stack(hidden_output, axis=1)
        return x, hidden_output



if __name__ == "__main__":
    model = MyModel(vocab_size=9000, args= args)
    model.compile('rmsprop', 'mse')
    output, hidden = model.predict(np.random.randint(low=0, high=8999, size=(3, 26)))
    print(output.shape, hidden.shape)