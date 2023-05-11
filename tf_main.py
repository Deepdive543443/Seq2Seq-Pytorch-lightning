import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Embedding
import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.emb = Embedding(9000, 300)
        self.rnn = GRU(150, return_sequences=True, return_state=True)

    def call(self, inputs):
        x = self.emb(inputs)

        return self.rnn(x)

if __name__ == "__main__":
    model = MyModel()
    model.compile('rmsprop', 'mse')
    output, hidden = model.predict(np.random.randint(low=0, high=8999, size=(3, 26)))
    print(output.shape, hidden.shape)