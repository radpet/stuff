from keras import Input
from keras import Model
from keras.datasets import imdb
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras_resources.imbd_sentiment.attention import Attention
from keras_resources.imbd_sentiment.attention_context import AttentionWithContext


def load(num_words=None):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=num_words, seed=113)

    return x_train, y_train, x_test, y_test


def preprocess(x, maxlen=264):
    return pad_sequences(x, maxlen=maxlen)


def bi_lstm(num_words, maxlen):
    input = Input(shape=(maxlen,))

    embedding = Embedding(input_dim=num_words, output_dim=128)(input)
    lstm = Bidirectional(LSTM(128, return_sequences=False))(embedding)
    output = Dense(32, activation='relu')(lstm)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=input, outputs=output)

    model.summary()
    return model


def bi_lstm_attention(num_words, maxlen):
    input = Input(shape=(maxlen,))

    embedding = Embedding(input_dim=num_words, output_dim=128)(input)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    attention = AttentionWithContext()(lstm)
    # attention = Attention(step_dim=maxlen)(lstm)
    output = Dense(32, activation='relu')(attention)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=input, outputs=output)

    model.summary()
    return model


def run():
    x_train, y_train, x_test, y_test = load()
    print('Mean len of sentence in train', np.mean([len(x) for x in x_train]))
    print('Mean len of sentence in test', np.mean([len(x) for x in x_test]))

    maxlen = 264
    x_train = preprocess(x_train, maxlen=maxlen)
    x_test = preprocess(x_test, maxlen=maxlen)

    # model = bi_lstm(num_words=20000, maxlen=maxlen)
    # model = bi_lstm_attention(num_words=20000, maxlen=maxlen)
    # model = bi_lstm_attention(num_words=20000, maxlen=maxlen)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ~ val_loss: 0.3855 - val_acc: 0.8690
    # model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=3)


if __name__ == '__main__':
    run()
