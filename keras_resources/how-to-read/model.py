import pprint
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Lambda, Conv1D, MaxPooling1D, Dense, TimeDistributed, \
    Bidirectional, LSTM, Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

pp = pprint.PrettyPrinter(indent=2, width=41, compact=True)


def load():
    data = pd.read_csv('./data/labeledTrainData.tsv.zip', delimiter='\t')
    return data


def clean(text):
    TAG_RE = re.compile(r'<[^>]+>')
    text = TAG_RE.sub('', text)
    text = text.lower()
    text = text.replace('\\', '')
    text = re.sub('[^\x00-\x7f]', '', text)
    return text


def create_tokenizer(text):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text)
    return tokenizer


def preprocess(data, maxlen_doc=15, maxlen_sentence=512):
    # (document/sentence/char)
    X = np.zeros((data['review'].shape[0], maxlen_doc, maxlen_sentence))
    y = data['sentiment']
    data_cleaned = data['review'].map(lambda review: clean(review))
    tokenizer = create_tokenizer(data_cleaned)

    # corpus_size = len(tokenizer.word_counts)

    for i, (review, label) in enumerate(zip(data['review'][:5], data['sentiment'][:5])):
        sentences = re.split(r'(?<!\w\.\w.)(?<![a-z]\.)(?<=\.|\?)\s', review)

        tokenized = tokenizer.texts_to_sequences(sentences)
        tokenized = pad_sequences(tokenized, maxlen_sentence)

        if (tokenized.shape[0] > maxlen_doc):
            tokenized = np.choice(tokenized, maxlen_doc)

        X[i][np.arange(min(tokenized.shape[0], maxlen_doc))] = tokenized

    # pp.pprint(X[0])

    return X, y, tokenizer


def _one_hot(char_seq, corpus_len):
    return tf.to_float(tf.one_hot(char_seq, corpus_len, on_value=1, off_value=0, axis=-1))


def one_hot(corpus_len):
    return lambda char_seq: _one_hot(char_seq, corpus_len)


def _one_hot_outshape(in_shape, corpus_len):
    return in_shape[0], in_shape[1], corpus_len


def one_hot_outshape(corpus_len):
    return lambda in_shape: _one_hot_outshape(in_shape, corpus_len)


def cnn_bilstm(maxlen_doc, maxlen_sentence, corpus_len):
    in_sentence = Input(shape=(maxlen_sentence,), dtype='int64', name='input_sentence')

    one_hot_encoding = one_hot(corpus_len)
    output_shape_one_hot_encoding = one_hot_outshape(corpus_len)
    embedded = \
        Lambda(one_hot_encoding, output_shape=output_shape_one_hot_encoding, name='char_one_hot_encoding')(in_sentence)

    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 256]
    pool_length = 2

    for i in range(len(nb_filter)):
        embedded = Conv1D(filters=nb_filter[i],
                          kernel_size=filter_length[i],
                          padding='valid',
                          activation='relu',
                          strides=1)(embedded)

        embedded = Dropout(0.1)(embedded)
        embedded = MaxPooling1D(pool_size=pool_length)(embedded)

    bi_lstm_sent = Bidirectional(
        LSTM(100, return_sequences=False, dropout=0.15, recurrent_dropout=0.15))(embedded)

    sent_encode = Dropout(0.3)(bi_lstm_sent)
    encoder = Model(inputs=in_sentence, outputs=sent_encode)
    encoder.summary()

    document = Input(shape=(maxlen_doc, maxlen_sentence), dtype='int64', name='intput_document')

    encoded = TimeDistributed(encoder)(document)

    bi_lstm_doc = Bidirectional(LSTM(100, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))(encoded)

    output = Dropout(0.3)(bi_lstm_doc)
    output = Dense(32, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=document, outputs=output)

    model.summary()

    return model


def run():
    maxlen_doc = 20
    maxlen_sentence = 256

    data = load()
    X, y, tokenizer = preprocess(data, maxlen_doc=maxlen_doc, maxlen_sentence=maxlen_sentence)
    corpus_len = len(tokenizer.word_counts)

    model = cnn_bilstm(maxlen_doc=maxlen_doc, maxlen_sentence=maxlen_sentence, corpus_len=corpus_len)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=26, stratify=y)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=3)


if __name__ == '__main__':
    run()
    # pp.pprint(one_hot(5)(np.array([1, 2, 3])))
    # pp.pprint(one_hot_outshape(5)())
