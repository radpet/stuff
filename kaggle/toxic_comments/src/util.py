import json
import os
import pickle
from math import ceil

import numpy as np
import pandas as pd

from clean_comments import clean

TOXIC = 'toxic'
S_TOXIC = 'severe_toxic'
OBSCENE = 'obscene'
THREAT = 'threat'
INSULT = 'insult'
I_HATE = 'identity_hate'
ys = [TOXIC, S_TOXIC, OBSCENE, THREAT, INSULT, I_HATE]

TEXT = 'comment_text'

TRAINED_MODELS_PATH = 'models'
REPORTS_PATH = 'reports'


def load_train():
    return pd.read_csv('./data/train.csv')


def load_clean_train():
    clean_path = './data/train_clean.csv'
    # if os.path.exists(clean_path):
    #     print('Train has been already cleaned. Loading..')
    #     return pd.read_csv(clean_path)
    # else:
    train = load_train()
    corpus = train[TEXT]

    print('Cleaning train dataset')
    clean_corpus = corpus.apply(clean)
    print('Clean train done')

    train[TEXT] = clean_corpus

    train.to_csv(clean_path, index=False)

    return train


def load_test():
    return pd.read_csv('./data/test.csv')


def load_clean_test():
    clean_path = './data/test_clean.csv'

    test = load_test()
    corpus = test[TEXT]

    print('Cleaning test')
    clean_corpus = corpus.apply(clean)
    print('Clean test done')

    test[TEXT] = clean_corpus

    test.to_csv(clean_path, index=False)
    return test

def save_trained_model(obj, model_name):
    trained_path = './models/'

    if not os.path.exists(trained_path):
        os.mkdir(trained_path)

    path = os.path.join(trained_path, model_name)
    write_obj(obj, path)

    print('Saved model to', path)


def load_trained_model(model_name):
    path = os.path.join( './models/', model_name)
    print('Loading model from', path)
    return load_obj(path)

class AbstractTrainReport:
    def __init__(self, scorer, path):
        self.scorer = scorer
        self.path = path
        self.model_def = {}
        self.model_name = 'default'
        self.model_score = -1
        self.report = {}

    def _gen_model_path_name(self):
        return self.model_name + '_' + self.scorer.__name__ + str(np.around(self.model_score, 4))

    def update(self, model_name, model_def, model_score):
        self.model_def = model_def
        self.model_score = model_score
        self.model_name = model_name

    def save_to_fs(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        write_obj(self.model_def, os.path.join(self.path, self._gen_model_path_name() + '.pkl'))


class TrainReport(AbstractTrainReport):
    def __init__(self, scorer, base_path=''):
        super().__init__(scorer, os.path.join(base_path, REPORTS_PATH))


class BestModel(AbstractTrainReport):
    def __init__(self, scorer, base_path=''):
        super().__init__(scorer, os.path.join(base_path, TRAINED_MODELS_PATH))

    def update(self, model_name, model_def, model_score):
        if self.model_score < model_score:
            print('New model with higher {} found'.format(self.scorer.__name__), model_score)
            self.model = model_def
            self.model_score = model_score
            self.model_name = model_name


def write_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def write_txt(txt, path):
    with open(path, 'w') as f:
        f.write(txt)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, path)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_batch(iterable, n=1):
    l = len(iterable)
    print('Total batches %s' % (ceil(l / n)))
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
