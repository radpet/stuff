import json
import os
import pickle
from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

TOXIC = 'toxic'
S_TOXIC = 'severe_toxic'
OBSCENE = 'obscene'
THREAT = 'threat'
INSULT = 'insult'
I_HATE = 'identity_hate'
ys = [TOXIC, S_TOXIC, OBSCENE, THREAT, INSULT, I_HATE]

TEXT = 'comment_text'

TRAINED_MODELS_PATH = './models'


def load_train():
    return pd.read_csv('./data/train.csv')


class BestModel:
    def __init__(self, scorer):
        self.scorer = scorer
        self.model = {}
        self.model_name = None
        self.model_score = -1
        self.report = {}

    def try_to_update_best_model(self, model_name, model, y_true, y_pred, model_score):
        if self.model_score < model_score:
            print('New model with higher f1 found', model_score)
            self.model = model
            self.model_score = model_score
            self.report['conf_matrix'] = confusion_matrix(y_true, y_pred)
            self.report['classification_report'] = classification_report(y_true, y_pred)
            self.model_name = model_name

    def _gen_model_path_name(self):
        return self.model_name + '_' + self.scorer.__name__ + str(np.around(self.model_score,4))

    def save_to_fs(self):
        if not os.path.exists(TRAINED_MODELS_PATH):
            os.mkdir(TRAINED_MODELS_PATH)
        write_obj(self.model, os.path.join(TRAINED_MODELS_PATH, self._gen_model_path_name() + '.pkl'))
        write_txt(str(self.report['conf_matrix']),
                  os.path.join(TRAINED_MODELS_PATH, self._gen_model_path_name() + '_conf_matrix.txt'))
        write_txt(self.report['classification_report'],
                  os.path.join(TRAINED_MODELS_PATH, self._gen_model_path_name() + '_cls_report.json'))


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
