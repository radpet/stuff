import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from classifiers import models
from util import TEXT, ys, save_trained_model, load_clean_train_senti


def find_best_model(train, y):
    now = datetime.now()
    time_path = os.path.join('./', now.strftime("%Y_%m_%d-%H:%M:%S"))
    if not os.path.exists(time_path):
        os.mkdir(time_path)
    print('Current time_path', time_path)

    print('Training model for', y)

    kf = StratifiedKFold(n_splits=3, random_state=12345, shuffle=True)

    for model_name, model_def in models.items():
        try:
            print('Training model', model_name)
            model = model_def()
            scores = cross_val_score(model, train, train[y], scoring="f1", cv=kf)
            mean_score = round(np.mean(scores), 3)
            print('Mean f1 score of {} is {}'.format(model_name, mean_score))
            # preds = cross_val_predict(model, train, train[y], n_jobs=-1)
            # conf_matrix = confusion_matrix(y_true=train[y], y_pred=preds)
            # print(conf_matrix)
            with open(os.path.join(time_path, '{}_{}_{}'.format(y, model_name, mean_score)), 'w') as f:
                f.write(str(mean_score))

        except Exception as e:
            print(e)
            print('Error training', model_name, 'Skipping.')


def run():
    train = load_clean_train_senti()
    for y in ys:
        find_best_model(train, y)

    # train_models(train)


def train_models(train):
    chosen_model = 'TF_IDF_NB_SENTI'
    for y in ys:
        print('Training model for', y)
        model = models[chosen_model]()
        model.fit(train[TEXT].values, train[y].values)
        save_trained_model(model, y + '_' + chosen_model)


if __name__ == '__main__':
    run()
