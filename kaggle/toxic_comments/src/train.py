import os
import traceback
from datetime import datetime

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from classifiers import models
from util import TEXT, BestModel, ys, load_clean_train, TrainReport, save_trained_model


def find_best_model(train, y):
    now = datetime.now()
    time_path = os.path.join('./', now.strftime("%Y_%m_%d-%H:%M:%S"))
    if not os.path.exists(time_path):
        os.mkdir(time_path)
    print('Current time_path', time_path)

    print('Training model for', y)

    kf = StratifiedKFold(n_splits=3, random_state=12345, shuffle=True)
    results = {}
    best_model = BestModel(log_loss, base_path=time_path)
    try:
        for idx, (train_index, test_index) in enumerate(kf.split(train[TEXT], train[y])):
            print('Split', idx)
            X_train = train.iloc[train_index][TEXT]
            y_train = train.iloc[train_index][y]
            X_test = train.iloc[test_index][TEXT]
            y_test = train.iloc[test_index][y]

            print('Train shape', X_train.shape)
            print('Test shape', X_test.shape)

            for model_name, model_def in models.items():
                try:
                    print('Training model', model_name)
                    model = model_def()

                    model.fit(X_train.values, y_train.values)
                    y_prob = model.predict_proba(X_test.values)
                    model_score = log_loss(y_test.values, y_prob[:, 1])
                    if model_name not in results:
                        results[model_name] = []

                    results[model_name].append(model_score)
                except Exception:
                    traceback.print_exc()
                    print('Error training %s . Skipping..' % model_name)

        for model_name, model_scores in results.items():
            model_score = np.mean(model_scores)
            print("Model={} score is {}".format(model_name, model_score))
            train_report = TrainReport(log_loss, base_path=time_path)
            train_report.update(y + '_' + model_name, models[model_name], model_score)
            best_model.update(y + '_' + model_name, models[model_name], model_score)
            train_report.save_to_fs()
    finally:
        best_model.save_to_fs()


def run():
    train = load_clean_train()
    # for y in ys[:1]:
    #     find_best_model(train, y)

    train_models(train)


def train_models(train):
    chosen_model = 'TF_IDF_NB_SENTI'
    for y in ys:
        print('Training model for', y)
        model = models[chosen_model]()
        model.fit(train[TEXT].values, train[y].values)
        save_trained_model(model, y + '_' + chosen_model)


if __name__ == '__main__':
    run()
