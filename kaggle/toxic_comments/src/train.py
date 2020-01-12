from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import numpy as np
from classifiers import models
from util import load_train, ys, TEXT, BestModel


def run():
    data = load_train()

    for y in ys[:1]:
        print('Training model for ', y)

        X_train, X_test, y_train, y_test = train_test_split(data[TEXT], data[y], stratify=data[y], random_state=1234)

        print('Train shape', X_train.shape)
        print('Test shape', X_test.shape)
        best_model = BestModel(roc_auc_score)
        try:
            for model_name, model_def in models:
                print('Training model', model_name)
                model = model_def()

                cross_val = cross_validate(model, X=data[TEXT], y=data[y], scoring='roc_auc', cv=5,return_estimator=True)
                model_score = cross_val['test_score'].mean()
                be = cross_val['estimator'][np.argmax(cross_val['test_score'])]
                y_prob = be.predict_proba(X_test)
                y_pred = y_prob[:,1] >= 0.5
                print("Model={} score is {}".format(model_name, model_score))
                best_model.try_to_update_best_model(y + '_' + model_name, model, y_test, y_pred, model_score)
        finally:
            best_model.save_to_fs()


if __name__ == '__main__':
    run()
