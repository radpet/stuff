import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from classifiers import models
from features.sentiment import VaderSentimentWithMem, VaderSentiment
from util import ys, save_trained_model, load_clean_train, TEXT

MODEL = models['TF_IDF_NB_SENTI']()


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def replace_sentiment_step(model, sentiment):
    return model.set_params(featureunion__sentiment=sentiment)


def run():
    print(MODEL.get_params().keys())
    parameters = {
        'featureunion__tf_idf__tfidfvectorizer__max_df': (0.5, 0.75, 1.0),
        'featureunion__tf_idf__tfidfvectorizer__min_df': (0.01, 0.05),
        'featureunion__tf_idf__tfidfvectorizer__max_features': (10000, 50000),
        'logisticregression__C': [0.01, 0.1, 1, 10],
        'logisticregression__class_weight': ["balanced"]
    }

    train = load_clean_train()
    model = replace_sentiment_step(MODEL, VaderSentimentWithMem())

    for y in ys:
        print("Fine tuning for", y)
        search = RandomizedSearchCV(model, parameters, cv=StratifiedKFold(n_splits=3), n_jobs=-1, n_iter=2,
                                    scoring='f1', random_state=12345)
        search.fit(train[TEXT].values, train[y].values)
        report(search.cv_results_)

        save_trained_model(replace_sentiment_step(search.best_estimator_, VaderSentiment()),
                           "{}_TF_IDF_NB_SENTI_ft".format(y))


if __name__ == '__main__':
    run()
