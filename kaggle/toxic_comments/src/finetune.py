import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from classifiers import models
from util import ys, load_clean_train_senti, TOXIC, S_TOXIC, OBSCENE, THREAT

BEST_MODELS = {
    TOXIC: (models["TF_IDF_NB_SENTI"],{
        'featureunion__tfidf__pipeline__tfidfvectorizer__ngram_range': ((1,1),(1,2)),
        'featureunion__tfidf__pipeline__tfidfvectorizer__max_features': (None, 50000),
        'logisticregression__C': [0.01, 0.1, 1, 10],
    }),
    S_TOXIC: (models["TF_IDF_NB_SENTI"],{
        'featureunion__tfidf__pipeline__tfidfvectorizer__ngram_range': ((1,1),(1,2)),
        'featureunion__tfidf__pipeline__tfidfvectorizer__max_features': (None, 50000),
        'logisticregression__C': [0.01, 0.1, 1, 10],
    }),
    OBSCENE: (models["TF_IDF_NB_SENTI"],{
        'featureunion__tfidf__pipeline__tfidfvectorizer__ngram_range': ((1,1),(1,2)),
        'featureunion__tfidf__pipeline__tfidfvectorizer__max_features': (None, 50000),
        'logisticregression__C': [0.01, 0.1, 1, 10],
    }),
    THREAT: (models["TF_IDF_SVM"], {
        'tfidfvectorizer__ngram_range': ((1,1),(1,2)),
        'tfidfvectorizer__max_features': (None, 50000),
        'linearsvc__C': [0.01, 0.1, 1, 10],
    })
}

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



def run():

    train = load_clean_train_senti()

    for y in ys:

        print("Fine tuning for", y)
        model = BEST_MODELS[y][0]()
        parameters = BEST_MODELS[y][1]
        search = RandomizedSearchCV(model, parameters, cv=StratifiedKFold(n_splits=3), n_iter=5,
                                    scoring='f1', random_state=12345, n_jobs=-1)

        search.fit(train, train[y].values)

        report(search.cv_results_)
        score = round(search.best_score_,3)
        print(score)

        with open('./models/clf_report_{}_{}'.format(y, score), 'w') as f:
            f.write(str(search.best_score_))
        with open('./models/clf_params_{}_{}'.format(y, score), 'w') as f:
            f.write(str(search.best_params_))

        # save_trained_model(search.best_estimator_,
        #                    "{}_TF_IDF_NB_SENTI_ft".format(y))


if __name__ == '__main__':
    run()
