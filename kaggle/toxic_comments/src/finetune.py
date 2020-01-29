import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline

from classifiers import nb_tf_idf
from features.column import ColumnSelector
from util import ys, TEXT, load_clean_train_senti


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


def tf_idf_nb_sent():
    features = FeatureUnion([
        ("tfidf", make_pipeline(
            ColumnSelector(TEXT),
            nb_tf_idf(TfidfVectorizer(ngram_range=(1, 2),
                                      min_df=3, max_df=0.9, strip_accents='unicode',
                                      stop_words='english', use_idf=1,
                                      smooth_idf=1, sublinear_tf=1, max_features=10000))
        )),
        ("sentiment", ColumnSelector(['pos', 'neu', 'neg'])),
        ("hate_model", ColumnSelector(['hate_model']))
    ], n_jobs=-1)

    return make_pipeline(features, LogisticRegression(max_iter=1000, class_weight='balanced'))


def run():
    parameters = {
        'featureunion__tfidf__pipeline__tfidfvectorizer__max_df': (0.5, 0.75, 1.0),
        'featureunion__tfidf__pipeline__tfidfvectorizer__min_df': (0.01, 0.05),
        'featureunion__tfidf__pipeline__tfidfvectorizer__max_features': (10000, 50000),
        'logisticregression__C': [0.01, 0.1, 1, 10],
    }

    model = tf_idf_nb_sent()
    print(model.get_params().keys())

    train = load_clean_train_senti()

    for y in ys:
        print("Fine tuning for", y)
        train, test = train_test_split(train, train_size=0.9, stratify=train[y])
        print(train.shape, test.shape)
        search = RandomizedSearchCV(model, parameters, cv=StratifiedKFold(n_splits=3), n_iter=2,
                                    scoring='f1', random_state=12345, n_jobs=-1)

        search.fit(train, train[y].values)

        report(search.cv_results_)
        preds = search.best_estimator_.predict(test)
        f1score = round(f1_score(y_pred=preds, y_true=test[y].values), 3)
        print(f1score)
        clf_report = classification_report(y_pred=preds, y_true=test[y].values)
        print(clf_report)

        conf_matrix = confusion_matrix(y_pred=preds, y_true=test[y].values)

        with open('./models/clf_report_{}_{}'.format(y, f1score), 'w') as f:
            f.write(clf_report)
        with open('./models/clf_conf_matrix_{}_{}'.format(y, f1score), 'w') as f:
            f.write(str(conf_matrix))

        # save_trained_model(search.best_estimator_,
        #                    "{}_TF_IDF_NB_SENTI_ft".format(y))


if __name__ == '__main__':
    run()
