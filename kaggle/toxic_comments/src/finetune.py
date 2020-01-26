import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline

from classifiers import models, nb_tf_idf
from features.column import ColumnSelector
from features.sentiment import VaderSentimentWithMem, VaderSentiment
from util import ys, save_trained_model, load_clean_train, TEXT, load_clean_train_senti

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


def tf_idf_nb_sent():
    features = FeatureUnion([
        ("tfidf", make_pipeline(
            ColumnSelector(TEXT),
            nb_tf_idf(TfidfVectorizer(ngram_range=(1, 2),
                                      min_df=3, max_df=0.9, strip_accents='unicode',
                                      stop_words='english', use_idf=1,
                                      smooth_idf=1, sublinear_tf=1, max_features=10000))
        )),
        ("sentiment_pos", ColumnSelector('pos')),
        ("sentiment_neu",ColumnSelector('neu')),
        ("sentiment_neg",ColumnSelector('neg'))], n_jobs=-1)

    return make_pipeline(features, LogisticRegression(max_iter=1000))


def run():
    print(MODEL.get_params().keys())
    parameters = {
        'featureunion__tf_idf__tfidfvectorizer__max_df': (0.5, 0.75, 1.0),
        'featureunion__tf_idf__tfidfvectorizer__min_df': (0.01, 0.05),
        'featureunion__tf_idf__tfidfvectorizer__max_features': (10000, 50000),
        'logisticregression__C': [0.01, 0.1, 1, 10],
    }

    train = load_clean_train_senti()

    model = tf_idf_nb_sent()

    for y in ys:
        print("Fine tuning for", y)
        train, test = train_test_split(train, train_size=0.9, stratify=train[y])
        print(train.shape, test.shape)
        search = RandomizedSearchCV(model, parameters, cv=StratifiedKFold(n_splits=3), n_iter=2,
                                    scoring='f1', random_state=12345)
        search.fit(train, train[y].values)
        report(search.cv_results_)
        clf_report = classification_report(y_pred=model.predict(test), y_true=test[y].values)
        print(clf_report)
        with open('./models/clf_{}'.format(y), 'w') as f:
            f.write(clf_report)
        save_trained_model(model,
                           "{}_TF_IDF_NB_SENTI_ft".format(y))


if __name__ == '__main__':
    run()
