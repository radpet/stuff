import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted
from tensorflow_core.python.keras.callbacks import EarlyStopping

from features.column import ColumnSelector
from features.nbweights import NBWeights
from features.sentiment import VaderSentiment
from features.use_embeddings import USEEncoder
from util import TEXT


def tf_idf_svm_pipeline():
    return make_pipeline(ColumnSelector(TEXT),
                         TfidfVectorizer(stop_words='english'), LinearSVC(class_weight='balanced', max_iter=2000))


def tf_idf_logistic():
    return make_pipeline(ColumnSelector(TEXT), TfidfVectorizer(stop_words='english'),
                         LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=2000))


def tf_idf_logistic_senti():
    features = FeatureUnion([
        ("tfidf", make_pipeline(
            ColumnSelector(TEXT),
            TfidfVectorizer(stop_words='english')
        )),
        ("sentiment", ColumnSelector(['pos', 'neu', 'neg'])),
        ("hate_model", ColumnSelector(['hate_model']))
    ], n_jobs=-1)
    return make_pipeline(features, LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=2000))


def bow_logistic():
    features = FeatureUnion([
        ("bow", make_pipeline(
            ColumnSelector(TEXT),
            CountVectorizer(stop_words='english')
        ))
    ], n_jobs=-1)

    return make_pipeline(features,
                         LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=2000))


def bow_svm():
    features = FeatureUnion([
        ("bow", make_pipeline(
            ColumnSelector(TEXT),
            CountVectorizer(stop_words='english')
        ))
    ], n_jobs=-1)

    return make_pipeline(features,
                         LinearSVC(class_weight='balanced'))


def bow_logistic_senti():
    features = FeatureUnion([
        ("bow", make_pipeline(
            ColumnSelector(TEXT),
            CountVectorizer(stop_words='english')
        )),
        ("sentiment", ColumnSelector(['pos', 'neu', 'neg'])),
        ("hate_model", ColumnSelector(['hate_model']))
    ], n_jobs=-1)

    return make_pipeline(features,
                         LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=2000))


def tf_idf_multi_nb():
    features = FeatureUnion([
        ("tfidf", make_pipeline(
            ColumnSelector(TEXT),
            TfidfVectorizer(stop_words='english')
        )),
        ("sentiment", ColumnSelector(['pos', 'neu', 'neg']))
    ], n_jobs=-1)
    return make_pipeline(features, MultinomialNB())


def use_svm_pipeline():
    return make_pipeline(ColumnSelector(TEXT), USEEncoder(), LinearSVC(class_weight='balanced'))


def use_random_forest():
    return make_pipeline(ColumnSelector(TEXT), USEEncoder(),
                         RandomForestClassifier(n_estimators=10, random_state=1234, class_weight='balanced'))


def use_logistic():
    return make_pipeline(ColumnSelector(TEXT), USEEncoder(), LogisticRegression(class_weight='balanced', max_iter=2000))


def bow_xgboost():
    return make_pipeline(CountVectorizer(stop_words='english'),
                         xgb.XGBClassifier(random_state=1234, learning_rate=0.01))


def tf_idf_senti_xgboost():
    features = FeatureUnion([
        ("tfidf", make_pipeline(
            ColumnSelector(TEXT),
            TfidfVectorizer(stop_words='english')
        )),
        ("sentiment", ColumnSelector(['pos', 'neu', 'neg']))
    ], n_jobs=-1)
    return make_pipeline(features,
                         xgb.XGBClassifier(random_state=1234, learning_rate=0.01))


def tf_idf_nb_senti_xgboost():
    features = FeatureUnion([
        ("tfidf", make_pipeline(
            ColumnSelector(TEXT),
            nb_tf_idf(TfidfVectorizer(stop_words='english'))
        )),
        ("sentiment", ColumnSelector(['pos', 'neu', 'neg']))
    ], n_jobs=-1)

    return make_pipeline(features, xgb.XGBClassifier(random_state=1234, learning_rate=0.01))


# https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
# Credis to AlexSanchez https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb#261316
def nb_tf_idf(tf_idf_vect):
    return make_pipeline(tf_idf_vect, NBWeights())


def tf_idf_nb_senti_logistic():
    features = FeatureUnion([
        ("tfidf", make_pipeline(
            ColumnSelector(TEXT),
            nb_tf_idf(TfidfVectorizer(stop_words='english'))
        )),
        ("sentiment", ColumnSelector(['pos', 'neu', 'neg'])),
        ("hate_model", ColumnSelector(['hate_model']))
    ], n_jobs=-1)
    return make_pipeline(features, LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=2000))


class USEDenseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, pred_thresh=0.5):
        import tensorflow_hub as hub
        self.pred_thresh = 0.5
        encoder = USEEncoder()

        embedding_layer = hub.KerasLayer(encoder.load_model().model, output_shape=[encoder.output_dim], input_shape=[],
                                         dtype=tf.string, trainable=False)
        self._clf = tf.keras.Sequential([
            embedding_layer,
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        self._clf.summary()
        self._clf.compile(loss='binary_crossentropy', optimizer='adam')

    def predict(self, X):
        check_is_fitted(self, ['_clf'])
        return self._clf.predict(X) > self.pred_thresh

    def predict_proba(self, X):
        check_is_fitted(self, ['_clf'])
        return self._clf.predict(X)

    def fit(self, X, y):
        y = y.values

        es = EarlyStopping(monitor='val_loss')

        self._clf.fit(X, y, epochs=10, batch_size=1024, validation_split=0.2, callbacks=[es])

        return self


def use_dense_small():
    return USEDenseModel()


models = {
    'TF_IDF_SVM': tf_idf_svm_pipeline,
    'TF_IDF_LOGISTIC_SENTI': tf_idf_logistic_senti,
    'TF_IDF_LOGISTIC': tf_idf_logistic,
    'TF_IDF_NB_SENTI': tf_idf_nb_senti_logistic,
    'BOW_SVM': bow_svm,
    'BOW_LOGISTIC': bow_logistic,
    'BOW_LOGISTIC_SENTI': bow_logistic_senti,
    # 'USE_SVM': use_svm_pipeline,
    # 'USE_DENSE_SMALL': use_dense_small
    # 'TF_IDF_MILTI_MB': tf_idf_multi_nb,

    # 'BOW_XGBOOST': bow_xgboost,
    # 'TF_IDF_NB_SENTI_XGBOOST': tf_idf_nb_senti_xgboost,

    'USE_RANDOM_FOREST': use_random_forest,
    'USE_LOGISTIC': use_logistic,

}

if __name__ == '__main__':
    X = np.array(['This is cool', 'I am not happy with the content'])
    features = FeatureUnion([("tf_idf", TfidfVectorizer()),
                             ("sentiment", VaderSentiment())], n_jobs=-1)
    fe = features.fit_transform(X)

    print(fe)
