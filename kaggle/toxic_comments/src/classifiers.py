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

from features.nbweights import NBWeights
from features.sentiment import VaderSentiment
from features.use_embeddings import USEEncoder


def tf_idf_svm_pipeline():
    return make_pipeline(TfidfVectorizer(), LinearSVC())


def tf_idf_logistic():
    return make_pipeline(TfidfVectorizer(stop_words='english'),
                         LogisticRegression(C=5, class_weight='balanced', n_jobs=-1, max_iter=1000))


def bow_logistic():
    return make_pipeline(CountVectorizer(stop_words='english'),
                         LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=1000))


def tf_idf_multi_nb():
    return make_pipeline(TfidfVectorizer(), MultinomialNB())


def bow_multi_nb():
    return make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB())


def use_svm_pipeline():
    return make_pipeline(USEEncoder(), LinearSVC())


def use_random_forest():
    return make_pipeline(USEEncoder(),
                         RandomForestClassifier(n_estimators=10, random_state=1234, class_weight='balanced'))


def use_logistic():
    return make_pipeline(USEEncoder(), LogisticRegression(class_weight='balanced', max_iter=1000))


def bow_xgboost():
    return make_pipeline(CountVectorizer(stop_words='english'),
                         xgb.XGBClassifier(random_state=1234, learning_rate=0.01))


def tf_idf_xgboost():
    return make_pipeline(TfidfVectorizer(),
                         xgb.XGBClassifier(random_state=1234, learning_rate=0.01))


def tf_idf_nb_senti_xgboost():
    features = FeatureUnion([("tf_idf", nb_tf_idf(TfidfVectorizer(ngram_range=(1, 2),
                                                                  min_df=3, max_df=0.9, strip_accents='unicode',
                                                                  stop_words='english', use_idf=1,
                                                                  smooth_idf=1, sublinear_tf=1, max_features=10000))),
                             ("sentiment", VaderSentiment())], n_jobs=-1)

    return make_pipeline(features, xgb.XGBClassifier(random_state=1234, learning_rate=0.01))


# https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
# Credis to AlexSanchez https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb#261316
def nb_tf_idf(tf_idf_vect):
    return make_pipeline(tf_idf_vect, NBWeights())


def tf_idf_nb():
    return make_pipeline(nb_tf_idf(TfidfVectorizer(ngram_range=(1, 2),
                                                   min_df=3, max_df=0.9, strip_accents='unicode', stop_words='english',
                                                   use_idf=1,
                                                   smooth_idf=1, sublinear_tf=1, max_features=10000)),
                         LogisticRegression(max_iter=1000))


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


def tf_idf_nb_sent():
    features = FeatureUnion([("tf_idf", nb_tf_idf(TfidfVectorizer(ngram_range=(1, 2),
                                                                  min_df=3, max_df=0.9, strip_accents='unicode',
                                                                  stop_words='english', use_idf=1,
                                                                  smooth_idf=1, sublinear_tf=1, max_features=10000))),
                             ("sentiment", VaderSentiment())], n_jobs=-1)

    return make_pipeline(features, LogisticRegression(max_iter=1000))


models = {
    # 'TF_IDF_SVM': tf_idf_svm_pipeline,
    # 'COUNT_SVM': tf_idf_svm_pipeline,
    # 'USE_SVM': use_svm_pipeline,
    # 'USE_DENSE_SMALL': use_dense_small
    'TF_IDF_MILTI_MB': tf_idf_multi_nb,
    'TF_IDF_LOGISTIC': tf_idf_logistic,
    'BOW_XGBOOST': bow_xgboost,
    'TF_IDF_XGBOOST': tf_idf_xgboost,
    'TF_IDF_NB': tf_idf_nb,
    'BOW_LOGISTIC': bow_logistic,
    'BOW_MULTI_NB': bow_multi_nb,
    # 'USE_RANDOM_FOREST': use_random_forest,
    'USE_LOGISTIC': use_logistic,
    'TF_IDF_NB_SENTI': tf_idf_nb_sent,
    'TF_IDF_NB_SENTI_XGBOOST': tf_idf_nb_senti_xgboost
}

if __name__ == '__main__':
    X = np.array(['This is cool', 'I am not happy with the content'])
    features = FeatureUnion([("tf_idf", TfidfVectorizer()),
                             ("sentiment", VaderSentiment())], n_jobs=-1)
    fe = features.fit_transform(X)

    print(fe)
