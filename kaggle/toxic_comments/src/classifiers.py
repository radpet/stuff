from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features.use_embeddings import USEEncoder


def tf_idf_svm_pipeline():
    return make_pipeline(TfidfVectorizer(), LinearSVC())


def tf_idf_logistic():
    return make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(C=5,class_weight='balanced'))



def bow_logistic():
    return make_pipeline(CountVectorizer(stop_words='english'),LogisticRegression(class_weight='balanced'))


def tf_idf_multi_nb():
    return make_pipeline(TfidfVectorizer(), MultinomialNB())


def use_svm_pipeline():
    return make_pipeline(USEEncoder(), LinearSVC())


def use_random_forest():
    return make_pipeline(USEEncoder(),
                         RandomForestClassifier(n_estimators=10, random_state=1234, class_weight='balanced', n_jobs=-1))


def use_logistic():
    return make_pipeline(USEEncoder(), LogisticRegression(class_weight='balanced'))


models = [
    # ('TF_IDF_SVM', tf_idf_svm_pipeline),
    # ('COUNT_SVM', use_svm_pipeline),
    # ('TF_IDF_MILTI_MB', tf_idf_multi_nb),
    # ('USE_SVM', use_svm_pipeline),
    ('TF_IDF_LOGISTIC', tf_idf_logistic),
    ('BOW_LOGISTIC', bow_logistic),
    ('USE_RANDOM_FOREST', use_random_forest),
    # ('USE_SVM', use_svm_pipeline()),
    ('USE_LOGISTIC', use_logistic)
]
