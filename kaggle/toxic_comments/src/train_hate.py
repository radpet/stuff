import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def train_hate_detector():
    train = pd.read_csv('./data/hate_speech.csv')

    print("Train shape is", train.shape)
    # .0 - hate
    # speech
    # 1 - offensive
    # language
    # 2 - neither

    print("Class values", train['class'].value_counts())
    train['neutral'] = train['class'] == 0
    train['hate_offensive'] = ~train['neutral']
    print(train.head(5)[['tweet', 'neutral', 'hate_offensive']])

    print('Offensive samples', train['hate_offensive'].sum(), 'out of', train.shape[0])

    train['tweet'] = train['tweet'].str.replace('!', '').replace(':', '').replace(
        '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '', regex=True)

    print(train.head(5)[['tweet', 'neutral', 'hate_offensive']])

    model = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000),
        LinearSVC(C=10)
    )

    train_X, test_X = train_test_split(train, test_size=0.1, stratify=train['class'])
    model.fit(train_X['tweet'], train_X['class'])
    # lets see if it can learn the train
    preds = model.predict(test_X['tweet'])

    print(confusion_matrix(y_pred=preds, y_true=test_X['class']))
    print(accuracy_score(y_pred=preds, y_true=test_X['class']))
    model.fit(train['tweet'], train['class'])

    with open('./data/hate_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Model saved')


if __name__ == '__main__':
    train_hate_detector()
