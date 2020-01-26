import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderSentiment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.load_model()
        print('Feature extraction via VaderSentiment')
        scores = []

        for i, _ in enumerate(range(0, len(X))):
            sentence = X[i]
            polarity_scores = self.model.polarity_scores(sentence)
            scores.append(
                np.array([polarity_scores['pos'], polarity_scores['neu'], polarity_scores['neg']]))
        print('Feature extraction via VaderSentiment done')

        return np.array(scores)

    def load_model(self):
        if self.model is None:
            self.model = SentimentIntensityAnalyzer()
        return self


VADER_SENTIMENT_COL = 'vaderSentiment'


class VaderSentimentWithMem(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None
        self.cache = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.load_model()
        print('Feature extraction via VaderSentiment')
        scores = []

        for i, _ in enumerate(range(0, len(X))):
            sentence = X[i]

            if sentence in self.cache:
                polarity_scores = self.cache[sentence]
            else:
                polarity_scores = self.model.polarity_scores(sentence)
                self.cache[sentence] = polarity_scores

            scores.append(
                    np.array([polarity_scores['pos'], polarity_scores['neu'], polarity_scores['neg']]))
        print('Feature extraction via VaderSentiment done')

        return np.array(scores)

    def load_model(self):
        if self.model is None:
            self.model = SentimentIntensityAnalyzer()
        return self

if __name__ == '__main__':
    vader = VaderSentiment()
    X = np.array(['This is cool', 'I am not happy with the content'])
    sentiments = vader.transform(X)
    print(sentiments)
