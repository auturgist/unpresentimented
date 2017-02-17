from sklearn.base import BaseEstimator, TransformerMixin
# NLTK tokenization and sentiment
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.pipeline import Pipeline, FeatureUnion, _transform_one
from sklearn.externals.joblib import Parallel, delayed

class DFFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        # non-optimized default implementation; override when a better
        # method is possible
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, weight, X)
            for name, trans, weight in self._iter())
        return pd.concat(Xs, axis=1, join='inner')

class NoFitMixin:
    def fit(self, X, y=None):
        return self

class DFTransform(TransformerMixin, NoFitMixin):
    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Extract a column from a pandas.DataFrame by name in feature Pipeline.
        
        Use at the start of a feature pipeline to extract the column of interest from the original dataset.
    """
    def __init__(self, column=0):
       
        self.column = column

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

    def transform(self, X, **kwargs):
        if isinstance(self.column, str):
            return X[self.column]
        elif isinstance(self.column, int):
            return X[:,self.column]


    def fit(self, X, y=None, **kwargs):
        return self

class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    """Returns matrix of positive,negative, and neutral sentiments. Uses Vader lexicon


    Data: expects dataframe or matrix with text to be analyzed 
    """
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        
        sid = SentimentIntensityAnalyzer()
        num_docs =  len(X)

        
        neg_sentiments = np.zeros((num_docs, 1), dtype=np.float)
        neu_sentiments = np.zeros((num_docs, 1), dtype=np.float)
        pos_sentiments = np.zeros((num_docs, 1), dtype=np.float)
        
        for i in range(0, num_docs):
            line = X.iloc[i]
            try:

                sentiment_aggregate = sid.polarity_scores(line) 
                # print(sentiment_aggregate)
                neg_sentiment = sentiment_aggregate['neg']
                # print("neg_sentiment: {}".format(neg_sentiment)),
                
                pos_sentiment = sentiment_aggregate['pos']
                neu_sentiment = sentiment_aggregate['neu']
            except(TypeError): # in event line is nan (likely due to clrf to rf switches)
                 neg_sentiment = pos_sentiment = neu_sentiment =0
            # print("neg_sentiment: {}".format(neg_sentiment)),    
            neg_sentiments[i] = neg_sentiment
            # print("neg_sentiments: {}".format(neg_sentiments[i]))    
            neu_sentiments[i] = neu_sentiment
            pos_sentiments[i] = pos_sentiment
            # print(pos_sentiment)
            
        negative_sent = coo_matrix(neg_sentiments)
        neutral_sent = coo_matrix(neu_sentiments)
        positive_sent = coo_matrix(pos_sentiments)
        
        
        return hstack([negative_sent, neutral_sent, positive_sent])


