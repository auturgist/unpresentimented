# NLTK tokenization and sentiment
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer




### SKLEARN
import sys
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectPercentile, chi2

from scipy.sparse import coo_matrix, hstack
import pandas as pd 

from ml_helpers import *



def get_data(level_key):
    data_dict ={'all':{'name':'realdonaldtrump.csv',
                        'desc': 'all the data',
                        'url':'./data/realdonaldtrump.csv'},
                'trim':{'name':'rdt_5trimmed.csv',
                        'desc':'First 500 tweets',
                        'url':'./data/rdt_5trimmed.csv'
                        }}
    

    url = './data/realdonaldtrump.csv'
    return pd.read_csv(data_dict[level_key]['url'])

def sample_data(df):
    odf = df.iloc[:500]
    print(len(odf))
    odf.to_csv('./data/rdt_5trimmed.csv')

if __name__ == "__main__":
    ## Get Data
    df = get_data('trim')
    ##
    pipe = Pipeline([        
                ('text', ColumnExtractor('text')),
                ('vect', SentimentAnalyzer()),
                ('dense', DenseTransformer())
                ])
    percentile_feats=10
    select = SelectPercentile(chi2, percentile=percentile_feats)
    
    # pipe = make_pipeline(sentiment_pipeline)
    # print('downsample')

    
    out = pipe.fit_transform(df)
    out = pd.DataFrame(out, columns=['neg','neu','pos'])
    df = pd.concat([df,out])
    # print(pipe)
    print(df.columns)
    # print(out)

    
    




