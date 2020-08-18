import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA

class DataToClassify:
    MODEL_TO_FIT  = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('logreg', RandomForestClassifier())
    ])
    LDA_PIPELINE = Pipeline([
        ('count', CountVectorizer()),
        ('lda', LDA(n_components=100,n_jobs=1))
    ])
    def __init__(self, data_to_classify):
        """
        Initialise a dataframe with the data to classify, 
        the labels, and the topics generated with the LDA
        """
        self.df = pd.DataFrame({
            'X': data_to_classify,
            'y': 0, 
            'score': None, 
            'topics': [list(el) for el in self.LDA_PIPELINE.fit_transform(data_to_classify)]
        })
        self.was_fitted = False
    
    def list_data(self, n=100):
        """
        list the 100 data which are the most likely to be of the category we look for
        """
        if not self.was_fitted:
            return list(self.df.X.sample(frac=1)[:n].iteritems())
        sorted_df = self.df.sort_values(by='score')
        return list(sorted_df.X[:n].iteritems())
    
    def print_data(self):
        """
        Helper to print the data returned by list_data
        """
        for ix, text in self.list_data():
            print(f'Index: {ix}')
            print(text)
            print('\n')
            
    def label_data(self, index):
        self.df.loc[index, 'y'] = 1
        
    @staticmethod
    def get_n_negative_data_to_sample(n_neg: int, n_pos: int):
        return n_pos * 10
    
    @staticmethod
    def get_negative_data(df):
        return df[df.y == 0]
    
    @staticmethod
    def get_positive_data(df):
        return df[df.y == 1]
    
    def sample_data_for_classifier(self):
        positive_data = self.get_positive_data(self.df)
        negative_data = self.get_negative_data(self.df)
        n_neg_data_to_sample = self.get_n_negative_data_to_sample(
            n_pos=len(positive_data),
            n_neg=len(positive_data),
        )
        sampled_negative_data = negative_data.sample(
            n_neg_data_to_sample
        )
        return positive_data.append(
            sampled_negative_data
        ).sample(frac=1)
    
    def update_score(self):
        sampled_data = self.sample_data_for_classifier()
        X = sampled_data.X
        y = sampled_data.y
        fitted_model = self.MODEL_TO_FIT.fit(X, y)
        self.df.score = fitted_model.predict_proba(self.df.X)[:, 1]
        self.was_fitted = True
        return self.df