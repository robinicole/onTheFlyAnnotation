import pandas as pd
import numpy as np

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA
class DataToClassifyBase:
    MODEL_TO_FIT: Pipeline
    LDA_PIPELINE : Pipeline
    
    @classmethod
    def process_data(cls, data: list) -> np.array:
        """
        Return a list of arrays of topics. 
        For example `[list(el) for el in self.LDA_PIPELINE.fit_transform(data)]`
        """
        raise NotImplementedError

    def __init__(self, data_to_classify: list):
        """
        Initialise a dataframe with the data to classify and preprocess them
        """
        self.df = pd.DataFrame({
            'X': data_to_classify,
            'y': 0, 
            'score': None, 
            'process_data': self.process_data(data_to_classify)
        })
        self.was_fitted = False
    
    def list_data(self, n=100) -> List[Tuple[int, str]]:
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
    def get_n_negative_data_to_sample(n_neg: int, n_pos: int) -> int:
        """
        Return the number of data from the negative class to sample 
        (y = 0) depending on the total number of positive classes (n_pos)
        and negative classes (n_neg) in the overall dataset
        """
        return n_pos * 10
    
    @staticmethod
    def get_negative_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a dataframe with only the negative class data
        """
        return df[df.y == 0]
    
    @staticmethod
    def get_positive_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a dataframe with only the positive class data
        """
        return df[df.y == 1]
    
    def sample_data_for_classifier(self) -> pd.DataFrame:
        """
        Get a sample of data to be fed to the classifier
        """
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
        X_sample = np.array([np.array(el) for el in sampled_data['process_data']])
        y = sampled_data.y
        fitted_model = self.MODEL_TO_FIT.fit(X_sample, y)
        X_data = np.array([np.array(el) for el in self.df['process_data']])
        self.df.score = fitted_model.predict_proba(X_data)[:, 1]
        self.was_fitted = True
        return self.df

class DataToClassifyStd(DataToClassifyBase):
    MODEL_TO_FIT: Pipeline  = RandomForestClassifier()
    LDA_PIPELINE : Pipeline = Pipeline([
        ('count', CountVectorizer()),
        ('lda', LDA(n_components=10,n_jobs=1))
    ])
    @classmethod
    def process_data(cls, data: list):
        return [list(el) for el in cls.LDA_PIPELINE.fit_transform(data)]
