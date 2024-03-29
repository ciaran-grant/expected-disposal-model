from expected_disposal_model.data_preparation.preprocessing import convert_chains_to_schema, create_gamestate_features, filter_disposals
from expected_disposal_model.modelling_data_contract import ModellingDataContract

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

class Preprocessor(BaseEstimator, TransformerMixin):
    """ Preprocessing class and functions for training inside50 model
    """
    
    def __init__(self):
        self.ModellingDataContract = ModellingDataContract
       
        
    def fit(self, X):

        # Keep only modelling columns
        self.modelling_cols = ModellingDataContract.feature_list
                        
        return self
    
    def transform(self, X):
        """ Applies transformations and preprocessing steps to dataframe.

        Args:
            X (Dataframe): Training or unseen data to transform.

        Returns:
            Dataframe: Transformed data with modelling columns and no missing values.
        """

        # Get schema
        X_schema = convert_chains_to_schema(X)
        # Get features
        X_features = create_gamestate_features(X_schema)
        
        X_schema = pd.concat([X_schema, X_features], axis='columns')
        X_schema = filter_disposals(X_schema)
        
        # X_features = X_schema[ModellingDataContract.feature_list]
                
        return X_schema
    