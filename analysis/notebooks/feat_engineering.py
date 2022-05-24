import seaborn as sns
import pandas as pd
import numpy as np


def get_relevant_features(relevance_table: pd.DataFrame) -> list:
    """Get a list of relevant features from relevance table from tsfresh

    Args:
        relevance_table (pd.DatFrame): Relevance table output from tsfresh feature

    Returns:
        features (list): Relevant features by the criteria of significant hypothesis test by tsfresh 
    """
    
    features = sorted_shortlist = (relevance_table
        .sort_values('p_value', inplace=False)
        .loc[relevance_table['relevant']==True, 'feature']
    ).tolist()
    return features
    
