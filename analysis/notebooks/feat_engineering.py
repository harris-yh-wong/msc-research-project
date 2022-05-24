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
    

def plot_relevant_features(relevance_table: pd.DataFrame, n_features=10) -> None:
    """Plot bar chart of relevant features from relevance table from tsfresh

    Args:
        relevance_table (pd.DataFrame): Relevance table output from tsfresh
        n_features (int, optional): Number of feature sto plot. Defaults to 10.
    """
    n_features = 50
    relevance_table['stage'] = relevance_table['feature'].str[:5]
    relevance_table['-logp'] = np.log10(relevance_table['p_value'])*-1

    show_features = get_relevant_features(relevance_table)[0:n_features]
    relevance_table_subset = relevance_table.loc[relevance_table['feature'].isin(show_features), ]

    sns.set(rc={'figure.figsize':(12,n_features/2)})
    sns.barplot(y = 'feature', x = '-logp', hue='stage', data=relevance_table_subset);

