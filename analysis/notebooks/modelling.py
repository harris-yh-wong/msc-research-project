import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

from typing import Tuple
from clean_data import split_id_new


def train_test_split_grouped(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
):
    groups = split_id_new(pd.Series(y.index))["pid"]
    splitter = GroupShuffleSplit(
        test_size=test_size, n_splits=1, random_state=random_state
    )
    split = splitter.split(y, groups=groups)
    train_index, test_index = next(split)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    return X_train, X_test, y_train, y_test


def validate_grouped_split(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    setA = set(X_train.index.str.split("_", expand=True).get_level_values(0))
    setB = set(X_test.index.str.split("_", expand=True).get_level_values(0))
    setC = set(y_train.index.str.split("_", expand=True).get_level_values(0))
    setD = set(y_test.index.str.split("_", expand=True).get_level_values(0))

    print(f"Index match in training set: {setA == setC}")
    print(f"Index match in testing set:  {setB == setD}")
    print(
        f"Non -overlapping index in training vs testing: {setA.intersection(setB) == set()}"
    )
