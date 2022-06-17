from random import normalvariate
import pandas as pd
import numpy as np
import os
import logging


def report_change_in_nrow(before, after, operation=None):
    n_before = before.shape[0]
    n_after = after.shape[0]
    n_change = n_after - n_before
    pc_remain = round(n_after / n_before * 100, 2)

    msg = f"{n_before}->{n_after} rows (Change = {n_change}) (nrow after = {pc_remain}% of before)"
    if operation is not None:
        msg = "# " + operation + "\n" + msg

    print(msg)
    return None


def report_train_test_split(X_train, X_test, y_train, y_test):
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Training Labels Shape: {y_train.shape}")
    print(f"Testing Features Shape: {X_test.shape}")
    print(f"Testing Labels Shape: {y_test.shape}")
    return None


def report_preprocessed(df2):

    id_target_tuples = df2[["id_new", "target"]].drop_duplicates()

    subjects_by_target = id_target_tuples.value_counts("target")
    subjects_by_target_pc = id_target_tuples.value_counts("target", normalize=True)

    hours_by_target = df2.value_counts("target")
    hours_by_target_pc = df2.value_counts("target", normalize=True).apply(
        lambda x: "{:.2%}".format(x)
    )

    print(f"df shape: {df2.shape}")
    print(f"Number of subjects by target: {subjects_by_target}")
    print(f"Proportion of subjects by target: {subjects_by_target_pc}")
    print(f"Hours by target: {hours_by_target}")
    print(f"Hours by target (%): {hours_by_target_pc}")

    hours_by_target_and_id = df2.groupby("target")["id_new"].value_counts()
    return hours_by_target_and_id


def combine_classification_reports(report_dicts, names):

    results = []
    for report_dict, name in zip(report_dicts, names):
        df = pd.DataFrame(report_dict).transpose().reset_index(level=0)
        df.loc[df["index"] == "accuracy", ["precision", "recall"]] = None

        support = df.loc[df["index"] == "macro avg", "support"].values
        df.loc[df["index"] == "accuracy", "support"] = support

        df["name"] = name

        df = df[["name", "index", "precision", "recall", "f1-score", "support"]]
        results.append(df)

    combined = pd.concat(results)
    return combined


def search2df(search):
    """Convert search results to dataframe

    Args:
        search (sklearn.model_selection._search subclasses): Subclasses of the sklearn _search class

    Returns:
        pd.DataFrame: Dataframe
    """
    rs = search.cv_results_
    df_params = pd.DataFrame(rs["params"])

    cols = [col for col in rs.keys() if "param" not in col]
    dict_metrics = {k: rs[k] for k in cols if k in rs}
    df_metrics = pd.DataFrame(dict_metrics)

    df_output = pd.concat([df_params, df_metrics], axis=1)
    return df_output


def cv_results2df(cv_results: list, names: list, prefix=""):
    """Convert a list of cross validation results to one dataframe.

    Args:
        cv_results (list): List of cross validation result dictionaries
        names (list): List of names as strings
        prefix (str, optional): Prefix to strip in column names. Defaults to empty for pivoting later.
    """
    cv_results_dfs = [pd.DataFrame(result) for result in cv_results]
    out = pd.concat(cv_results_dfs, keys=names, names=["name", "split"]).reset_index()

    def remove_suffix(line, prefix):
        if line.startswith(prefix):
            return line[len(prefix) :]
        return line

    out.columns = [remove_suffix(f, prefix) for f in out.columns]
    return out
