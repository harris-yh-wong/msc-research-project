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
