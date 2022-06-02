import seaborn as sns
import pandas as pd
import numpy as np

import helper


def get_relevant_features(relevance_table: pd.DataFrame) -> list:
    """Get a list of relevant features from relevance table from tsfresh

    Args:
        relevance_table (pd.DatFrame): Relevance table output from tsfresh feature

    Returns:
        features (list): Relevant features by the criteria of significant hypothesis test by tsfresh
    """

    features = sorted_shortlist = (
        relevance_table.sort_values("p_value", inplace=False).loc[
            relevance_table["relevant"] == True, "feature"
        ]
    ).tolist()
    return features


def summarise_stage(ts_df: pd.DataFrame, stage: str, epoch_length=30) -> pd.DataFrame:
    """Summarise awakenings from time series dataframe into intervals

    Args:
        ts_df (pd.DataFrame): Time series dataframe.
        stage (str): Stage to summarise. "AWAKE", "DEEP", "LIGHT", "REM" or "UNKNOWN".
        epoch_length (int, optional): Epoch length (in seconds). Defaults to 30.

    Returns:
        pd.DataFrame: Interval dataframe of intervals per pid per night
    """
    assert stage in ["AWAKE", "DEEP", "LIGHT", "REM", "UNKNOWN"], "Invalid stage."

    ### Convert time series to interval data

    # AWAKE stages are grouped into 'consecutive' groups
    # then take the first and last timestamp
    ts_awake = ts_df.loc[ts_df["stages"] == stage]
    ts_awake_grouped = ts_awake.groupby(
        ["pid", "night_date", (ts_df["stages"] != stage).cumsum()]
    )
    awakenings = ts_awake_grouped["t"].agg(["first", "last"])
    awakenings["duration"] = (
        awakenings["last"] - awakenings["first"] + pd.Timedelta(seconds=epoch_length)
    )
    return awakenings


def drop_awakenings_outside_delta_sleep(
    awakenings: pd.DataFrame, stats_per_night: pd.DataFrame
) -> pd.DataFrame:
    """Drop the awakenings which appear before or after the
    i.e. Keep only awakenings starting after `first` non-AWAKE timestamp and
    ending before `last` non-AWAKE timestamp
    best case scenario: there are no awakenings excluded
    Worst case scenario: there are 2 awakenings excluded

    Args:
        awakenings (pd.DataFrame): An interval dataframe containing awakenings
        stats_per_night (pd.DataFrame): A summary statistics dataframe

    Returns:
        pd.DataFrame: _description_
    """
    assert awakenings.index.equals(stats_per_night.index)

    awakenings_joined = awakenings.join(
        stats_per_night[["first", "last"]], how="left", lsuffix="", rsuffix="_sleep"
    )
    keep_flag = (awakenings_joined["first"] >= awakenings_joined["first_sleep"]) & (
        awakenings_joined["last"] <= awakenings_joined["last_sleep"]
    )
    awakenings_excl = awakenings.loc[
        keep_flag,
    ]
    return awakenings_excl


def summarise_stats_per_night(ts: pd.DataFrame, epoch_length=30) -> pd.DataFrame:

    ### Basic aggregates per night
    stats_per_night = (
        ts.loc[
            ts["stages"] != "AWAKE",
        ]
        .groupby(["pid", "night_date"])["t"]
        .agg(["first", "last"])
    )

    ### Counts of hours per stage per night
    hours_per_stage_per_night = (
        (ts.groupby(["pid", "night_date", "stages"])["t"].size().unstack(fill_value=0))
        * epoch_length
        / 3600
    )
    hours_per_stage_per_night["total"] = hours_per_stage_per_night.sum(axis=1)

    ### Combine them
    assert stats_per_night.index.equals(hours_per_stage_per_night.index)
    combined = stats_per_night.join(hours_per_stage_per_night, how="left")

    ### Other stats
    combined["total_sleep_time"] = combined[["DEEP", "LIGHT", "REM"]].sum(axis=1)
    combined["offset_time_hour"] = helper.time2second(combined["last"]) / 3600
    combined["TSTover10"] = combined["total_sleep_time"] > 10
    combined["awake_pct"] = combined["AWAKE"] / combined["total"] * 100

    ### summarise awakenings
    awakenings = summarise_stage(ts, epoch_length, stage="AWAKE")
    ### drop awakenings that are outside the delta sleep
    awakenings_excl = drop_awakenings_outside_delta_sleep(awakenings, combined)

    ### summarise statistics on awakenings
    combined["awake_5"] = (
        (awakenings["duration"] > pd.Timedelta(minutes=5))
        .groupby(["pid", "night_date"])
        .sum()
    )
    combined["awake_30"] = (
        (awakenings["duration"] >= pd.Timedelta(minutes=30))
        .groupby(["pid", "night_date"])
        .sum()
    )
    combined["awake_5_excl"] = (
        (awakenings_excl["duration"] > pd.Timedelta(minutes=5))
        .groupby(["pid", "night_date"])
        .sum()
    )
    combined["awake_30_excl"] = (
        (awakenings_excl["duration"] >= pd.Timedelta(minutes=30))
        .groupby(["pid", "night_date"])
        .sum()
    )

    return combined
