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
    ts_stage = ts_df.loc[ts_df["stages"] == stage]
    ts_stage_grouped = ts_stage.groupby(
        ["pid", "night_date", (ts_df["stages"] != stage).cumsum()]
    )
    intervals_stage = ts_stage_grouped["t"].agg(["first", "last"])
    intervals_stage["duration"] = (
        intervals_stage["last"]
        - intervals_stage["first"]
        + pd.Timedelta(seconds=epoch_length)
    )
    return intervals_stage


def drop_awakenings_outside_delta_sleep(
    awakenings: pd.DataFrame, stats_per_night: pd.DataFrame
) -> pd.DataFrame:
    """Helper function for `summarise_stats_per_night`
    Drop the awakenings which appear before or after the defined sleep times
    i.e. Keep only awakenings starting after `first` non-AWAKE timestamp and
    ending before `last` non-AWAKE timestamp
    Best case scenario: there are no awakenings excluded
    Worst case scenario: there are 2 awakenings excluded

    Args:
        awakenings (pd.DataFrame): An interval dataframe containing awakenings
        stats_per_night (pd.DataFrame): A summary statistics dataframe

    Returns:
        pd.DataFrame: _description_
    """
    # the indices may not be equal, there may be nights with entirely empty nights
    # use left join instead
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


def summarise_stats_per_night(
    ts: pd.DataFrame, sleep_hour_start, sleep_hour_end, epoch_length=30
) -> pd.DataFrame:
    """Generate domain-knowledge-driven summary statistics per subject per night
    - note the difference between defined 'sleep times' and $\Delta t$

    | Column                       | Description                                                                                                  |
    |------------------------------|--------------------------------------------------------------------------------------------------------------|
    | `first`                      | the first non-AWAKE timestamp within the defined 'sleep times'
    | `last`                       | the last non-AWAKE timestamp within the defined 'sleep times'
    | `delta_sleep`                | `last` minus `first` + 1 epoch
    | `AWAKE`/`DEEP`/`LIGHT`/`REM` | number of **recorded** hours of AWAKE/DEEP/LIGHT/REM within the defined 'sleep times'
    | `time_bed`                   | total number of **recorded** hours within the defined 'sleep times'.
    | `total_sleep_time`           | total number of **recorded** non-awake hours within the defined 'sleep times'
    | `..._time_hour`              | `first`/`last` in terms of #hours from midnight
    | `TSTover10`                  | `total_sleep_time` over 10 hours
    | `efficiency`                 | percentage of `total_sleep_time` over `time_bed`.
    | `..._pct`                    | percentage of AWAKE/DEEP/LIGHT/REM/non-REM over `time_bed`.
    | `weekend`                    | Weekend (boolean). Whether deltaT starts on a Saturday or Sunday.
    | `fri_sat`                    | Friday_saturday (boolean). Whether deltaT starts on a Friday or Saturday (i.e. ending on weekends).
    | `missing`                    | `delta_sleep` minus `total_nonawake` ?????
    | `awake_..._...`              | Number of awakenings: >=30m/>5m; `excl`=excluding awakenings outsdie of `first` and `last`
    | `insomnia`                   | Middle insomnia (boolean), defined as `total_sleep_time` < 6h AND >= 1 prolonged awakening of >=30 minutes

    Args:
        ts (pd.DataFrame): Time series dataframe
        epoch_length (int, optional): Epoch length in seconds. Defaults to 30.

    Returns:
        pd.DataFrame: Summary statistics dataframe
    """
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
    hours_per_stage_per_night["time_bed"] = hours_per_stage_per_night.sum(axis=1)

    ### Combine them
    # the indices may not be equal, there may be nights with entirely empty nights
    # use left join instead
    combined = stats_per_night.join(hours_per_stage_per_night, how="left")

    ### onset/offset times
    start = pd.to_datetime(sleep_hour_start)
    start_hours = (start - start.normalize()).total_seconds() / 3600
    end = pd.to_datetime(sleep_hour_end)
    end_hours = (end - end.normalize()).total_seconds() / 3600

    onset = helper.time2second(combined["first"]) / 3600
    onset = [h + 24 if h <= end_hours else h for h in onset]
    combined["onset_time_hour"] = pd.Series(onset, index=combined.index)

    offset = helper.time2second(combined["last"]) / 3600
    offset = [h + 24 if h <= end_hours else h for h in offset]
    combined["offset_time_hour"] = pd.Series(offset, index=combined.index)

    ### Other stats
    combined["total_sleep_time"] = combined[["DEEP", "LIGHT", "REM"]].sum(axis=1)
    combined["TSTover10"] = combined["total_sleep_time"] > 10
    combined["efficiency"] = combined["total_sleep_time"] / combined["time_bed"] * 100

    ### Percentages
    combined["awake_pct"] = combined["AWAKE"] / combined["time_bed"] * 100
    combined["deep_pct"] = combined["DEEP"] / combined["time_bed"] * 100
    combined["light_pct"] = combined["LIGHT"] / combined["time_bed"] * 100
    combined["REM_pct"] = combined["REM"] / combined["time_bed"] * 100
    combined["NREM_pct"] = (
        (combined["DEEP"] + combined["LIGHT"]) / combined["time_bed"] * 100
    )

    ### Day of Week
    def flag_day_of_week(df, days_to_flag):
        # monday=0, sunday=6
        night_dates = df.index.get_level_values(level="night_date")
        night_dates = pd.to_datetime(pd.Series(night_dates, index=df.index))

        flag = night_dates.dt.dayofweek.isin(days_to_flag)
        return flag

    combined["weekend"] = flag_day_of_week(combined, [5, 6])
    combined["fri_sat"] = flag_day_of_week(combined, [4, 5])

    ### summarise awakenings
    awakenings = summarise_stage(ts, stage="AWAKE", epoch_length=epoch_length)
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

    ### summarise insomnia
    TSTbelow6 = combined["total_sleep_time"] < 6
    awake30atleast1 = combined["awake_30_excl"] >= 1
    combined["insomnia"] = TSTbelow6 & awake30atleast1

    ### output
    return combined
