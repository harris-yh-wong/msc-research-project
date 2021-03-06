import os
from re import A
from telnetlib import TSPEED
import numpy as np
import pandas as pd
import datetime as dt
from helper import *
import report
from datetime import datetime

from feat_engineering import summarise_stage

# from pandarallel import pandarallel  # parallel processing


def sample_df(df, SAMPLE_PERCENT):
    if SAMPLE_PERCENT == 100:
        return df
    keep_flag = int(np.floor(df.shape[0] * SAMPLE_PERCENT * 0.01))
    sampled = df.head(keep_flag)
    return sampled


def clean_all_phq(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = ["pid", "timestamp", "phq", "sleep"]
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    return df2


def clean_slps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sleep stages by cleaning time formats and creating columns of start/end times
    """
    df2 = df.copy()
    df2["start"] = pd.to_datetime(df2.time)
    df2["start_date"] = df2["start"].dt.date
    df2["end"] = df2["start"] + pd.to_timedelta(df2["duration"] - 30, "s")
    df2["intervalID"] = np.arange(len(df2))
    df2.drop(columns="time")
    return df2


def explode2ts(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Explode a dataframe in interval format to time series format

    Args:
        ts_df (pd.DataFrame): Dataframe in interval format with 'start' and 'end' columns

    Returns:
        pd.DataFrame: Dataframe in time series format with epoch timestamp
    """
    # the start_date is kept because overlap-removal step requires this column for deciding which label to keep/discard
    data = (
        (row.pid, t, row.start, row.stages)
        for row in ts_df.itertuples()
        for t in pd.date_range(row.start, row.end, freq="30s")
    )
    ts = pd.DataFrame(data=data, columns=["pid", "t", "interval_start", "stages"])
    return ts


def time2datetime(t: dt.time) -> dt.datetime:
    """Convert time to a datetime object

    Args:
        t (dt.time): time

    Returns:
        dt.datetime: datetime object
    """
    date = dt.date(1970, 1, 1)
    return dt.datetime.combine(date, t)


def time2seconds(t: dt.time) -> float:
    """Convert time to seconds from midnight
    Note:
    alternatively,
    > freq = '1s'
    > ((time - time.dt.normalize()) / pd.Timedelta(freq)

    Args:
        t (dt.time): time

    Returns:
        int: seconds from midnight
    """
    delta = time2datetime(t) - dt.datetime(1970, 1, 1)
    return delta.total_seconds()


def subset_timeseries_within_interval(
    timeseries: pd.DataFrame, start: dt.date, end: dt.date, inclusive="both"
) -> pd.DataFrame:
    subset = (
        timeseries.set_index("t", inplace=False)
        .between_time(start, end, inclusive=inclusive)
        .reset_index(inplace=False)
    )
    report.report_change_in_nrow(timeseries, subset)
    return subset


def dedup_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate timeseries. Logic: grouped by subject id and timestamp, sort by interval start datetime, select last record

    Args:
        ts (pd.DataFrame): time series dataframe

    Returns:
        pd.DataFrame: time series dataframe, deduplicated
    """
    dedupped = (
        ts.sort_values(["pid", "t", "interval_start"], inplace=False)
        .groupby(["pid", "t"])
        .tail(1)
    )
    report.report_change_in_nrow(ts, dedupped)
    return dedupped


def bin_by_hour(ts: pd.DataFrame) -> pd.DataFrame:
    """DEPRECATED. Bin sleep stage time series by hour

    Args:
        ts (pd.DataFrame): time series dataframe
    """
    print("`bin_by_hour` is deprecated, use the more flexible `bin_by_time` instead.")

    df = ts.copy()
    df["hour"] = df["t"].dt.hour
    df = pd.concat([df, pd.get_dummies(df["stages"])], axis=1)
    binned = df.groupby(["pid", "start_date", "hour"])[
        ["AWAKE", "LIGHT", "DEEP", "REM"]
    ].agg("sum")
    binned["sum"] = binned.sum(axis=1)
    binned = binned.reset_index(inplace=False)
    return binned


def bin_by_time(ts: pd.DataFrame, freq="H") -> pd.DataFrame:
    """Bin time series by the time.
    Numbers correspond to the count of signals (each 30s) within the bin.
    Expected total = bin size / time series epoch,
    e.g. expecteded total for hourly binning = 1h / 30s = 120
    The 'sum' column may add up to less than the expected total due to missingness, but never above.

    Args:
        ts (pd.DataFrame): Time series dataframe
        freq (str, optional): Bin size to bin by. Defaults to "H".

    Returns:
        pd.DataFrame: Binned dataframe
    """

    df = ts.copy()

    ### binning
    stages = ["AWAKE", "LIGHT", "DEEP", "REM"]
    df["stages"] = pd.Categorical(df["stages"], categories=stages, ordered=False)
    df = pd.concat([df, pd.get_dummies(df["stages"])], axis=1)

    group_by = ["pid", pd.Grouper(key="t", freq=freq)]
    binned = df.groupby(group_by)[stages].agg("sum")
    binned.reset_index(inplace=True)

    ### annotation
    # Bin ID
    # Here, 'hour' refers to the bin ID -- the Nth bin from midnight.
    # e.g., if binning every 30 minutes, 8pm corresponds to the 20th bin from midnight.
    #! refractor later, including all downstream routines
    if freq == "H":
        binned["hour"] = binned["t"].dt.hour
    else:
        binned["hour"] = (
            (binned["t"] - binned["t"].dt.normalize()) / pd.Timedelta(freq)
        ).astype(int)

    binned["start_date"] = binned["t"].dt.date
    binned.drop("t", axis=1, inplace=True)

    ###
    binned["sum"] = binned[stages].sum(axis=1)

    report.report_change_in_nrow(ts, binned)
    return binned


def expand_full_hours(df: pd.DataFrame, hours=None, by_columns=None) -> pd.DataFrame:
    """Expand to hourly binned dataframe to a full list of hours (user-supplied, defaults to 8pm to 10am),
    fill with NAs if empty.
    Grouped by the columns specified in the 'by' option

    Args:
        df (pd.DataFrame): hourly binned dataframe
        by_columns (list): By what columns to bin. Defaults to 'pid' and 'start_date'.
        hours (list): If None, Defaults to 8pm-10am i.e. [20,21,22,23,0,1,2,3,4,5,6,7,8,9,10].

    Returns:
        pd.DataFrame: hourly binned dataframe (expanded)
    """
    if by_columns is None:
        by_columns = ["pid", "start_date"]
    if hours is None:
        hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #! inefficient approach, fix later
    pid_date_combinations = df[by_columns].drop_duplicates().reset_index(drop=True)
    nrow = pid_date_combinations.shape[0]
    hours_rep = [hours for _ in range(0, nrow)]
    hours_series = pd.Series(hours_rep, name="hour").reset_index(drop=True)
    fulldf = pd.concat([pid_date_combinations, hours_series], axis=1).explode("hour")

    on_columns = by_columns + ["hour"]
    expanded = pd.merge(fulldf, df, on=on_columns, how="left")
    expanded.fillna(0, inplace=True)

    report.report_change_in_nrow(df, expanded)
    return expanded


def normalize_binned(
    df: pd.DataFrame, over=None, freq="H", epoch_length=30
) -> pd.DataFrame:
    """Normalize binned dataframe by 120

    Args:
        df (pd.DataFrame): binned dataframe
        over (int): [Deprecated] normalize over what? Defaults to 120 (number of 30-second intervals in an hour)
        freq (str): Bin size
        epoch_length (int): Epoch length (in seconds). Defaults to 30

    Returns:
        pd.DataFrame: Normalized dataframe
    """

    epochs_per_bin = pd.Timedelta(freq).total_seconds() / epoch_length

    binned = df.copy()
    binned["AWAKE"] = binned["AWAKE"] / epochs_per_bin
    binned["LIGHT"] = binned["LIGHT"] / epochs_per_bin
    binned["DEEP"] = binned["DEEP"] / epochs_per_bin
    binned["REM"] = binned["REM"] / epochs_per_bin
    if "sum" in binned.columns:
        binned["sum"] = binned["sum"] / epochs_per_bin

    return binned


#### HAOTIAN's


def drop_days_delta(target_select: pd.DataFrame, threshold=14) -> pd.DataFrame:
    """
    Drop PHQ test records which is <14 days from previous result
    The logic is to take the last record
    Author: Haotian Gao
    """

    # make copy
    tgt = target_select.copy()

    # clean
    tgt["date"] = pd.to_datetime(tgt["test_date"])
    tgt = tgt.sort_values(["id", "date"])

    # calculate number of days from previous record
    target_date_diff = tgt[["id", "date"]].groupby("id").diff()

    # combine with main df
    target_date_diff["daysdelta"] = target_date_diff["date"] / np.timedelta64(1, "D")
    target_select_clean = pd.concat([tgt, target_date_diff["daysdelta"]], axis=1)

    # filter
    flag_pass_day_threshold = target_select_clean["daysdelta"] >= threshold
    flag_first_test_result = target_select_clean["daysdelta"].isna()
    flag_keep = flag_pass_day_threshold | flag_first_test_result
    target_select_clean_filtered = target_select_clean.loc[flag_keep, :]

    # clean output
    out = target_select_clean_filtered.drop(["date", "daysdelta"], axis=1)
    report.report_change_in_nrow(
        tgt,
        out,
        operation="Drop PHQ test results which are within 14 days from previous questionnaire",
    )

    return out


def generate_ts_y(
    df: pd.DataFrame,
    column_id="id_new",
    column_index="index",
    column_features=["AWAKE", "LIGHT", "DEEP", "REM"],
):

    ### Generate time series (many rows for each y label) and corresponding . The ID column in the time series dataframe

    col_select = column_features + [column_index, column_id, "target"]
    input_df = df[col_select].copy()

    ts = input_df.reset_index(drop=True)
    ts.drop("target", axis=1, inplace=True)
    y = input_df.groupby(column_id).tail(1).set_index(column_id)["target"]

    ts_ind = set(ts[column_id])
    y_ind = set(y.index)
    assert ts_ind == y_ind, "ts and y features don't match"
    return ts, y


def merge_slp_phq(
    expanded: pd.DataFrame, phqs_raw: pd.DataFrame, window=15
) -> pd.DataFrame:
    """Merge raw PHQ test results dataframe with 'expanded' binned sleep stages dataframe,
    grouped by 'deltaTs'

    Args:
        expanded (pd.DataFrame): Sleep stages dataframe, binned+expanded to a full list of hours
        phqs_raw (pd.DataFrame): PHQ test results dataframe
        window (int): 'Traceback' window size of 'deltaT'. Defaults to 15.

    Returns:
        pd.DataFrame: merged dataframe
    """

    ### CLEAN phqs_raw to match Haotian's code
    target = phqs_raw[["centre", "pid", "time", "phq"]].copy()
    target.columns = ["centre", "id", "time_y", "phq"]

    target["time_y"] = pd.to_datetime(target["time_y"])

    # Clean target data
    """target["test_date"] = target["time_y"].map(lambda x: x[:10])"""
    #! replaced by:
    target["test_date"] = target["time_y"].dt.date
    target_new = target.loc[:, ["id", "test_date", "phq"]]

    # Get observation start and end times
    """
    id_obs_start = data_tab.groupby('id')['date'].min()
    id_obs_end = data_tab.groupby('id')['date'].max()
    """

    #! replaced by
    expanded.columns = [
        "id",
        "date",
        "start_time",
        "AWAKE",
        "DEEP",
        "LIGHT",
        "REM",
        "total",
    ]
    id_obs_start = expanded.groupby("id")["date"].min()
    id_obs_end = expanded.groupby("id")["date"].max()

    # Adjust the time format
    target_new["obs_start"] = pd.to_datetime(target_new.test_date) - pd.Timedelta(
        days=window
    )

    target_new["obs_start"] = target_new["obs_start"].apply(
        lambda x: x.strftime("%Y-%m-%d")
    )

    # Join the tables
    target_new = target_new.merge(id_obs_start, on="id", how="left").merge(
        id_obs_end, on="id", how="left"
    )

    # Filter the target data during the observation period
    target_select = target_new.loc[
        (
            pd.to_datetime(target_new["test_date"])
            >= pd.to_datetime(target_new["date_x"])
        )
        & (
            pd.to_datetime(target_new["obs_start"])
            <= pd.to_datetime(target_new["date_y"])
        )
    ]
    # Drop extra columns
    target_select = target_select.drop(["date_x", "date_y"], axis=1)

    # Drop PHQs records within 14 days of the previous
    target_select = drop_days_delta(target_select)

    # Link target data and observation data

    ### outer join
    """data_merge = pd.merge(target_select, data_tab, how='outer', on=['id'])"""
    #! replaced by
    data_merge = pd.merge(target_select, expanded, how="outer", on=["id"])

    ### drop irrelevant rows
    """
    data_merge_select = data_merge.loc[(pd.to_datetime(data_merge.obs_start) <= pd.to_datetime(data_merge.date)) & (pd.to_datetime(data_merge.date) <= pd.to_datetime(data_merge.test_date))]
    """
    #! replaced by (formatting)
    mask1 = pd.to_datetime(data_merge.obs_start) <= pd.to_datetime(data_merge.date)
    mask2 = pd.to_datetime(data_merge.date) <= pd.to_datetime(data_merge.test_date)
    data_merge_select = data_merge.loc[mask1 & mask2]
    report.report_change_in_nrow(
        data_merge, data_merge_select, "Drop irrelevant rows from outer join"
    )

    # Set new ID: old_ID + PHQ time
    #! updated
    data_merge_select["id_new"] = (
        data_merge_select.id + "_" + data_merge_select.test_date.astype("string")
    )

    # Set new time: date + start_time
    #! updated
    data_merge_select["time"] = (
        data_merge_select.date.astype("string")
        + " "
        + data_merge_select.start_time.astype("string")
    )

    data_merge_select.drop(
        ["test_date", "obs_start", "start_time"], axis=1, inplace=True
    )

    return data_merge_select


def label_night_date(
    ts_df: pd.DataFrame, sleep_hour_end: str, time_column="t"
) -> pd.DataFrame:
    """Add a corresponding `night_date`.
    If not before midnight, night_date is the previous date.
    Otherwise, it is the same date

    Args:
        ts_df (pd.DataFrame): Time series dataframe.
        sleep_hour_end (str): When does the sleeping hours end? In string format e.g. '10:00'.
        time_column (str): Which column to label

    Returns:
        pd.DataFrame: Dataframe with added `night_date` column
    """
    ts = ts_df.copy()

    # length of morning (as a cutoff)
    #! optimize later; faster method should be using df['t'].dt.time
    morning_length = pd.to_datetime(sleep_hour_end) - pd.to_datetime("00:00")
    before_midnight_flag = (
        ts[time_column] - ts[time_column].dt.normalize()
    ) > morning_length

    # night_date = the same date
    ts["night_date"] = ts[time_column].dt.date
    # replace if flagged
    ts.loc[~before_midnight_flag, "night_date"] = ts["night_date"] - pd.Timedelta(
        days=1
    )
    return ts


def replace_slp_stage(
    ts_df: pd.DataFrame, from_stage="UNKNOWN", to_stage="AWAKE", epoch_length=30
) -> pd.DataFrame:
    """Replace UNKNOWN in sleep stages.

    Args:
        ts (pd.DataFrame): Time series dataframe
        replacement (str, optional): Replacement string. Defaults to 'AWAKE'.
        epoch_length (int, optional): Length of a epoch in seconds. Defaults to 30.

    Returns:
        pd.DataFrame: Time series dataframe with UNKNOWN stage replaced
    """
    ts = ts_df.copy()

    ### diagnostics
    ts_unknown = ts.loc[ts["stages"] == from_stage]
    # print a dataframe of which PIDs and nights with unknowns
    pid_nights = (
        ts_unknown.groupby(["pid"])["t"].agg("count") * epoch_length / 3600
    )  # how many hours?
    pid_nights.rename("hours_of_unknown", inplace=True)
    # summary statistics
    unknown_percent = ts_unknown.shape[0] / ts.shape[0] * 100
    unknown_total_time = pid_nights.sum()
    print(f"Unknown percent: {unknown_percent:.3f}%")
    print(f"Unknown time:    {unknown_total_time:.3f} hours")
    print(pid_nights)

    ### replacement
    ts.loc[ts["stages"] == from_stage, "stages"] = to_stage
    return ts


def ts2intervals(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Convert time series dataframe to intervals dataframe

    Args:
        ts_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    def mysummarise(tsdf, stage):
        return (
            summarise_stage(tsdf, stage=stage)
            .droplevel("stages", axis="index")
            .reset_index()
        )

    ts_sorted = ts_df.sort_values(["pid", "t"])
    stages = ["AWAKE", "DEEP", "LIGHT", "REM"]
    intervals_by_stage = [mysummarise(ts_sorted, stage=s) for s in stages]
    intervals_by_stage = (
        pd.concat(intervals_by_stage, keys=stages, names=["stages", "index"])
        .reset_index("stages")
        .rename(columns={"first": "start", "last": "end"})
    )
    intervals_by_stage["duration"] = (
        intervals_by_stage["duration"] / pd.Timedelta(seconds=1)
    ).astype(int)
    return intervals_by_stage


def prep_for_drop_days_delta(phqs_raw: pd.DataFrame) -> pd.DataFrame:
    target_new = phqs_raw[["pid", "time", "phq"]].copy()
    target_new["test_date"] = pd.to_datetime(target_new["time"]).dt.date
    target_new = target_new[["pid", "test_date", "phq"]].rename(columns={"pid": "id"})
    target_new_sorted = target_new.sort_values(["id", "test_date"])
    return target_new_sorted


def generate_id_new(df, pid_column: str, test_date_column: str) -> pd.Series:
    df2 = df[[pid_column, test_date_column]].copy()

    df2["test_date_str"] = df2["test_date"].astype(str)
    id_new = df2[[pid_column, "test_date_str"]].agg("_".join, axis=1)
    assert id_new.index.equals(df.index)
    return id_new


def split_id_new(id_new):
    df_id_new_pid = id_new.str.split("_", expand=True)
    df_id_new_pid.columns = ["pid", "night_date"]
    return df_id_new_pid
