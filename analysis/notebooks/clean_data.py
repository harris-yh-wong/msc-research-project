import os
from re import A
from telnetlib import TSPEED
import numpy as np
import pandas as pd
import datetime as dt
from helper import *
import report
from deprecated import deprecated

# from pandarallel import pandarallel  # parallel processing


def clean_all_phq(df):
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


@deprecated(reason="replaced by loop comprehension")
def _explode2ts(
    intervals,
    ID="intervalID",
    start="start",
    end="end",
    time="t",
    multiprocessing=False,
):
    """Explode interval dataframe to time series format

    Args:
        intervals (pd.DataFrame): dateaframe of intervals
        ID (str, optional): Column name for ID unique for each interval. Defaults to "intervalID".
        start (str, optional): Column name for interval start time. Defaults to "start".P
        end (str, optional): Column name for interval end time. Defaults to "end".
        time (str, optional): Column name for time in the output. Defaults to "t".

    Returns:
        pd.DataFrame: time series dataframe
    """

    def expand_to_start_times(row, start_col_name="start", end_col_name="end"):
        """
        Expand a time interval denoted by start-end to a list of start times
        Helper function called by explode2ts

        Input: row[start_column] = 2019-01-01 00:01:00, row[end_column] = 2019-0101 00:02:30
        Output: ['2019-01-01 00:01:00', '2019-01-01 00:01:30', '2019-01-01 00:02:00']
        """
        return pd.date_range(
            start=row[start_col_name], end=row[end_col_name], freq="30s", closed=None
        )

    timeseries = intervals[[ID, start, end]]
    if multiprocessing:
        print("no multiprocessing, fix later")
        ### currently the pandarallel package breaks
        # timeseries[time] = timeseries.parallel_apply(
        #     expand_to_start_times, start=start, end=end, axis=1
        # )
        timeseries[time] = timeseries.apply(
            expand_to_start_times, start_col_name=start, end_col_name=end, axis=1
        )
    else:
        timeseries[time] = timeseries.apply(
            expand_to_start_times, start_col_name=start, end_col_name=end, axis=1
        )

    timeseries = timeseries[[ID, time]].explode(time)
    return timeseries


def explode2ts(df):
    # the start_date is kept because overlap-removal step requires this column for deciding which label to keep/discard
    data = (
        (row.pid, t, row.start_date, row.stages)
        for row in df.itertuples()
        for t in pd.date_range(row.start, row.end, freq="30s")
    )
    ts = pd.DataFrame(data=data, columns=["pid", "t", "start_date", "stages"])
    return ts


def time2datetime(t):
    """Convert time to a datetime object

    Args:
        t (dt.time): time

    Returns:
        dt.datetime: datetime object
    """
    date = dt.date(1970, 1, 1)
    return dt.datetime.combine(date, t)


def time2seconds(t):
    """Convert time to seconds from midnight

    Args:
        t (dt.time): time

    Returns:
        int: seconds from midnight
    """
    delta = time2datetime(t) - dt.datetime(1970, 1, 1)
    return delta.total_seconds()

def subset_timeseries_within_interval(timeseries, start, end, inclusive='left'):
    subset = (
        timeseries.set_index("t", inplace=False)
        .between_time(start, end, inclusive)
        .reset_index(inplace=False)
    )
    report.report_change_in_nrow(timeseries, subset)
    return subset


def dedup_timeseries(ts: pd.DataFrame):
    """Deduplicate timeseries. Logic: grouped by subject id and timestamp, sort by interval start datetime, select last record

    Args:
        ts (pd.DataFrame): time series dataframe

    Returns:
        pd.DataFrame: time series dataframe, deduplicated
    """
    dedupped = ts.sort_values(["pid", "t", "start_date"]).groupby(["pid", "t"]).tail(1)
    report.report_change_in_nrow(ts, dedupped)
    return dedupped


@deprecated(reason="replaced by a more flexible bin_by_time routine")
def bin_by_hour(ts: pd.DataFrame):
    """Bin sleep stage time series by hour

    Args:
        ts (pd.DataFrame): time series dataframe
    """
    df = ts.copy()
    df["hour"] = df["t"].dt.hour
    df = pd.concat([df, pd.get_dummies(df["stages"])], axis=1)
    binned = df.groupby(["pid", "start_date", "hour"])[
        ["AWAKE", "LIGHT", "DEEP", "REM"]
    ].agg("sum")
    binned["sum"] = binned.sum(axis=1)
    binned = binned.reset_index(inplace=False)
    return binned


from datetime import datetime

def bin_by_time(ts: pd.DataFrame, freq="H"):
    """Bin time series by the time

    Args:
        ts (pd.DataFrame): Time series dataframe
        freq (str, optional): Frequency to bin by. Defaults to "H".

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
    binned['hour'] = binned['t'].dt.hour
    binned['start_date'] = binned['t'].dt.date
    binned.drop("t", axis=1, inplace=True)
    
    ### 
    binned["sum"] = binned[stages].sum(axis=1)

    report.report_change_in_nrow(ts, binned)
    return binned


def expand_full_hours(df: pd.DataFrame, hours=None, by_columns=None):
    """Expand to hourly binned dataframe to a full list of hours (user-supplied, defaults to 8pm to 10am), fill with NAs if empty. Grouped by the columns specified in the 'by' option

    Args:
        df (pd.DataFrame): hourly binned dataframe
        by_columns (list): by what columns?
        hours (list): If None, Defaults to 8pm-10am i.e. [20,21,22,23,0,1,2,3,4,5,6,7,8,9,10].

    Returns:
        _type_: _description_
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


def normalize_binned(df: pd.DataFrame, over=120):
    """Normalize binned dataframe by 120

    Args:
        df (pd.DataFrame): binned dataframe
        over (int): normalize over what? Defaults to 120 (number of 30-second intervals in an hour)

    Returns:
        pd.DataFrame: Normalized dataframe
    """
    binned = df.copy()
    binned["AWAKE"] = binned["AWAKE"] / over
    binned["LIGHT"] = binned["LIGHT"] / over
    binned["DEEP"] = binned["DEEP"] / over
    binned["REM"] = binned["REM"] / over
    if "sum" in binned.columns:
        binned["sum"] = binned["sum"] / over

    return binned


#### HAOTIAN's


def drop_days_delta(target_select, threshold=14):
    """
    Drop PHQ test records which is <14 days from previous result
    The logic is to take the last record
    Author: Haotian Gao
    """

    # make copy
    tgt = target_select.copy()

    # clean
    tgt["date"] = pd.to_datetime(tgt["test_date"])

    # calculate number of days from previous record
    target_date_diff = (
        tgt[["id", "date"]].sort_values(["id", "date"]).groupby("id").diff()
    )
    # combine with main df
    target_date_diff["daysdelta"] = target_date_diff["date"] / np.timedelta64(1, "D")
    target_select_clean = pd.concat([tgt, target_date_diff["daysdelta"]], axis=1)

    # filter
    out = target_select_clean.loc[
        (target_select_clean["daysdelta"] >= threshold)
        | (target_select_clean["daysdelta"].isna()),
        :,
    ]
    # clean output
    out = out.drop(["date", "daysdelta"], axis=1)
    report.report_change_in_nrow(
        tgt,
        out,
        operation="Drop PHQ test results which are within 14 days from previous result",
    )

    return out


def generate_ts_y(
    df,
    column_id="id_new",
    column_index="index",
    column_features=["AWAKE", "LIGHT", "DEEP", "REM"],
):

    ### Generate time series (many rows for each y label) and corresponding . The ID column in the time series dataframe

    col_select = column_features + [column_index] + [column_id] + ["target"]
    input_df = df[col_select].copy()

    ts = input_df.reset_index(drop=True)
    ts.drop("target", axis=1, inplace=True)
    y = input_df.groupby(column_id).tail(1).set_index(column_id)["target"]

    ts_ind = set(ts[column_id])
    y_ind = set(y.index)
    assert ts_ind == y_ind, "ts and y features don't match"
    return ts, y


def merge_slp_phq(expanded, phqs_raw):

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
        days=15
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
    """data_merge = pd.merge(target_select, data_tab, how='outer', on=['id'])"""
    #! replaced by
    data_merge = pd.merge(target_select, expanded, how="outer", on=["id"])

    """
    data_merge_select = data_merge.loc[(pd.to_datetime(data_merge.obs_start) <= pd.to_datetime(data_merge.date)) & (pd.to_datetime(data_merge.date) <= pd.to_datetime(data_merge.test_date))]
    """

    #! replaced by (formatting)
    mask1 = pd.to_datetime(data_merge.obs_start) <= pd.to_datetime(data_merge.date)
    mask2 = pd.to_datetime(data_merge.date) <= pd.to_datetime(data_merge.test_date)
    data_merge_select = data_merge.loc[mask1 & mask2]
    report.report_change_in_nrow(data_merge, data_merge_select)

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
