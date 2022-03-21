import os
from re import A
import numpy as np
import pandas as pd
import datetime as dt
from helper import *


# from pandarallel import pandarallel  # parallel processing


def clean_all_phq(df):
    df2 = df.copy()
    df2.columns = ["pid", "timestamp", "phq", "sleep"]
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    return df2


def clean_slps(df):
    # data types
    df2 = df.copy()
    df2["start"] = pd.to_datetime(df2.time)
    df2["start_date"] = df2["start"].dt.date
    df2["end"] = df2["start"] + pd.to_timedelta(df2["duration"] - 30, "s")
    df2["intervalID"] = np.arange(len(df2))
    df2.drop(columns="time")
    return df2


def explode2ts(
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
    timeseries = intervals[[ID, start, end]]
    if multiprocessing:
        print("no multiprocessing, fix later")
        ### currently the pandarallel package breaks
        # timeseries[time] = timeseries.parallel_apply(
        #     expand_to_start_times, start=start, end=end, axis=1
        # )
        timeseries[time] = timeseries.apply(
            expand_to_start_times, start=start, end=end, axis=1
        )
    else:
        timeseries[time] = timeseries.apply(
            expand_to_start_times, start=start, end=end, axis=1
        )
    timeseries = timeseries[[ID, time]].explode(time)
    return timeseries


def expand_to_start_times(row, start="start", end="end"):
    """Expand a time interval denoted by start-end to a list of start times
    Helper function called by explode2ts

    Input: Start = 2019-01-01 00:01:00, End = 2019-0101 00:02:30
    Output: ['2019-01-01 00:01:00', '2019-01-01 00:01:30', '2019-01-01 00:02:00']
    """
    return pd.date_range(start=row[start], end=row[end], freq="30s", closed="left")


def time2datetime(t):
    date = dt.date(1970, 1, 1)
    return dt.datetime.combine(date, t)


def dedup_timeseries(ts: pd.DataFrame):
    """Deduplicate timeseries. Logic: grouped by subject id and timestamp, sort by interval start datetime, select last record

    Args:
        ts (pd.DataFrame): time series dataframe

    Returns:
        pd.DataFrame: time series dataframe, deduplicated
    """
    dedupped = ts.sort_values(["pid", "t", "start_date"]).groupby(["pid", "t"]).tail(1)
    report_change_in_nrow(ts, dedupped)
    return dedupped


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


def expand_full_hours(df: pd.DataFrame, hours=None, by_columns=None):
    """Expand to hourly binned dataframe to a full list of hours (user-supplied, defaults to 8pm to 10am), fill with NAs if empty. Grouped by the columns specified in the 'by' option

    Args:
        df (pd.DataFrame): hourly binned dataframe
        by_columns (list): by what columns?
        hours (list): If None, Defaults to 8pm-10am i.e. [20,21,22,23,0,1,2,3,4,5,6,7,8,9,10].

    Returns:
        _type_: _description_
    """
    if not by_columns:
        by_columns = ["pid", "start_date"]
    if not hours:
        hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #! inefficient approach, fix later
    pid_date_combinations = df[by_columns].drop_duplicates().reset_index(drop=True)
    nrow = pid_date_combinations.shape[0]
    hours_rep = [hours for _ in range(0, nrow)]
    hours_series = pd.Series(hours_rep, name="hour").reset_index(drop=True)
    fulldf = pd.concat([pid_date_combinations, hours_series], axis=1).explode("hour")

    on_columns = by_columns + ["hour"]
    expanded = pd.merge(fulldf, df, on=on_columns, how="left")
    out = expanded.fillna(0, inplace=False)
    return out


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
    Drop PHQ test records which is <14 days from previous.
    From Haotian's script
    """
    
    # make copy
    cp = target_select.copy()
    
    # clean
    cp['date'] = pd.to_datetime(cp['test_date'])

    # calculate number of days from previous record
    target_date_diff = ( cp[['id','date']]
        .sort_values(['id', 'date'])
        .groupby('id')
        .diff()
    )
    # combine with main df
    target_date_diff['daysdelta'] = target_date_diff['date'] / np.timedelta64(1, 'D')
    target_select_clean = pd.concat([cp, target_date_diff['daysdelta']], axis=1)

    # filter
    out = target_select_clean.loc[(target_select_clean['daysdelta'] >= threshold) | (target_select_clean['daysdelta'].isna()),:]
    # clean output
    out = out.drop(['date', 'daysdelta'], axis=1)
    report_change_in_nrow(target_select, out, operation="Drop PHQ records which is <14 days from previous")   

    return out

    
def generate_ts_y(df, column_id='id_new', column_index='index', column_features=['AWAKE','LIGHT','DEEP','REM']):

    ### Generate time series (many rows for each y label) and corresponding . The ID column in the time series dataframe 
    
    col_select = column_features + [column_index] + [column_id]
    input_df = df[col_select].copy()
    
    ts = input_df.reset_index(drop=True)
    y = input_df.groupby(column_id).tail(1).set_index(column_id)['target']

    ts_ind = set(ts[column_id])
    y_ind = set(y.index)
    assert (ts_ind == y_ind), "ts and y features don't match"
    return ts, y
