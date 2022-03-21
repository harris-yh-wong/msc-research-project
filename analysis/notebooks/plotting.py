import os
import numpy as np
import pandas as pd

from mappings import *
from helper import *


from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches
import datetime as dt


def subset_intervals(intervals: pd.DataFrame, start_date=None, end_date=None, id=None):

    """Subset intervals dataframe by dates and pid"""

    ### if none: do not subset
    if type(id) is str:
        id = [id]

    if not id:
        interval_query = intervals.loc[intervals["pid"].isin(id), :]
    if not start_date:
        interval_query = interval_query.query("start_date >= @start_date")
    if not end_date:
        interval_query = interval_query.query("end_date <= @end_date")

    return interval_query


def plot_gannts(df, metadata, max_plots=5):
    pid_date_tuples = (
        df[["pid", "start_date"]].drop_duplicates().to_records(index=False)
    )
    n_plots = len(pid_date_tuples)

    assert (
        n_plots < max_plots
    ), f"There would be {n_plots} plots (limitted to {max_plots}). Increase max_plots."

    for (pid, date) in pid_date_tuples:
        plotdf = df.loc[(df["start_date"] == date) & (df["pid"] == pid)]
        plot_gannt(plotdf, metadata)
    return None


def plot_gannt(plotdf, metadata, c_dict=None):
    # default colors
    if not c_dict:
        c_dict = {
            "AWAKE": "red",
            "DEEP": "steelblue",
            "LIGHT": "lightblue",
            "REM": "gold",
        }

    # clean data
    plotdf = plotdf.sort_values("start")
    plotdf["ObsIndex"] = plotdf.groupby(["pid", "start_date"]).cumcount()

    # setup
    fig, ax = plt.subplots(1, figsize=(16, 10))

    # plot
    for i in range(plotdf.shape[0]):
        row = plotdf.reset_index(drop=True).iloc[i, :]
        color = c_dict[row["stages"]]
        ax.barh(
            y=row["ObsIndex"],
            width=row["end"] - row["start"],
            # width=row['duration']/(60*60*24),
            left=row["start"],
            color=color,
        )

    # axis
    ax.set_xlim(plotdf["start"].min(), plotdf["end"].max())
    ax.set_ylim(-1, plotdf["ObsIndex"].max() + 1)
    plt.gca().invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))

    # legends
    patch_list = [
        matplotlib.patches.Patch(color=color, label=label)
        for label, color in c_dict.items()
    ]
    ax.legend(handles=patch_list)

    # title
    title = plotdf["start_date"].iloc[0].strftime("%Y.%m.%d")
    if metadata is not None:
        pid = plotdf["pid"].iloc[0]
        IDnum = metadata.loc[metadata["subject_id"] == pid, "ID"].iloc[0].astype(str)
        title = title + "  (ID: " + IDnum + " / " + pid + ")"
    ax.title.set_text(title)

    return fig


def plot_stages(plotdf, metadata, c_dict=None):
    # default colors
    if not c_dict:
        c_dict = {"AWAKE": "red", "DEEP": "green", "LIGHT": "blue", "REM": "purple"}

    # setup
    fig, ax = plt.subplots(1, figsize=(16, 10))

    # plot
    color = c_dict[plotdf["stages"]]
    ax.barh(
        y=plotdf["start_date"],
        width=plotdf["end"] - plotdf["start"],
        # width=row['duration']/(60*60*24),
        left=plotdf["start"]
        # color=color
    )

    # # axis
    # ax.set_xlim(plotdf['start'].min(), plotdf['end'].max())
    # ax.set_ylim(-1, plotdf['start_date'].max()+1)
    # plt.gca().invert_yaxis()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))

    # # legends
    # patch_list = [matplotlib.patches.Patch(
    #     color=color, label=label) for label, color in c_dict.items()]
    # ax.legend(handles=patch_list)

    # # title
    # title = plotdf['start_date'].iloc[0].strftime('%Y.%m.%d')
    # if metadata is not None:
    #     pid = plotdf['pid'].iloc[0]
    #     IDnum = metadata.loc[metadata['subject_id']
    #                          == pid, 'ID'].iloc[0].astype(str)
    #     title = title + '  (ID: ' + IDnum + ' / ' + pid + ')'
    # ax.title.set_text(title)

    return fig


def plot_phq_trajectories(df, outfile=None):

    ### params
    depr_cutoff = 10
    subplot_height = 2
    subplot_width = 16

    ### check outfile is accesible
    if outfile:
        check_file_accesible(outfile)

    ### variables
    dfcp = df.copy()
    date_min = dfcp["time"].dt.date.min()
    date_max = dfcp["time"].dt.date.max()
    phq_min = dfcp["phq"].min()
    phq_max = dfcp["phq"].max()

    ### sort the PIDs in PHQS by % depr
    dfcp["depr"] = dfcp["phq"] >= depr_cutoff
    pid_depr_pc = (
        dfcp[["pid", "depr"]]
        .groupby(["pid"])
        .agg(["mean"])
        .pipe(flatten_multiindex, join_by="_")
        .sort_values("depr_mean")
    )
    pids = pid_depr_pc.index
    depr_pc = round(pid_depr_pc["depr_mean"] * 100, 2)
    n_pid = len(pids)

    ### plotting setup
    top = 1 - 1 / n_pid
    if top:
        print(top)
        # return None

    fig, axes = plt.subplots(
        n_pid,  # nrows
        gridspec_kw=dict(left=0.05, right=0.95, bottom=0.05, top=0.95),
        figsize=(subplot_width, subplot_height * n_pid),
    )
    flt = axes.flatten()

    ### plotting loop
    for i, (pid, depr_pc) in enumerate(zip(pids, depr_pc)):
        # set axis
        ax = flt[i]
        # subset by pid
        subset = dfcp.loc[dfcp["pid"] == pid]
        x = subset["time"].dt.date
        y = subset["phq"]
        # plot
        ax.plot(x, y, "o")
        # plot horizontal cutoff
        ax.axhline(y=depr_cutoff, color="r", linestyle="dashed")

        # formatting
        ax.title.set_text(f"{pid} ({depr_pc}%)")
        ax.set_xlim(date_min, date_max)
        ax.set_ylim(phq_min - 1, phq_max + 1)
        # plt.gca().invert_yaxis()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))

    ### Formatting for all subplots
    plt.subplots_adjust(hspace=0.5)  # leave margin for title for each subplot

    ### Formatting overall
    # plt.suptitle(
    #     f"PHQ trajectories for {n_pid} patients, depression percent in parenthesis",
    #     fontsize=24
    # )

    plt.savefig(outfile)
    return None
