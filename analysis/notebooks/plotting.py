import os
import numpy as np
import pandas as pd

from mappings import *
from helper import *
import report

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches
import datetime as dt


def subset_intervals(
    intervals: pd.DataFrame, start_date=None, end_date=None, id=None, msg=None
):

    """s intervals dataframe by dates and pid"""

    ### if none: do not s

    s = intervals.copy()

    if id is not None:
        if type(id) is str:
            id = [id]
        s = s[s["pid"].isin(id)]

    if start_date is not None:
        start_timestamp = pd.Timestamp(start_date)
        s = s.loc[s["start"] >= start_timestamp]

    if end_date is not None:
        end_timestamp = pd.Timestamp(end_date + pd.Timedelta(1, "day"))
        s = s.loc[s["end"] <= end_timestamp]

    report.report_change_in_nrow(intervals, s, operation=msg)

    return s


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


def plot_phs_trajectories(df, outfile=None):

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
    phs_min = dfcp["phs"].min()
    phs_max = dfcp["phs"].max()

    ### sort the PIDs in PHsS by % depr
    dfcp["depr"] = dfcp["phs"] >= depr_cutoff
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
        # s by pid
        s = dfcp.loc[dfcp["pid"] == pid]
        x = s["time"].dt.date
        y = s["phs"]
        # plot
        ax.plot(x, y, "o")
        # plot horizontal cutoff
        ax.axhline(y=depr_cutoff, color="r", linestyle="dashed")

        # formatting
        ax.title.set_text(f"{pid} ({depr_pc}%)")
        ax.set_xlim(date_min, date_max)
        ax.set_ylim(phs_min - 1, phs_max + 1)
        # plt.gca().invert_yaxis()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))

    ### Formatting for all subplots
    plt.subplots_adjust(hspace=0.5)  # leave margin for title for each subplot

    ### Formatting overall
    # plt.suptitle(
    #     f"PHs trajectories for {n_pid} patients, depression percent in parenthesis",
    #     fontsize=24
    # )

    plt.savefig(outfile)
    return None


def plot_binned(df, max_plots=5):

    primary_keys = df["id_new"].unique()
    n_plots = len(primary_keys)

    if n_plots > max_plots:
        print(f"TOO MANY PLOTS ({n_plots}).\nOnly showing {max_plots} plots.\nIncrease max_plots")
        n_plots = max_plots

    for i in range(n_plots):
        key = primary_keys[i]
        delta_t = df.loc[df["id_new"] == key].reset_index(drop=True)
        delta_t[["AWAKE", "LIGHT", "DEEP", "REM"]].plot.line(subplots=True)
        label = str(delta_t["target"].unique()[0])
        nobs = delta_t['nights_recorded'].unique()[0]
        plt.suptitle(f"{nobs}, {label}, {key}")
        plt.show()

    return None


def plot_imp(classifier, features, n_features=5):
    ### Default
    imp = classifier.feature_importances_
    sorted_idx = imp.argsort()
    idx = sorted_idx[0:n_features]
    plt.barh(features[idx], classifier.feature_importances_[idx])
    plt.xlabel("Random Forest Feature Importance")


def plot_imp_perm(imp, features, n=5):
    ### Permutation importance
    perm_sorted_idx = imp["importances_mean"].argsort()
    perm_idx = perm_sorted_idx[::-1][:n]  # n largest first
    plt.barh(features[perm_idx], imp["importances_mean"][perm_idx])
    plt.xlabel("Permutation Importance")
    plt.gca().invert_yaxis()


def plot_search_results(grid):
    """
    Plot
    Params:
        grid: A trained GridSearchCV object.
    """
    #todo  Compatibility with 1 parameter

    ## Results from grid search
    results = grid.cv_results_
    means_test = results["mean_test_score"]
    stds_test = results["std_test_score"]
    means_train = results["mean_train_score"]
    stds_train = results["std_train_score"]

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results["param_" + p_k].data == p_v))

    params = grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex="none", sharey="all", figsize=(20, 5))
    fig.suptitle("Score per parameter")
    fig.text(0.04, 0.5, "MEAN SCORE", va="center", rotation="vertical")
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1 :])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle="--", marker="o", label="test")
        ax[i].errorbar(x, y_2, e_2, linestyle="-", marker="^", label="train")
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()
