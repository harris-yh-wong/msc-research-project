import os
from attr import asdict
import numpy as np
import pandas as pd

from mappings import *
from helper import *
import report
import feat_engineering

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches
import datetime as dt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay


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


def plot_dedupped_intervals(
    plotdf: pd.DataFrame,
    metadata: pd.DataFrame,
    sleep_hour_start: str,
    sleep_hour_end: str,
    c_dict=None,
    max_nights=30,
):
    """Plot dedupped intervals into a coloured stacked bars

    Args:
        plotdf (pd.DataFrame): Interval dataframe for plotting. Should be dedupped.
        metadata (pd.DataFrame): Metadata from `import_data.import_data()`
        sleep_hour_start (str): When does the 'sleep time' start? e.g. '20:00'
        sleep_hour_end (str): When does the 'sleep time' end? e.g. '10:00'
        c_dict (dict, optional): Color palette dictionary. Defaults to None, using a prespecified colour palette
    """

    ### CHECKS
    # ensure all columns are there
    required_cols = [
        "pid",
        "night_date",
        "duration",
        "start",
        "end",
        "stages",
    ]
    assert set(required_cols).issubset(
        set(plotdf.columns)
    ), "Required column(s) missing."
    plotdf = plotdf[required_cols]

    # ensure all observations come from the same patient
    assert plotdf["pid"].nunique() == 1, "PIDs not unique in the input dataframe."

    # ensure duration is in number of seconds
    assert plotdf["duration"].dtype in [
        "int32",
        "float64",
    ], "Duration should be in number of seconds."

    # number of nights / fig size
    n_nights = plotdf["night_date"].nunique()
    assert n_nights <= max_nights, "Too many nights. Increase `max_nights`."

    # default hours
    c_dict = {
        "AWAKE": "red",
        "REM": "gold",
        "LIGHT": "lightblue",
        "DEEP": "steelblue",
        "UNKNOWN": "gray",
    }
    colors = [c_dict[stage] for stage in plotdf["stages"]]

    # axis limits,
    # required later
    start = pd.to_datetime(sleep_hour_start)
    start_hours = (start - start.normalize()).total_seconds() / 3600
    end = pd.to_datetime(sleep_hour_end)
    end_hours = (end - end.normalize()).total_seconds() / 3600

    # convert to number of hours
    widths = plotdf["duration"] / 3600
    lefts = (plotdf["start"] - plotdf["start"].dt.normalize()).dt.total_seconds() / 3600
    # put everything after midnight
    lefts = [left + 24 if left <= end_hours else left for left in lefts]

    # plot
    fig, ax = plt.subplots(1, figsize=(16, n_nights))
    ax.barh(
        y=plotdf["night_date"].astype(str),
        # width = plotdf['duration'].apply(lambda sec: pd.Timedelta(seconds=sec)),
        width=widths,
        left=lefts,
        color=colors,
    )

    # axis
    ax.set_xlim(start_hours, 24 + end_hours)
    ax.invert_yaxis()

    # title
    title = plotdf.iloc[0]["pid"]
    ax.title.set_text(title)
    return fig


def plot_deltaT_similarity(df):
    assert df.shape[0] != 0, "Input dataframe cannot be empty."

    # plot
    stages = ["AWAKE", "REM", "LIGHT", "DEEP"]

    fig, axes = plt.subplots(ncols=4, figsize=(24, 3.5))
    for i, ax in enumerate(axes.flatten()):
        stage = stages[i]
        sns.lineplot(
            y=stage, x="slp_stage_hour_to_plot", hue="night_date", data=df, ax=ax
        )
        ax.set_title(stage)
        if i + 1 >= len(stages):  # last subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        else:
            ax.get_legend().remove()

    title = df["id_new"].unique()[0]
    fig.suptitle(title)
    return fig


def plot_hypnogram(
    plotdf: pd.DataFrame,
    metadata: pd.DataFrame,
    sleep_hour_start: str,
    sleep_hour_end: str,
    c_dict=None,
    max_nights=30,
):
    """Plot dedupped intervals into a coloured stacked bars

    Args:
        plotdf (pd.DataFrame): Interval dataframe for plotting. Should be dedupped.
        metadata (pd.DataFrame): Metadata from `import_data.import_data()`
        sleep_hour_start (str): When does the 'sleep time' start? e.g. '20:00'
        sleep_hour_end (str): When does the 'sleep time' end? e.g. '10:00'
        c_dict (dict, optional): Color palette dictionary. Defaults to None, using a prespecified colour palette
    """

    ### CHECKS
    # ensure all columns are there
    required_cols = [
        "pid",
        "night_date",
        "duration",
        "start",
        "end",
        "stages",
    ]
    assert set(required_cols).issubset(
        set(plotdf.columns)
    ), "Required column(s) missing."
    plotdf = plotdf[required_cols]

    # ensure all observations come from the same patient
    assert plotdf["pid"].nunique() == 1, "PIDs not unique in the input dataframe."

    # ensure duration is in number of seconds
    assert plotdf["duration"].dtype in [
        "int32",
        "float64",
    ], "Duration should be in number of seconds."

    # number of nights / fig size
    n_nights = plotdf["night_date"].nunique()
    assert n_nights <= max_nights, "Too many nights. Increase `max_nights`."

    # default stage colours
    c_dict = {
        "AWAKE": "red",
        "REM": "gold",
        "LIGHT": "lightblue",
        "DEEP": "steelblue",
        "UNKNOWN": "gray",
    }

    # axis limits,
    # required later
    start = pd.to_datetime(sleep_hour_start)
    start_hours = (start - start.normalize()).total_seconds() / 3600
    end = pd.to_datetime(sleep_hour_end)
    end_hours = (end - end.normalize()).total_seconds() / 3600

    # plotting order of nights and stages (for stages, sort later)
    plotdf.sort_values("night_date", inplace=True)
    stages = ["AWAKE", "REM", "LIGHT", "DEEP"]
    mapping = {day: i for i, day in enumerate(stages)}
    n_nights = plotdf["night_date"].nunique()

    # plot
    fig, axes = plt.subplots(nrows=n_nights, ncols=1, figsize=(16, n_nights))
    # loop for each night
    for i, ax in enumerate(axes.flat):

        night = plotdf["night_date"].unique()[i]
        night_df = plotdf.query("night_date == @night")

        # plotting order
        key = night_df["stages"].map(mapping)
        night_df = night_df.iloc[key.argsort()]

        colors = [c_dict[stage] for stage in night_df["stages"]]

        # convert to number of hours
        widths = night_df["duration"] / 3600
        lefts = (
            night_df["start"] - night_df["start"].dt.normalize()
        ).dt.total_seconds() / 3600
        # put everything after midnight
        lefts = [left + 24 if left <= end_hours else left for left in lefts]

        ax.barh(
            y=night_df["stages"].astype(str),
            width=widths,
            left=lefts,
            color=colors,
        )

        # axis
        ax.set_xlim(start_hours, 24 + end_hours)
        ax.invert_yaxis()

        ax.text(start_hours, 0, str(night))

        # frames
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if n_nights - i > 1:
            ax.xaxis.set_ticklabels([])

    # title
    title = plotdf.iloc[0]["pid"]
    fig.suptitle(title)
    return fig


def plot_gannt(plotdf, metadata, c_dict=None, overlap=True):
    # default colors
    if not c_dict:
        c_dict = {
            "AWAKE": "red",
            "REM": "gold",
            "LIGHT": "lightblue",
            "DEEP": "steelblue",
            "UNKNOWN": "gray",
        }

    # clean data
    plotdf = plotdf.sort_values("start")
    plotdf["ObsIndex"] = plotdf.groupby(["pid", "start_date"]).cumcount()

    # setup
    fig, ax = plt.subplots(1, figsize=(16, 10))

    # plot
    colors = [c_dict[stage] for stage in plotdf["stages"]]

    ax.barh(
        y=plotdf["ObsIndex"],
        width=plotdf["end"] - plotdf["start"],
        left=plotdf["start"],
        color=colors,
    )

    # print(plotdf.shape[0])
    # for i in range(plotdf.shape[0]):
    #     row = plotdf.reset_index(drop=True).iloc[i, :]
    #     color = c_dict[row["stages"]]
    #     y = 1
    #     if overlap:
    #         y = row["ObsIndex"]
    #     ax.barh(
    #         y=y,
    #         width=row["end"] - row["start"],
    #         left=row["start"],
    #         color=color,
    #     )

    # axis
    ax.set_xlim(plotdf["start"].min(), plotdf["end"].max())
    ax.set_ylim(-1, plotdf["ObsIndex"].max() + 1)
    plt.gca().invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.tick_params(axis="x", labelrotation=45)

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
        print(
            f"TOO MANY PLOTS ({n_plots}).\nOnly showing {max_plots} plots.\nIncrease max_plots"
        )
        n_plots = max_plots

    for i in range(n_plots):
        key = primary_keys[i]
        delta_t = df.loc[df["id_new"] == key].reset_index(drop=True)
        delta_t[["AWAKE", "LIGHT", "DEEP", "REM"]].plot.line(subplots=True)
        label = str(delta_t["target"].unique()[0])
        nobs = delta_t["nights_recorded"].unique()[0]
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
    # todo  Compatibility with 1 parameter

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


def plot_relevant_features(relevance_table: pd.DataFrame, n_features=10) -> None:
    """Plot bar chart of relevant features from relevance table from tsfresh

    Args:
        relevance_table (pd.DataFrame): Relevance table output from tsfresh
        n_features (int, optional): Number of feature sto plot. Defaults to 10.
    """
    n_features = 50
    relevance_table["stage"] = relevance_table["feature"].str[:5]
    relevance_table["-logp"] = np.log10(relevance_table["p_value"]) * -1

    show_features = feat_engineering.get_relevant_features(relevance_table)[
        0:n_features
    ]
    relevance_table_subset = relevance_table.loc[
        relevance_table["feature"].isin(show_features),
    ]

    sns.set(rc={"figure.figsize": (12, n_features / 2)})
    sns.barplot(y="feature", x="-logp", hue="stage", data=relevance_table_subset)


def plot_discrimination(
    y_test: pd.Series, y_pred: pd.Series, phq: pd.Series, phq_threshold=10
) -> None:
    """Plot two plots showing model discrimination.
    Plot 1: Prediction probabilities by ground truth
    Plot 2:

    Args:
        y_test (pd.Series): Ground truth
        y_pred (pd.Series): Prediction probabilities
        phq (pd.Series): PHQ score
        phq_threshold (int): PHQ score threshold for True/False depressive status, where PHQ score >= threshold gives a True label. Defaults to 10.
    """

    # prepare the data for plotting
    plot_df = phq.set_index("id_new").loc[y_test.index]
    plot_df["y_pred"] = y_pred
    plot_df["class"] = plot_df["target"] >= phq_threshold

    # Bin PHQ scores into categories
    depression_categories = ["none", "mild", "moderate", "moderately severe", "severe"]
    plot_df["category"] = pd.cut(
        plot_df["target"],
        [0, 5, 10, 15, 20, 27],
        right=False,
        labels=depression_categories,
    )
    plot_df["category"] = pd.Categorical(
        plot_df["category"], categories=depression_categories, ordered=True
    )
    plot_df = plot_df.sort_values("category", inplace=False)

    # plot 1
    plt.figure()
    sns.violinplot(x="category", y="y_pred", cut=0, data=plot_df)
    # plot 2
    plt.figure()
    sns.violinplot(x="class", y="y_pred", cut=0, data=plot_df)


def plot_freq_lists(*args, show=10):
    n_plots = len(args)
    dfs = [series.sort_values(ascending=False).head(show) for series in args]
    fig, ax = plt.subplots(1, n_plots)
    for i in range(n_plots):
        sns.barplot(dfs[i].index, dfs[i].values, ax=ax[i], palette="colorblind")
    fig.autofmt_xdate()
    fig.show()


def plot_cv_results(cv_results_df, order="keep", aggregate="mean"):
    """Plot a bar chart showing the performance metrics from a dataframe of cross-validation runs

    Args:
        cv_results_df (pd.DataFrame): Cross-validation run result dataframe, generated from cv_results2df
        order (str, optional): Defaults to 'keep'.
        aggregate (str, optional): Aggregate function used by pd.DataFrame.agg(). Defaults to 'mean'.
    """

    if order == "keep":
        ordered = False
    else:
        ordered = True
    cv_results_df["name"] = pd.Categorical(
        cv_results_df["name"],
        categories=cv_results_df["name"].unique(),
        ordered=ordered,
    )

    longdf = pd.wide_to_long(
        cv_results_df,
        stubnames="test",
        i=["name", "split"],
        j="metric",
        sep="_",
        suffix=".+",
    )
    aggdf = (
        longdf.reset_index().groupby(["name", "metric"]).agg(aggregate).reset_index()
    )

    g = sns.barplot(x="name", y="test", hue="metric", data=aggdf)
    g.set_ylabel("Score")
    g.set_xlabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def plot_roc(y, y_pred, col_index=1):
    """Plot a ROC curve

    Args:
        y_test (pd.Series): true class
        y_pred (np.ndarray): predicted probabilities
        col_index (int): which column to index? Defaults to 1.

    """
    RocCurveDisplay.from_predictions(y, y_pred[:, col_index])
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_cdf(s: pd.Series):
    """Plot the cumulative distribution function

    Args:
        s (pd.Series): A count series e.g. from pd.DataFrame.value_counts('column')
    """
    cdf = (s / sum(s)).cumsum()

    fig, ax = plt.subplots()
    ax.plot(cdf, marker="o")
    plt.ylim([-0.05, 1.05])
    plt.show()


def plot_correlation_heatmap(df, title=None):
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    if title is not None:
        ax.set_title(title)


def plot_learningCurve(history_dict, score="accuracy", stack_vertical=False):

    epoch = pd.DataFrame(history_dict).shape[0]
    epoch_range = range(1, epoch + 1)

    nrow, ncol = (1, 2)
    figsize = (12, 4)
    if stack_vertical:
        nrow, ncol = (2, 1)
        figsize = (4, 12)

    fig, (ax1, ax2) = plt.subplots(nrow, ncol, figsize=figsize)

    # Plot training & validation accuracy values

    ax1.plot(epoch_range, history_dict[score])
    ax1.plot(epoch_range, history_dict[f"val_{score}"])
    ax1.set_title(f"Model {score}")
    ax1.set_ylabel(score)
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Val"], loc="upper left")

    # Plot training & validation loss values
    ax2.plot(epoch_range, history_dict["loss"])
    ax2.plot(epoch_range, history_dict["val_loss"])
    ax2.set_title("Model loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Val"], loc="upper left")

    return fig
