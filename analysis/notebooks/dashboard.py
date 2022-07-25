import streamlit as st

st.set_page_config(layout="wide")

import time

time_start = time.time()

### LIBRARIES

##### MODULES
import os
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.dates
import matplotlib.patches
import datetime as dt
import ast

import import_data
import clean_data
import feat_engineering
import plotting
import report
import helper

import random

import feather
import json
import seaborn as sns


##### OPTIONS
pd.options.mode.chained_assignment = None

##### DIRECTORIES
proj_dir = Path(".") / ".." / ".."
source_data_dir = proj_dir / "data" / "source"
clean_data_dir = proj_dir / "data" / "clean"

##### IMPORT DATA
phqs_raw, _, metadata = import_data.import_data(proj_dir / "data/source")

intervals = pd.read_feather(
    proj_dir / "data/clean/preprocessApr09/dedupped_intervals.ftr"
)

binned60 = pd.read_feather(proj_dir / "data/clean/preprocess0608_60m/data_merge2.ftr")

clinical_features_summary = pd.read_feather(
    proj_dir / "data/clean/preprocess0617/clinical_features_Xy.ftr"
)
clinical_features_per_night = pd.read_feather(
    proj_dir / "data/artefacts/clinical_features_per_night_0617.ftr"
)
clinical_features_per_night["night_date"] = pd.to_datetime(
    clinical_features_per_night["night_date"]
)


##### INPUTS


with st.form(key="my_form_to_submit"):
    PID = st.text_input("PID", value="0104dfff-4dcd-48ff-b912-51362f098ed0")
    END_DATE = st.date_input("End", value=dt.date(2019, 7, 31))
    START_DATE = st.date_input("Start (override if checked)")
    WINDOW_14 = st.checkbox("Window 14 days")

    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    if WINDOW_14:
        START_DATE = END_DATE - dt.timedelta(days=14)

    ### SUBSET DATA
    plotdf = plotting.subset_intervals(
        intervals, start_date=START_DATE, end_date=END_DATE, id=PID
    )

    ### PLOT HYPNOGRAM
    if plotdf.shape[0] == 0:
        st.write("No data found with this criteria!")
    else:
        plt.figure()
        fig = plotting.plot_hypnogram(
            plotdf, metadata, sleep_hour_start="20:00", sleep_hour_end="10:00"
        )
        st.pyplot(fig)

    ### SHOW PHQS QUESTIONNAIRE RESULTS
    show = phqs_raw.query("pid == @PID")
    show = show.loc[pd.to_datetime(show["time"]).dt.date == END_DATE]

    st.dataframe(show)

    ### PLOT BINNED TIME SERIES
    # plt.figure()
    # fig = plotting.plot_binned(binned60, max_plots=1)
    # st.pyplot(fig)

    show = clinical_features_per_night.query(
        "pid == @PID & night_date >= @START_DATE & night_date <= @END_DATE"
    )
    st.dataframe(show)
