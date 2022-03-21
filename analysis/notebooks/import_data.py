import os
import numpy as np
import pandas as pd
from mappings import *


def import_from_all_phq(f):
    df = pd.read_csv(
        f, usecols=['patient id', 'time stamp', 'phq score', 'sleep score'])
    return df


def import_data(dir):
    centres = ['ciber', 'iispv', 'kcl', 'vumc']
    phq_files = [dir / ('all_phq_' + centre + '.csv') for centre in centres]
    slp_files = [dir / ('all_sleep_features_timing_' + centre + '.csv')
                 for centre in centres]

    phqs = pd.concat(map(pd.read_csv, phq_files), keys=centres,
                     names=['centre', 'index']).reset_index(level=0)
    slps = pd.concat(map(pd.read_csv, slp_files), keys=centres,
                     names=['centre', 'index']).reset_index(level=0)

    phqs.rename(columns=get_map_phq_colnames(), inplace=True)
    slps.rename(columns=get_map_slp_colnames(), inplace=True)

    metadata_path = dir / 'radar_mdd_ids.xlsx'
    metadata = pd.read_excel(metadata_path)

    return(phqs, slps, metadata)
