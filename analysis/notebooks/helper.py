import pandas as pd
import numpy as np
import os
import logging
import time
import winsound


class Timer(object):
    """
    helper class to timing blocks of code
    Source: https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            print(
                "# [%s]" % self.name,
            )

    def __exit__(self, type, value, traceback):
        print("Elapsed: %.1f seconds" % (time.time() - self.tstart))


def is_consecutive(l):
    ### usage: out2 = index_PID.groupby('PatientID')['index'].apply(list).apply(is_consecutive)
    return sorted(l) == list(range(min(l), max(l) + 1))


def vc2df(s):
    df = pd.DataFrame(s.reset_index())
    df.columns = [s.index.name, "count"]
    return df


def check_file_accesible(f):
    if os.path.exists(f):
        try:
            os.rename(f, f)
        except OSError as error:
            print(f"Access-error on file {f}! {str(error)}")


def flatten_multiindex(df, join_by):
    #! todo: check whether a is a tuple, or the input has multiindex
    dt = df.copy()
    dt.columns = [join_by.join(a) for a in dt.columns.to_flat_index()]
    return dt


def cartesian_product(*arrays):
    """Cartersian product from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/11146645#11146645

    Returns:
        array: cartesian product of input arays
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


# def dtale_url(obj, **kargs):
#     d = dtale.show(obj, **kargs)
#     return d._main_url


def beeps():
    winsound.Beep(440, 200)
    winsound.Beep(554, 200)
    winsound.Beep(660, 200)
    winsound.Beep(880, 200)


def generate_parse_config_script(config, config_name="config"):
    """Generate the script to parse a config dictionary (e.g. parsed from JSON) to global variables

    Args:
        config (dict): Config dictionary parsed from e.g. JSON
        config_name (str, optional): name of the config dictionary. Defaults to 'config'.
    """
    for key in config.keys():
        print(f"{key} = {config_name}['{key}']")


def time2second(x: pd.Series) -> pd.Series:
    """Convert datetime series time component to number of seconds

    Args:
        x (pd.Series): Pandas series of datetime objects

    Returns:
        pd.Series: Pandas series of the time component in seconds
    """
    return (x.dt.hour * 3600 + x.dt.minute * 60 + x.dt.second).astype(int)
