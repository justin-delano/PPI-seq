#%%
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn import metrics
import statsmodels.api as sm
from matplotlib import style
from scipy.stats import norm

np.seterr(all="ignore")

# style.use("dark_background")
TAG_TO_LABEL = {
    "NES": "Cytosol",
    "NLS": "Nuclear",
    "MEM": "Secretory",
    "OMM": "Mitochondria",
}
LABEL_TO_TAG = {
    "Cytosol": "NES",
    "Nuclear": "NLS",
    "Secretory": "MEM",
    "Mitochondria": "OMM",
}

#%%
data = {}

data["09/22"] = {}
data["09/22"]["labels"] = pd.read_excel(
    "../data/0922/0922_24karat_groundtruth.xlsx", index_col=0
).dropna(how="all")
data["09/22"]["reads"] = pd.read_excel("../data/0922/0922_reads.xlsx", index_col=0).loc[
    data["09/22"]["labels"].index
]
data["09/22"]["editfrac"] = pd.read_excel(
    "../data/0922/0922_editfrac_over50.xlsx", index_col=0
).loc[data["09/22"]["labels"].index]

data["10/28"] = {}
data["10/28"]["labels"] = pd.read_excel(
    "../data/1028/1028_24karat_groundtruth.xlsx", index_col=0
).dropna(how="all")
data["10/28"]["reads"] = pd.read_excel("../data/1028/1028_reads.xlsx", index_col=0).loc[
    data["10/28"]["labels"].index
]
data["10/28"]["editfrac"] = pd.read_excel("../data/1028/1028_editfrac.xlsx", index_col=0).loc[
    data["10/28"]["labels"].index
]

data["11/15"] = {}
data["11/15"]["labels"] = pd.read_excel(
    "../data/1115/1115_24karat_groundtruth.xlsx", index_col=0
).dropna(how="all")
data["11/15"]["reads"] = pd.read_excel("../data/1115/1115_reads.xlsx", index_col=0).loc[
    data["11/15"]["labels"].index
]
data["11/15"]["editfrac"] = pd.read_excel("../data/1115/1115_editfrac.xlsx", index_col=0).loc[
    data["11/15"]["labels"].index
]

#%%
# https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
def get_concordance(x, y):
    """Concordance correlation coefficient."""
    # Remove NaNs
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    x = df["x"]
    y = df["y"]
    cor = np.corrcoef(x, y)[0][1]
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    sd_x, sd_y = np.std(x), np.std(y)
    # Calculate CCC
    numerator = 2 * cor * sd_x * sd_y
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    return numerator / denominator


def rep_concordance(data_df: pd.DataFrame, labels: pd.DataFrame, data_type: str, tag: str):
    ax = plt.gca()

    rep_columns = [
        col
        for col in data_df.columns.to_list()
        if tag in col and "3H" in col.upper() and "REPLICATE" not in col
    ]

    nonmatch_x_data = data_df[rep_columns[0]][labels[f"ABE-{tag}"] != 1]
    nonmatch_y_data = data_df[rep_columns[1]][labels[f"ABE-{tag}"] != 1]
    match_x_data = data_df[rep_columns[0]][labels[f"ABE-{tag}"] == 1]
    match_y_data = data_df[rep_columns[1]][labels[f"ABE-{tag}"] == 1]

    ax.scatter(
        nonmatch_x_data,
        nonmatch_y_data,
        label=f"Negatives, CCC = {round(get_concordance(nonmatch_x_data, nonmatch_y_data),2)}",
        s=5,
    )
    ax.scatter(
        match_x_data,
        match_y_data,
        label=f"Positives, CCC = {round(get_concordance(match_x_data, match_y_data),2)}",
        s=5,
    )
    ax.set(title=f"{tag} {data_type}")

    ax.axline((0, 0), slope=1)
    ax.legend(loc="best")
    ax.set(xlabel=f"Rep A {data_type}", ylabel=f"Rep B {data_type}")
    if "read" in data_type.lower():
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.axis("square")


def concordance_plots(date: str, data_dict: dict) -> None:
    plt.style.use("default")
    for tag in TAG_TO_LABEL:
        _, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        plt.suptitle(f"{date} {tag} Replicate Concordance")
        plt.sca(ax[0])
        rep_concordance(data_dict[date]["reads"], data_dict[date]["labels"], "Read Counts", tag)
        plt.sca(ax[1])
        rep_concordance(data_dict[date]["editfrac"], data_dict[date]["labels"], "Edit Fractions", tag)


# %%
concordance_plots("09/22", data)
# %%
