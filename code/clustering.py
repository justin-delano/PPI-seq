#%%
import typing
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn import cluster, metrics, preprocessing

#%%
data = {}
for date in ["09/22", "10/28", "11/15"]:
    data[date] = {}
    path_start = f"../data/{date.replace('/','')}/{date.replace('/','')}_"
    data[date]["str_lab"] = pd.read_excel(path_start + "str_labels.xlsx", index_col=0)
    data[date]["bin_lab"] = pd.read_excel(path_start + "bin_labels.xlsx", index_col=0)
    data[date]["reads"] = pd.read_excel(path_start + "reads.xlsx", index_col=0)
    data[date]["editfrac"] = pd.read_excel(path_start + "editfrac.xlsx", index_col=0)
# %%
def get_size_factor(X: np.array):
    mean_x = np.mean(X, axis=1)
    # geom_mean_x = np.exp((1 / n_samples) * np.log(X + 0.5).sum(axis=1))
    norm_count = X / mean_x[:, None]
    size_factor = np.median(norm_count, axis=0)
    return size_factor


def flatten_rep_cols(data: pd.DataFrame, func: Callable) -> pd.DataFrame:
    data = data[
        [
            col
            for col in data.columns.to_list()
            if "3H" in col.upper() and "REPLICATE" not in col and "CDNAREP" not in col
        ]
    ]
    flat_df = pd.DataFrame(index=data.index)
    tags = {str(col).split("_", maxsplit=1)[0] for col in data.columns}
    for tag in tags:
        flat_df[f"ABE-{tag}"] = data.filter(regex=f"(?=.*{tag})(?=.*{'3H'})", axis=1).apply(
            func, raw=True, axis=1
        )
    return flat_df


y_true = (
    data["10/28"]["str_lab"]["24karat"]
    .map({"Cytosol": 0, "Secretory": 1, "Mitochondria": 2, "Nuclear": 3})
    .dropna()
)
test_data = data["10/28"]["editfrac"].loc[y_true.index].dropna()
test_data = test_data[
    [
        col
        for col in test_data.columns.to_list()
        if "3H" in col.upper() and "REPLICATE" not in col and "CDNAREP" not in col
    ]
]
y_true = y_true.loc[test_data.index]


# %%
size_factors = get_size_factor(np.array(test_data))
size_factors[size_factors == 0] = 1
test_data = test_data.div(size_factors, axis=1)
# test_data = test_data.div(test_data.mean(axis=0), axis=1)


# %%
reducer = umap.UMAP(random_state=42)
# scaled_data = preprocessing.StandardScaler().fit_transform(test_data)
# embedding = reducer.fit_transform(scaled_data)
embedding = reducer.fit_transform(test_data)

# ranked_data = preprocessing.quantile_transform(test_data)
# embedding = reducer.fit_transform(ranked_data)

fig, ax = plt.subplots()
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y_true, s=5)
ax.legend(
    handles=scatter.legend_elements()[0],
    labels=["ABE-NES", "ABE-MEM", "ABE-OMM", "ABE-NLS"],
    title="Classes",
)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
# %%

# https://scikit-learn.org/stable/modules/clustering.html#fowlkes-mallows-scores
ward = cluster.AgglomerativeClustering(n_clusters=4, linkage="ward")
complete = cluster.AgglomerativeClustering(n_clusters=4, linkage="complete")
average = cluster.AgglomerativeClustering(n_clusters=4, linkage="average")
single = cluster.AgglomerativeClustering(n_clusters=4, linkage="single")
kmeans = cluster.KMeans(n_clusters=4)
spectral = cluster.SpectralClustering(
    n_clusters=4,
    assign_labels="discretize",
)

for algo in [ward, complete, average, single]:
    algo.fit(embedding)

    if hasattr(algo, "labels_"):
        y_pred = algo.labels_.astype(int)
    else:
        y_pred = algo.predict(embedding)

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, s=5)
    ax.legend(
        handles=scatter.legend_elements()[0],
        # labels=["ABE-NES", "ABE-MEM", "ABE-OMM", "ABE-NLS"],
        title="Classes",
    )
    ax.set_title(f"Agglomerative Clustering, linkage={algo.linkage}")
    plt.show()
    print("Fowlkes-Mallows Score:", metrics.fowlkes_mallows_score(y_true, y_pred))


# %%

# Questions:
# handling NAs?
# Type of clustering?
# Standardization/normalization?
# Cluster on umap or full?
# single replicate or combine replicates?
# rank normalized clustering (k-medians/manhattan)?
sum(
    "ABE-" + test_data.rank(axis=1, pct=True).idxmax(axis=1).str.split("_", n=1, expand=True)[0]
    == data["10/28"]["bin_lab"].loc[test_data.index, :].idxmax(axis=1)
) / len(test_data.index)

# %%
