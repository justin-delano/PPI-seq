import pickle as pkl
import warnings

import dispersion_fitting
import helper_functions as hf
import numpy as np
import pandas as pd
import torch


def get_size_factors(X: pd.DataFrame, use_geom: bool = False) -> np.ndarray:
    """Calculate Size Factors

    Args:
        X (np.array): Count data from full experiment
        use_geom (bool, optional): Whether to replace arithmetic mean with geometric mean. Defaults to False.

    Returns:`
        np.array: Size factors per experiment
    """
    n_orfs, n_tags = X.shape

    if use_geom:
        mean_x = np.exp((1 / n_tags) * np.log(X + 0.5).sum(axis=1))
    else:
        mean_x = np.nanmean(X, axis=1)

    assert mean_x.shape == (n_orfs,)
    norm_count = X / mean_x[:, None]
    size_factor = np.nanmedian(norm_count, axis=0)
    assert size_factor.shape == (n_tags,)
    return size_factor


def main():
    """Main function"""
    dates = {
        "1028",
        "1115",
    }
    for date in dates:
        data_path = f"/data/pinello/PROJECTS/2022_PPIseq/data/{date}/{date}"

        reads = pd.read_excel(f"{data_path}_reads.xlsx", index_col=0).T.fillna(0)
        editfracs = pd.read_excel(f"{data_path}_editfrac.xlsx", index_col=0).T.fillna(0)
        q_design = pd.read_excel(f"{data_path}_q_design.xlsx", index_col=0)
        pi_design = pd.read_excel(f"{data_path}_pi_design.xlsx", index_col=0)
        groundtruth_labels = pd.read_excel(f"{data_path}_str_labels.xlsx", index_col=0)

        reads.drop(
            hf.str_subs(reads.index, ["CDNA", "PCR1"]),
            inplace=True,
        )
        editfracs.drop(
            hf.str_subs(editfracs.index, ["CDNA", "PCR1"]),
            inplace=True,
        )
        groundtruth_indices = groundtruth_labels.index[
            groundtruth_labels["24karat"].notna()
        ]
        reads = reads[groundtruth_indices]
        editfracs = editfracs[groundtruth_indices]
        reads_edited = np.around(reads.values * editfracs.values)
        reads_unedited = reads.values - reads_edited
        size_factors = torch.from_numpy(get_size_factors(reads.T)).unsqueeze(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dispersions = (
                torch.from_numpy(
                    dispersion_fitting.get_trend_fitted_dispersion(
                        reads.T,
                        q_design,
                    )[0].values
                )
                .unsqueeze(-1)
                .T
            )

        data_dict = {
            "reads_edited": torch.from_numpy(reads_edited),
            "reads_unedited": torch.from_numpy(reads_unedited),
            "size_factors": size_factors,
            "fit_dispersions": dispersions,
            "q_design_matrix": torch.from_numpy(q_design.values),
            "pi_design_matrix": torch.from_numpy(pi_design.values),
        }

        with open(f"{data_path}_pyro_dict.pkl", "wb") as file:
            pkl.dump(data_dict, file)


if __name__ == "__main__":
    main()
