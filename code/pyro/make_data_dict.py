import pickle as pkl

import dispersion_fitting
import numpy as np
import pandas as pd


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
    date = "0922"
    data_path = f"/data/pinello/PROJECTS/2022_PPIseq/data/{date}/{date}"

    reads_all = pd.read_excel(f"{data_path}_reads.xlsx", index_col=0)
    editfracs = pd.read_excel(f"{data_path}_editfrac.xlsx", index_col=0)
    D_exp = pd.read_excel(f"{data_path}_D_exp.xlsx", index_col=0)
    reads_edited = pd.DataFrame(
        np.around(reads_all.values * editfracs.values),
        columns=reads_all.columns,
        index=reads_all.index,
    )
    reads_unedited = reads_all - reads_edited

    data_dict = {
        "reads_edited": reads_edited,
        "reads_unedited": reads_unedited,
        "size_factors": get_size_factors(reads_all),
        "fit_dispersions": dispersion_fitting.get_trend_fitted_dispersion(
            reads_all,
            D_exp.drop(["Localized", "Dispersed", "EditEnriched"], axis=1),
        ),
        "D_exp": D_exp,
    }
    # with open(f"{data_path}_data_dict.pkl", "wb") as file:
    #     pkl.dump(data_dict, file)


if __name__ == "__main__":
    main()
