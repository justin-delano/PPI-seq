import pickle as pkl

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
    dates_reps = {
        "1028": ["REPA_", "REPB_", "REPC_"],
        "1115": ["REPA_", "REPB_", "REPC_"],
    }
    for date, reps in dates_reps.items():
        data_path = f"/data/pinello/PROJECTS/2022_PPIseq/data/{date}/{date}"

        reads = pd.read_excel(f"{data_path}_reads.xlsx", index_col=0)
        editfracs = pd.read_excel(f"{data_path}_editfrac.xlsx", index_col=0)
        pyro_design = pd.read_excel(f"{data_path}_pyro_design.xlsx", index_col=0)
        disp_design = pd.read_excel(f"{data_path}_disp_design.xlsx", index_col=0)
        groundtruth_labels = pd.read_excel(f"{data_path}_str_labels.xlsx", index_col=0)

        reads.drop(
            hf.str_subs(reads.columns, ["CDNA", "PCR1"]),
            inplace=True,
            axis=1,
        )
        editfracs.drop(
            hf.str_subs(editfracs.columns, ["CDNA", "PCR1"]),
            inplace=True,
            axis=1,
        )
        groundtruth_indices = groundtruth_labels.index[
            groundtruth_labels["24karat"].notna()
        ]
        reads = reads.loc[groundtruth_indices]
        editfracs = editfracs.loc[groundtruth_indices]

        reads_edited_final = torch.zeros(
            size=[len(reps), len(reads.index), len(pyro_design.index)]
        )
        reads_unedited_final = torch.zeros(
            size=[len(reps), len(reads.index), len(pyro_design.index)]
        )

        size_factors = torch.zeros(size=[len(reps), len(pyro_design.index)])
        dispersions = torch.zeros(size=[len(reads.index), len(pyro_design.index)])

        # ! this is super broken, probably due to small number of replicates
        for exp_num, exp in enumerate(pyro_design.index):
            exp_cols = hf.get_matching_replicates(
                general_sample=exp, all_samples=reads.columns, replicates=reps
            )

            print(reads[exp_cols].fillna(0) + 0.5)
            print(disp_design.loc[disp_design.index.isin(exp_cols)])
            # bkleh
            dispersions[:, exp_num] = torch.from_numpy(
                dispersion_fitting.get_trend_fitted_dispersion(
                    reads[exp_cols].fillna(0) + 0.5,
                    disp_design.loc[disp_design.index.isin(exp_cols)],
                )[0].values
            )

        for rep_num, rep in enumerate(reps):
            reads_rep = reads[hf.str_subs(reads.columns, rep)]
            editfracs_rep = editfracs[hf.str_subs(editfracs.columns, rep)]

            reads_edited = np.around(reads_rep.values * editfracs_rep.values)
            reads_unedited = reads_rep.values - reads_edited

            reads_edited_final[rep_num, :, :] = torch.from_numpy(reads_edited)
            reads_unedited_final[rep_num, :, :] = torch.from_numpy(reads_unedited)
            size_factors[rep_num, :] = torch.from_numpy(get_size_factors(reads_rep))

        data_dict = {
            "reads_edited": reads_edited_final,
            "reads_unedited": reads_unedited_final,
            "size_factors": size_factors,
            "fit_dispersions": dispersions,
            "design_matrix": pyro_design,
        }

        with open(f"{data_path}_pyro_dict.pkl", "wb") as file:
            pkl.dump(data_dict, file)


if __name__ == "__main__":
    main()
