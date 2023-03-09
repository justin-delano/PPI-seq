from typing import Iterable

import pandas as pd


def str_subs(
    strs: Iterable, subs: str | list[str]  # pylint: disable=E1131
) -> list[str]:
    """Find strings in list containing one of given substrings

    Args:
        strs (Iterable): List of strings (or pandas index)
        subs (str | list[str]): Substrings to match

    Returns:
        list[str]: Strings that contain one of given substrings
    """

    if isinstance(subs, str):
        subs = [subs]
    return [str for str in strs if any(sub in str for sub in subs)]


def main():
    """Main function"""

    for date in ["0922", "1028", "1115"]:
        data_path = f"/data/pinello/PROJECTS/2022_PPIseq/data/{date}/{date}"

        read_data = pd.read_excel(f"{data_path}_reads_edited.xlsx", index_col=0)

        D_exp = pd.DataFrame(
            0,
            index=read_data.columns,
            columns=[
                "Baseline",
                "RepA",
                "RepB",
                "1H",
                "3H",
                "Localized",
                "Dispersed",
                "EditEnriched",
            ],
        )
        D_exp.drop(str_subs(D_exp.index, ["CDNA", "PCR1"]), inplace=True)

        D_exp["Baseline"] = 1
        if date == "0922":
            D_exp["RepA"][str_subs(D_exp.index, " A")] = 1
            D_exp["RepA"][
                [row for row in D_exp.index if " A" not in row and " B" not in row]
            ] = 1
            D_exp["RepB"][str_subs(D_exp.index, " B")] = 1
        else:
            D_exp.insert(loc=3, column="RepC", value=0)
            D_exp["RepA"][str_subs(D_exp.index, "REPA")] = 1
            D_exp["RepB"][str_subs(D_exp.index, "REPB")] = 1
            D_exp["RepC"][str_subs(D_exp.index, "REPC")] = 1

        D_exp["1H"][str_subs(D_exp.index, ["1h", "1H"])] = 1
        D_exp["3H"][str_subs(D_exp.index, ["3h", "3H"])] = 1

        D_exp["Dispersed"][str_subs(D_exp.index, "NES")] = 1
        D_exp["Localized"][str_subs(D_exp.index, "OMM")] = 1
        D_exp["EditEnriched"][str_subs(D_exp.index, ["PTSI", "PSTI"])] = 1

        D_exp.to_excel(f"{data_path}_D_exp.xlsx")


if __name__ == "__main__":
    main()
