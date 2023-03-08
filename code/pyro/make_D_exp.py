from typing import Iterable

import pandas as pd


def str_subs(strs: Iterable, subs: str | list[str]) -> list[str]:
    if isinstance(subs, str):
        subs = [subs]
    return [str for str in strs if any(sub in str for sub in subs)]


for date in ["0922", "1028", "1115"]:
    path = f"../../data/{date}/{date}"

    read_data = pd.read_excel(f"{path}_reads_edited.xlsx", index_col=0)
    D_exp = pd.DataFrame(
        0,
        index=read_data.columns,
        columns=["Baseline", "1H", "3H", "Nomadic", "Settled", "EditEnriched"],
    )
    D_exp.drop(str_subs(D_exp.index, ["CDNA", "PCR1"]), inplace=True)

    D_exp["Baseline"] = 1
    D_exp["1H"][str_subs(D_exp.index, ["1h", "1H"])] = 1
    D_exp["3H"][str_subs(D_exp.index, ["3h", "3H"])] = 1
    D_exp["Nomadic"][str_subs(D_exp.index, "NES")] = 1
    D_exp["Settled"][str_subs(D_exp.index, "OMM")] = 1
    D_exp["EditEnriched"][str_subs(D_exp.index, ["PTSI", "PSTI"])] = 1

    D_exp.to_excel(f"{path}_D_exp.xlsx")
