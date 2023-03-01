import numpy as np
import pandas as pd

for date in ["0922", "1028", "1115"]:
    path = f"../data/{date}/{date}"

    read_data = pd.read_excel(f"{path}_reads.xlsx", index_col=0)
    editfrac_data = pd.read_excel(f"{path}_editfrac.xlsx", index_col=0)

    edited_read_data = pd.DataFrame(
        np.around(read_data.values * editfrac_data.values),
        columns=read_data.columns,
        index=read_data.index,
    )

    unedited_read_data = read_data - edited_read_data


    edited_read_data.to_excel(f"{path}_reads_edited.xlsx")
    unedited_read_data.to_excel(f"{path}_reads_unedited.xlsx")
