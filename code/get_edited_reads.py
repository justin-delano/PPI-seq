import numpy as np
import pandas as pd

read_data = pd.read_excel("0922_PPIseqdata_reads.xlsx", index_col=0)
editfrac_data = pd.read_excel("0922_PPIseqdata_editfrac.xlsx", index_col=0)

edited_read_data = pd.DataFrame(
    np.around(read_data.values * editfrac_data.values),
    columns=read_data.columns,
    index=read_data.index,
)

edited_read_data.to_csv("0922_PPIseqdata_editedreads.csv")
