import helper_functions as hf
import pandas as pd


def make_pyro_design_matrix(date: str, reps: list[str]):
    data_path = f"/data/pinello/PROJECTS/2022_PPIseq/data/{date}/{date}"

    read_data = pd.read_excel(f"{data_path}_reads_edited.xlsx", index_col=0)
    read_data.drop(
        hf.str_subs(read_data.columns, ["CDNA", "PCR1"]), inplace=True, axis=1
    )

    tags = list({col.split("_")[0] for col in read_data.columns})

    samples = []
    for column in read_data.columns:
        for rep in reps:
            column = column.replace(rep, "")
        samples.append(column)

    design_matrix = pd.DataFrame(
        0,
        index=hf.ordered_unique(samples),
        columns=["Baseline", "1H", "3H", "EditEnriched"] + tags,
    )

    design_matrix["Baseline"] = 1
    design_matrix["1H"][hf.str_subs(design_matrix.index, ["1h", "1H"])] = 1
    design_matrix["3H"][hf.str_subs(design_matrix.index, ["3h", "3H"])] = 1
    design_matrix["EditEnriched"][
        hf.str_subs(design_matrix.index, ["PSTI", "PTSI"])
    ] = 1

    for tag in tags:
        design_matrix[tag][hf.str_subs(design_matrix.index, tag)] = 1
    design_matrix.to_excel(f"{data_path}_pyro_design.xlsx")


def make_disp_design_matrix(date: str, reps: list[str]):
    data_path = f"/data/pinello/PROJECTS/2022_PPIseq/data/{date}/{date}"

    read_data = pd.read_excel(f"{data_path}_reads_edited.xlsx", index_col=0)
    read_data.drop(
        hf.str_subs(read_data.columns, ["CDNA", "PCR1"]), inplace=True, axis=1
    )

    design_matrix = pd.DataFrame(1, index=read_data.columns, columns=["Baseline"])

    for rep in reversed(reps):
        design_matrix.insert(loc=1, column=rep, value=0)
        design_matrix[rep][hf.str_subs(design_matrix.index, rep)] = 1

    design_matrix.to_excel(f"{data_path}_disp_design.xlsx")


def main():
    """Main function"""
    dates_reps = {
        "1028": ["REPA_", "REPB_", "REPC_"],
        "1115": ["REPA_", "REPB_", "REPC_"],
    }
    for date, reps in dates_reps.items():
        make_pyro_design_matrix(date, reps)
        make_disp_design_matrix(date, reps)


if __name__ == "__main__":
    main()
