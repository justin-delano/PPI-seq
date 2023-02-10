@dataclass
class Settings:
    filter_str: str = ""
    editrate_thresh: int = 0
    p_thresh: int = 0.05


def split_columns(df, extra_filter: str = "", cull_thresh: int = 0) -> tuple[pd.DataFrame]:
    label_colnames = [str(idx) for idx in row.index if LABEL_TO_TAG[row["label"]] in idx and extra_filter in idx]
    label_columns = row.loc[label_colnames]

    bg_colnames = [str(idx) for idx in row.index if LABEL_TO_TAG[row["label"]] not in idx and extra_filter in idx]
    if "label" in bg_colnames:
        bg_colnames.remove("label")

    bg_columns = row.loc[bg_colnames]
    bg_columns = bg_columns[bg_columns >= cull_thresh]

    return pd.to_numeric(label_columns), pd.to_numeric(bg_columns)

def model_predict(data: pd.DataFrame, labels: pd.Series, pred_fn: Callable, settings: Settings = Settings()) -> pd.Series:
    data["label"] = labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return data.apply(pred_fn, args=(settings,), axis=1).squeeze().rename("Predictions", inplace=True)


def model_metric(metric_fn: Callable):
    # metric_fn(labels[mask].to_list(),preds[mask].to_list()), preds[mask], labels[mask]
    ...

def rowmax_match(row, settings: Settings) -> str:
    guess = pd.to_numeric(row.drop("label")).idxmax()
    return TAG_TO_LABEL.get(guess.split()[0]) == row["label"]


def ttest_match(row, settings: Settings):
    label_columns, bg_columns = split_columns(row, settings.filter_str, settings.editrate_thresh)
    p_test = scipy.stats.ttest_ind(label_columns, bg_columns, alternative="greater").pvalue
    return p_test <= settings.p_thresh


# def kstest_match(row, dist: Callable|str, p_thresh:int):
#     label_columns, bg_columns = split_columns(row)
#     p_test = scipy.stats.ks_test(label_columns, bg_columns).pvalue
#     return p_test <= p_thresh

results = {}
results["rowmax"] = model_predict(editfrac50_data, labels, rowmax_match)
results["ttest1"] = model_predict(editfrac50_data, labels, ttest_match)
set3 = Settings(filter_str="1h")
results["ttest2"] = model_predict(editfrac50_data, labels, ttest_match, set3)
set4 = Settings(filter_str="3h")
results["ttest3"] = model_predict(editfrac50_data, labels, ttest_match, set4)
set5 = Settings(editrate_thresh=0.1)
results["ttest4"] = model_predict(editfrac50_data, labels, ttest_match, set5)

for test, result in results.items():
    print(f"{test} accuracy is {round(sum(result)/len(result),3)}")

    def plot_reads_editfrac(read_data:pd.DataFrame, editfrac_data:pd.DataFrame, tags:list[str], lobf:bool=False):
    for tag in tags[-2:]:
        idx = np.isfinite(read_data[tag]) & read_data[tag] > 0
        x_data = np.log10(read_data[tag][idx])
        y_data = editfrac_data[tag][idx]
        plt.scatter(x_data, y_data, label=tag, s=5)
        if lobf:
            slope, intercept, r_value, _, _ = scipy.stats.linregress(x_data, y_data)
            plt.plot(x_data, slope*x_data+intercept, label = f"R^2 = {round(r_value,2)}, m={round(slope,2)}")
    plt.legend(loc='best')
    plt.xlabel('Log10(cDNA Read Counts)')
    plt.ylabel('Edit Fraction')

plt.style.use('default')
for tag in TAG_TO_LABEL:
    plt.figure()
    tag_cols = [str(col) for col in read_data.columns if tag in col]
    plot_reads_editfrac(read_data, editfrac50_data, tag_cols, lobf=True)

plt.style.use('default')
def difference(x1,x2):
    return x2-x1
def log2FC(x1,x2):
    # x1[x1==0] = 0.0001
    # x2[x2==0] = 0.0001
    y = np.log2(x2)-np.log2(x1)
    # y[~np.isfinite(y)] = 0
    return y[np.isfinite(y)]
def ratio(x1,x2):
    # x1[x1==0] = 0.0001
    # x2[x2==0] = 0.0001
    y = x1/x2
    # y[~np.isfinite(y)] = 0
    return y[np.isfinite(y)]

def plot_normalizing_dist(editfrac_data:pd.DataFrame, query_tag:str|list[str], norm_tag:str|list[str], norm_fn:Callable):
    query_column = editfrac_data[query_tag].mean(axis=1)
    # query_zeros_idx = query_column==0 | np.isnan(query_column)

    norm_column = editfrac_data[norm_tag].mean(axis=1)
    # norm_zeros_idx = norm_column==0 | np.isnan(norm_column)

    # query_column = query_column[~query_zeros_idx & ~norm_zeros_idx]
    # norm_column = norm_column[~query_zeros_idx & ~norm_zeros_idx]

    results = norm_fn(query_column, norm_column)
    sns.kdeplot(results, label=f"{query_tag[0].split()[0]}/median vs. {norm_tag[0].split()[0]}/median")

mean_EF50 = editfrac50_data / editfrac50_data.mean(axis=0)
median_EF50 = editfrac50_data / editfrac50_data.median(axis=0)

norm_cols = [str(col) for col in editfrac50_data.columns if list(TAG_TO_LABEL)[0] in col and "3h" in col]
for norm_fn in [difference, log2FC, ratio]:
    plt.figure()
    for tag in list(TAG_TO_LABEL)[1:]:
        query_cols = [str(col) for col in editfrac50_data.columns if tag in col and "3h" in col]
        plot_normalizing_dist(median_EF50, query_cols, norm_cols, norm_fn=norm_fn)
    plt.legend()
    plt.xlabel(f"Metric: {norm_fn.__name__}")