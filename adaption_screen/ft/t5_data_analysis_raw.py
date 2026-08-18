# %%
import pandas as pd
import numpy as np
from IPython.display import display

# %%
df_pred = pd.read_csv("infer_test.csv")
df_pred

# %%
df_raw = pd.read_csv("../data/dataset_qc.csv")
df_raw

# %%
# If df_pred has duplicate ID-dose combinations, deduplicate df_pred first to avoid inflating df_raw:
df_filtered = pd.merge(
    df_raw, 
    df_pred[['ID', 'dose']].drop_duplicates(), 
    on=['ID', 'dose'], 
    how='inner'
)
df_filtered

# %%
# RPA wide table
rpa_wide = (
    df_filtered.pivot_table(
        index=["ID", "dose"],
        columns="Accession",
        values="RPA",
        aggfunc="mean"   # If the same ID/dose/Accession has duplicates, keep the first
    )
    .reset_index()
)

# pred_prob wide table
pred_prob_wide = (
    df_pred.pivot_table(
        index=["ID", "dose"],
        columns="Accession",
        values="pred_prob",
        aggfunc="mean"
    )
    .reset_index()
)

# Remove column names to clean the header
rpa_wide.columns.name = None
pred_prob_wide.columns.name = None

# %%
rpa_wide


# %%
id_dose_same = rpa_wide[["ID", "dose"]].equals(pred_prob_wide[["ID", "dose"]])
columns_same = rpa_wide.columns.equals(pred_prob_wide.columns)

print("ID 和 dose 是否完全一致:", id_dose_same)
print("列名是否完全一致，包括顺序:", columns_same)


# %%
rpa_wide.insert(
    0,
    "ID_dose",
    rpa_wide["ID"].astype(str) + "_" + rpa_wide["dose"].astype(str)
)
rpa_wide.drop(columns=["ID", "dose"], inplace=True)
rpa_wide

# %%
rpa_wide_T = rpa_wide.set_index("ID_dose").T.reset_index()
rpa_wide_T = rpa_wide_T.rename(columns={"index": "Accession"})
rpa_wide_T

# %%
# Remove column names to clean the header
rpa_wide_T.columns.name = None
rpa_wide_T

# %%
pred_prob_wide

# %%
pred_prob_wide.insert(
    0,
    "ID_dose",
    pred_prob_wide["ID"].astype(str) + "_" + pred_prob_wide["dose"].astype(str)
)
pred_prob_wide.drop(columns=["ID", "dose"], inplace=True)
pred_prob_wide

# %%
pred_prob_wide_T = pred_prob_wide.set_index("ID_dose").T.reset_index()
pred_prob_wide_T = pred_prob_wide_T.rename(columns={"index": "Accession"})
pred_prob_wide_T

# %%
# Remove column names to clean the header
pred_prob_wide_T.columns.name = None
pred_prob_wide_T


# %%
def protein_abundance_metrics_from_column(df, col, threshold=0):
    """
    计算一列蛋白相对丰度的覆盖度、非零数量和均匀度指标。

    Parameters
    ----------
    df : pandas.DataFrame
        输入数据框。
    col : str
        蛋白丰度所在列名。
    threshold : float, default=0
        检测阈值。丰度 > threshold 的蛋白被认为是检测到。
        如果你的数据中非常小的非零值可能是噪声，建议设为如 1e-6 或 0.001。

    Returns
    -------
    metrics : dict
        包含覆盖度、非零数量、Shannon entropy、Pielou evenness 等指标。
    """

    x = df[col].astype(float).to_numpy()

    # Total number of proteins
    N = len(x)

    # Detected proteins, default x > 0
    detected = x > threshold
    x_detected = x[detected]

    # Number of nonzero proteins or proteins above threshold
    S = len(x_detected)

    # Coverage
    coverage = S / N if N > 0 else np.nan

    # If no protein is detected
    if S == 0:
        return {
            "total_proteins": N,
            "detected_proteins": S,
            "coverage": coverage,
            "shannon_entropy": np.nan,
            "pielou_evenness": np.nan,
            "shannon_effective_number": np.nan,
            "simpson_index": np.nan,
            "simpson_effective_number": np.nan,
            "gini_nonzero": np.nan
        }

    # Normalize to probabilities
    p = x_detected / x_detected.sum()

    # Shannon entropy
    shannon_entropy = -np.sum(p * np.log(p))

    # Pielou's evenness
    if S > 1:
        pielou_evenness = shannon_entropy / np.log(S)
    else:
        pielou_evenness = 1.0

    # Shannon effective number
    shannon_effective_number = np.exp(shannon_entropy)

    # Simpson index
    simpson_index = np.sum(p ** 2)

    # Simpson effective number
    simpson_effective_number = 1 / simpson_index

    # Gini coefficient
    def gini(array):
        array = np.asarray(array, dtype=float)
        array = np.sort(array)
        n = len(array)
        if n == 0 or array.sum() == 0:
            return np.nan
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * array) / (n * np.sum(array))) - (n + 1) / n

    gini_nonzero = gini(x_detected)

    metrics = {
        "total_proteins": N,
        "detected_proteins": S,
        "coverage": coverage,
        "shannon_entropy": shannon_entropy,
        "pielou_evenness": pielou_evenness,
        "shannon_effective_number": shannon_effective_number,
        "simpson_index": simpson_index,
        "simpson_effective_number": simpson_effective_number,
        "gini_nonzero": gini_nonzero
    }

    return metrics

# %%
np_cols = rpa_wide_T.columns.difference(["Accession"])
print("RPA 列名：", np_cols.tolist())

# %%
# Compute protein abundance metrics for each RPA column and arrange them into a new DataFrame

metrics_list = []

for col in np_cols:
    metrics = protein_abundance_metrics_from_column(rpa_wide_T, col, threshold=1e-6)
    
    # Add the current sample/column name
    metrics["sample"] = col
    
    metrics_list.append(metrics)

# Convert to DataFrame
df_rpa_metrics = pd.DataFrame(metrics_list)

# Move the sample column to the front
df_rpa_metrics = df_rpa_metrics[
    ["sample"] + [c for c in df_rpa_metrics.columns if c != "sample"]
]

df_rpa_metrics

# %%
df_rpa_metrics.describe()

# %%
# Sort by coverage in descending order and save
df_rpa_metrics_coverage_sorted = df_rpa_metrics.sort_values(
    by="coverage",
    ascending=False
)

display(df_rpa_metrics_coverage_sorted)


# %%
df_rpa_metrics_coverage_sorted.to_csv(
    "exper_stat.csv",
    index=False
)

# %%
