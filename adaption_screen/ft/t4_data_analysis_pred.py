# %%
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# %%
df_test = pd.read_csv("infer_test.csv")
df_test

# %%
history = np.load("output/fine_tuning/history_ft.npy", allow_pickle=True).item()

print(history.keys())

# 找到 val_aupr 最大的 epoch
best_idx = np.argmax(history["aupr"])

# 对应的最大 AUPR 和 threshold
best_aupr = history["aupr"][best_idx]
threshold = history["best_thresh"][best_idx]

print("Best epoch:", best_idx + 1)
print("Best AUPR:", best_aupr)
print("Best threshold:", threshold)

# %%
y_true = df_test["Protein corona composition"].astype(int).values
preds = df_test["pred_label"].astype(int).values
probs = df_test["pred_prob"].values

acc = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds, zero_division=0)
recall = recall_score(y_true, preds, zero_division=0)
f1 = f1_score(y_true, preds, zero_division=0)

auroc = roc_auc_score(y_true, probs)
auprc = average_precision_score(y_true, probs)

cm = confusion_matrix(y_true, preds)

mean_five = (acc + precision + recall + auroc + auprc)/5

print("===== Evaluation Metrics =====")
print(f"Threshold : {threshold}")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUROC     : {auroc:.4f}")
print(f"AUPRC     : {auprc:.4f}")
print(f"Mean(five): {mean_five:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, preds, digits=4, zero_division=0))

# %%
def evaluate_metrics_by_id(df, threshold=None):
    results = []

    for id_name, g in df.groupby("ID"):
        y_true = g["Protein corona composition"].astype(int).values
        y_pred = g["pred_label"].astype(int).values
        y_prob = g["pred_prob"].values

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUROC/AUPRC 要求 y_true 至少有两个类别
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
        else:
            auroc = np.nan
            auprc = np.nan

        mean_five = np.nanmean([acc, precision, recall, auroc, auprc])

        results.append({
            "ID": id_name,
            "threshold": threshold,
            "n_samples": len(g),
            "n_pos": int(np.sum(y_true == 1)),
            "n_neg": int(np.sum(y_true == 0)),
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Mean(five)": mean_five
        })

    return pd.DataFrame(results)

id_metrics_df = evaluate_metrics_by_id(df_test, threshold=threshold)

id_metrics_df

# %%
def evaluate_metrics_by_prob_bin(
    df,
    prob_col="pred_prob",
    y_true_col="Protein corona composition",
    y_pred_col="pred_label",
    bin_width=0.1
):

    bins=np.arange(0,1+bin_width,bin_width)

    df=df.copy()

    df["prob_bin"]=pd.cut(
        df[prob_col],
        bins=bins,
        include_lowest=True,
        right=False
    )

    results=[]

    for bin_name,g in df.groupby("prob_bin"):

        if len(g)==0:
            continue

        y_true=g[y_true_col].astype(int).values
        y_pred=g[y_pred_col].astype(int).values
        y_prob=g[prob_col].values

        acc=accuracy_score(y_true,y_pred)

        precision=precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall=recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1=f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if len(np.unique(y_true))>1:

            auroc=roc_auc_score(
                y_true,
                y_prob
            )

            auprc=average_precision_score(
                y_true,
                y_prob
            )

        else:

            auroc=np.nan
            auprc=np.nan

        results.append({

            "prob_bin":str(bin_name),

            "n_samples":len(g),

            "avg_prob":g[prob_col].mean(),

            "pos_rate":np.mean(y_true),

            "Accuracy":acc,

            "Precision":precision,

            "Recall":recall,

            "F1":f1,

            "AUROC":auroc,

            "AUPRC":auprc

        })

    return pd.DataFrame(results)


prob_metrics = evaluate_metrics_by_prob_bin(
    df_test,
    bin_width=0.1
)

prob_metrics


# %%
df = df_test.copy()
df

# %%

# pred_prob 宽表
pred_prob_wide = (
    df.pivot_table(
        index=["ID", "dose"],
        columns="Accession",
        values="pred_prob",
        aggfunc="mean"
    )
    .reset_index()
)

# 去掉 columns 的名字，使表头更干净
pred_prob_wide.columns.name = None



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
# 去掉 columns 的名字，使表头更干净
pred_prob_wide_T.columns.name = None
pred_prob_wide_T

# %%
history = np.load("output/fine_tuning/history_ft.npy", allow_pickle=True).item()

print(history.keys())

# 找到 val_aupr 最大的 epoch
best_idx = np.argmax(history["aupr"])

# 对应的最大 AUPR 和 threshold
best_aupr = history["aupr"][best_idx]
threshold = history["best_thresh"][best_idx]

print("Best epoch:", best_idx + 1)
print("Best AUPR:", best_aupr)
print("Best threshold:", threshold)

# %%
# 找到 NP 开头的列
np_cols = [col for col in pred_prob_wide_T.columns if col.startswith("NP")]

np_df = pred_prob_wide_T[np_cols]

# 每列均值
np_mean = np_df.mean()

# 每列 > threshold 的比例
np_gt_ratio = np_df.gt(threshold).mean()

# 每列 > threshold 的值的均值
np_gt_mean = np_df.where(np_df.gt(threshold)).mean()

# Pielou evenness
def pielou_evenness(x):
    x = x.dropna()
    x = x[x > 0]

    if len(x) <= 1:
        return np.nan

    p = x / x.sum()
    shannon = -(p * np.log(p)).sum()
    return shannon / np.log(len(x))

# 对每个 NP 列计算 Pielou evenness
np_pielou = np_df.where(np_df.gt(threshold)).apply(pielou_evenness, axis=0)

# 汇总
np_summary = pd.DataFrame({
    "mean": np_mean,
    "gt_threshold_ratio": np_gt_ratio,
    "gt_threshold_mean": np_gt_mean,
    #"pielou_evenness": np_pielou,
})

# 按 mean 从大到小排序
np_summary = np_summary.sort_values("mean", ascending=False)

print(np_summary)

# %%
np_summary.to_csv("pred_stat.csv")

# %%
