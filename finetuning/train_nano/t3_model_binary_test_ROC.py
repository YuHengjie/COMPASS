# %%
import pickle
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import boxcox
from datetime import datetime 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, accuracy_score
import argparse
import json
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     
    torch.cuda.manual_seed(seed)                

set_seed(42)

# %%
# ================================
# Step 2: Build Dataset
# ================================
class EmbeddingPairDataset(Dataset):
    def __init__(self, df,  text_embed_data, pro_esm_dict, pro_esmfold_dict):
        self.df = df.reset_index(drop=True)
        self.text_embed_data = text_embed_data
        self.pro_esm_dict = pro_esm_dict
        self.pro_esmfold_dict = pro_esmfold_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text_embed = self.text_embed_data[int(row["x_index"])]
        
        # Fetch the protein embedding
        pro_esm_embed = self.pro_esm_dict[row["Accession"]]
        pro_esmfold_embed = self.pro_esmfold_dict[row["Accession"]]
        
        # Affinity category
        rpa = row["Protein corona composition"]
        w = float(row.get("Overall data quality", 1.0))
        
        return torch.tensor(text_embed, dtype=torch.float32), \
               torch.tensor(pro_esm_embed, dtype=torch.float32), \
               torch.tensor(pro_esmfold_embed, dtype=torch.float32), \
               torch.tensor(rpa, dtype=torch.float32), \
               torch.tensor(w, dtype=torch.float32)

# %%
class CrossAttentionClassifierGated(nn.Module):
    def __init__(self, 
                 x_dim,              # Text / nanomaterial feature dimension
                 pro_seq_dim,        # Protein sequence feature dimension (e.g. ESM2: 2560)
                 pro_str_dim,        # Protein structure feature dimension (e.g. ESMFold: 384)
                 hidden_dim=1024, 
                 dropout=0.3):
        super().__init__()

        # --------- Text branch: unchanged ---------
        self.x_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --------- Protein branch: project seq/struct to hidden_dim, then fuse ---------

        # Project sequence and structure to hidden_dim (aligned with x_mlp)
        self.proj_seq = nn.Sequential(
            nn.Linear(pro_seq_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.proj_str = nn.Sequential(
            nn.Linear(pro_str_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Dimension-wise residual gating: input concat([seq_h, str_h]) in R^{B×2H}
        # Output gate g in (0,1)^{B×H}
        self.pro_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Apply BN + ReLU + Dropout again to the fused protein vector
        # There is no Linear layer here, keeping only one mapping to hidden_dim
        self.pro_mlp  = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --------- Attention: unchanged ---------
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout/2
        )
        
        # --------- Classification head: unchanged ---------
        self.classifier  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, hidden_dim//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//16, 1)  # Binary logits (with BCEWithLogitsLoss)
        )
 
        # Initialize parameters
        self._init_weights()
 
    def _init_weights(self):
        # Linear layers originally in x_mlp / pro_mlp / classifier
        for module in [self.x_mlp, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.1)

        # New Linear layers in proj_seq / proj_str / pro_gate
        for m in [self.proj_seq, self.proj_str] + \
                 [l for l in self.pro_gate if isinstance(l, nn.Linear)]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

        # Optional: initialize the final gate bias slightly toward sequence
        last_linear = [l for l in self.pro_gate if isinstance(l, nn.Linear)][-1]
        nn.init.constant_(last_linear.bias, -1.0)  # sigmoid(-1)≈0.27, initially biased toward seq

    def forward(self, x_embed, pro_seq_embed, pro_str_embed):
        """
        x_embed       : [B, x_dim]
        pro_seq_embed : [B, pro_seq_dim]  (ESM2 输出等)
        pro_str_embed : [B, pro_str_dim]  (ESMFold 或结构特征)
        """

        # --------- Text branch: unchanged ---------
        x_feat = self.x_mlp(x_embed)        # [B, hidden_dim]

        # --------- Protein sequence + structure fusion ---------
        # 1) Project each to hidden_dim
        seq_h = self.proj_seq(pro_seq_embed)   # [B, hidden_dim]
        str_h = self.proj_str(pro_str_embed)   # [B, hidden_dim]

        # 2) Dimension-wise residual gating
        # Each dimension has one gate: g in (0,1)^{B×hidden_dim}
        g = self.pro_gate(torch.cat([seq_h, str_h], dim=-1))      # [B, hidden_dim]

        # 3) Residual form: start from seq_h and correct with structure
        # g ≈ 0 -> close to seq_h; g ≈ 1 -> close to str_h
        pro_fused = seq_h + g * (str_h - seq_h)                   # [B, hidden_dim]

        # 4) Apply BN + ReLU + Dropout again (consistent with x_mlp style)
        pro_feat = self.pro_mlp(pro_fused)                        # [B, hidden_dim]
        
        # --------- Cross-attention: unchanged ---------
        x_tok   = x_feat.unsqueeze(1)   # [B, 1, hidden_dim]
        pro_tok = pro_feat.unsqueeze(1) # [B, 1, hidden_dim]
        
        attn_out, _ = self.attn(
            query=x_tok,
            key=pro_tok,
            value=pro_tok,
            need_weights=False
        )
        
        fused_feature = x_tok + attn_out          # [B, 1, hidden_dim]
        fused_feature = fused_feature.squeeze(1)  # [B, hidden_dim]
        
        # --------- Classification output: unchanged ---------
        logits = self.classifier(fused_feature).squeeze(-1)   # [B]

        # Return logits + gate for later analysis of gate usage
        return logits, g

# %%
def print_model_params_count(model):
    # Iterate over every submodule of the model
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if num_params > 0:  # Print only modules with parameters
            print(f"Layer: {name}, Number of parameters: {num_params}")

# %%
# -------------------------
# 4) Evaluation utilities
# -------------------------
@torch.no_grad()
def _eval_cls(logits_all, y_all, sample_weight=None, thresh=0.5):
    probs = torch.sigmoid(torch.tensor(logits_all)).numpy()
    y_true = torch.tensor(y_all).numpy().astype(int)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    # Protect against NaN when only one class is present
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, probs, sample_weight=sw)
        aupr = average_precision_score(y_true, probs, sample_weight=sw)
    else:
        auroc, aupr = np.nan, np.nan

    y_pred = (probs >= thresh).astype(int)
    f1 = f1_score(y_true, y_pred, sample_weight=sw, zero_division=0,)
    acc = accuracy_score(y_true, y_pred, sample_weight=sw)
    return auroc, aupr, f1, acc, probs

def find_best_threshold(probs, y_true, sample_weight=None):
    grid = np.linspace(0.05, 0.95, 19)
    y_true = np.asarray(y_true).astype(int)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    best_t, best_f1 = 0.5, -1
    for t in grid:
        f1 = f1_score(y_true, (probs >= t).astype(int), sample_weight=sw, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

@torch.no_grad()
def evaluate_on_loader(model, loader, threshold=0.5, auto_find=True, sample_weighted=True):
    model.eval()
    logits_all, y_all, w_all = [], [], []
    for text, pro_seq, pro_str, y, w in loader:
        text = text.to(device, non_blocking=True)
        pro_seq = pro_seq.to(device, non_blocking=True)
        pro_str = pro_str.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        logits, _ = model(text, pro_seq, pro_str)
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
        w_all.append(w.detach().cpu())

    logits_all = torch.cat(logits_all).numpy()
    y_all = torch.cat(y_all).numpy().astype(int)
    w_all = torch.cat(w_all).numpy().astype(float)

    sw = w_all if sample_weighted else None

    # Fixed threshold 0.5
    auroc_05, aupr_05, f1_05, acc_05, probs = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=0.5)
    # New: fixed threshold 0.25
    auroc_025, aupr_025, f1_025, acc_025, _ = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=0.25)

    base_dict = {
        "probs": probs,
        "y_true": y_all,
        "weights": w_all,
        "metrics@0.5":  {"AUROC": auroc_05,  "AUPRC": aupr_05,  "F1": f1_05,  "ACC": acc_05,  "thr": 0.5},
        "metrics@0.25": {"AUROC": auroc_025, "AUPRC": aupr_025, "F1": f1_025, "ACC": acc_025, "thr": 0.25},
    }

    if auto_find:
        best_thr, best_f1 = find_best_threshold(probs, y_all, sample_weight=sw)
        auroc_b, aupr_b, f1_b, acc_b, _ = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=best_thr)
        base_dict["metrics@best_on_test"] = {
            "AUROC": auroc_b, "AUPRC": aupr_b, "F1": f1_b, "ACC": acc_b, "thr": best_thr
        }

    # Use the given threshold
    auroc_t, aupr_t, f1_t, acc_t, _ = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=threshold)
    base_dict["metrics@thr"] = {"AUROC": auroc_t, "AUPRC": aupr_t, "F1": f1_t, "ACC": acc_t, "thr": threshold}
    return base_dict

def load_threshold_from_history(history_path, load_stage: int):
    """
    从训练保存的 history_*.npy 中，根据 load_stage 选择评估指标与阈值字段，
    找到该指标最优 epoch 对应的阈值，并返回 (thr, best_idx, best_metric)。

    规则：
      - load_stage ∈ {1, 5}: 指标='aupr', 阈值='best_thresh'
      - load_stage ∈ {2, 3, 4}: 指标='avg_aupr', 阈值='val_out_best_t'
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    hist = np.load(history_path, allow_pickle=True).item()
    if not isinstance(hist, dict):
        raise ValueError(f"History file is not a dict-like object: {history_path}")

    # Select fields according to stage
    if load_stage in (1, 5):
        metric_key = "aupr"
        thr_key = "best_thresh"
    elif load_stage in (2, 3, 4):
        metric_key = "avg_aupr"
        thr_key = "val_out_best_t"
    else:
        raise ValueError(f"Unsupported load_stage={load_stage}. Expected one of {{1,2,3,4,5}}.")

    # Read metric and threshold lists
    metrics = np.array(hist.get(metric_key, []), dtype=float)
    thr_list = np.array(hist.get(thr_key, []), dtype=float)

    # Robustness check
    if metrics.size == 0:
        raise ValueError(f"'{metric_key}' is empty or missing in {history_path}. keys={list(hist.keys())}")
    if thr_list.size == 0:
        raise ValueError(f"'{thr_key}' is empty or missing in {history_path}. keys={list(hist.keys())}")
    if len(metrics) != len(thr_list):
        raise ValueError(
            f"Length mismatch in {history_path}: len({metric_key})={len(metrics)} vs len({thr_key})={len(thr_list)}"
        )
    if np.all(np.isnan(metrics)):
        raise ValueError(f"All values in '{metric_key}' are NaN in {history_path}.")

    # Get the index of the best metric
    metrics_safe = np.where(np.isnan(metrics), -np.inf, metrics)
    best_idx = int(np.argmax(metrics_safe))

    thr = float(thr_list[best_idx])
    best_metric = float(metrics[best_idx])
    return thr, best_idx, best_metric

# =============== Safe JSON conversion ===============
def _to_py(obj):
    """把 numpy / torch 类型安全转成原生 Python 类型，便于 json.dump。"""
    try:
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# %%
@torch.no_grad()
def collect_probs_on_loader(model, loader, device, sample_weighted=True):
    """
    返回：
      y_true: (N,)
      y_prob: (N,)  # 正类概率
      weights: (N,) # sample weights
      logits: (N,)  # 原始 logits（可选）
    """
    model.eval()
    logits_all, y_all, w_all = [], [], []

    for text, pro_seq, pro_str, y, w in loader:
        text = text.to(device, non_blocking=True)
        pro_seq = pro_seq.to(device, non_blocking=True)
        pro_str = pro_str.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        logits, _ = model(text, pro_seq, pro_str)  # Same as evaluate_on_loader
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
        w_all.append(w.detach().cpu())

    logits_all = torch.cat(logits_all).numpy()
    y_true = torch.cat(y_all).numpy().astype(int)
    weights = torch.cat(w_all).numpy().astype(float)

    # _eval_cls should apply sigmoid(logits) to obtain probabilities
    # Explicitly ensure y_prob is the positive-class probability
    # Handle logits shape (N,1)/(N,) as (N,)
    logits_1d = logits_all.reshape(-1)
    y_prob = 1.0 / (1.0 + np.exp(-logits_1d))  # sigmoid

    if not sample_weighted:
        weights = None

    return y_true, y_prob, weights, logits_1d

# %%
# -------- Data loading --------
print("Loading data...")
# Read the first file
with open("../../protein_embedding/protein_embeddings_all_esm.pkl", "rb") as f:
    pro_esm_dict = pickle.load(f)
    

# Check the array dimension of the first entry
first_key = next(iter(pro_esm_dict))  # Get the first key
array_shape = pro_esm_dict[first_key].shape
print(f"Array shape (pro_esm_dict) for {first_key}: {array_shape}")


# Read the first file
with open("../../protein_embedding/protein_embeddings_all_esmfold.pkl", "rb") as f:
    pro_esmfold_dict = pickle.load(f)
    

# Check the array dimension of the first entry
first_key = next(iter(pro_esmfold_dict))  # Get the first key
array_shape = pro_esmfold_dict[first_key].shape
print(f"Array shape (pro_esmfold_dict) for {first_key}: {array_shape}")

text_embed_data = np.load("../../text_embedding/text_embeddings_nonfill.npy")  # Replace with the file path
print(f"Shape for text embedding: {text_embed_data.shape}")

x_dim = text_embed_data.shape[1]
pro_esm_dim = next(iter(pro_esm_dict.values())).shape[0]
pro_esmfold_dim = next(iter(pro_esmfold_dict.values())).shape[0]


# --------- Prepare test set and DataLoader (build once) ---------
test_out_df = pd.read_csv("data/nano_external_plasma_human_test.csv", keep_default_na=False, na_values=[''])

test_out_loader = DataLoader(
    EmbeddingPairDataset(test_out_df,  text_embed_data, pro_esm_dict, pro_esmfold_dict),
    batch_size=4096, shuffle=False, num_workers=8
)

# --------- Evaluate stage by stage ---------
load_stage = 5
work_dir = 'output/stage_12345'
ckpt_path = os.path.join(work_dir, f"saved_model_stage_{load_stage}.pt")


# Reload weights at each stage (the model structure is the same)
model = CrossAttentionClassifierGated(x_dim, pro_esm_dim, pro_esmfold_dim).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
print(f"\n--------- Load from stage {load_stage} ---------")
print(f"Loaded checkpoint: {ckpt_path}")

# %%
# 1) Collect scores
y_true, y_prob, weights, logits_1d = collect_probs_on_loader(
    model, test_out_loader, device, sample_weighted=True
)

# 2) Compute weighted ROC
fpr, tpr, roc_thr = roc_curve(y_true, y_prob, sample_weight=weights)
roc_auc = roc_auc_score(y_true, y_prob, sample_weight=weights)

# 3) Save plotting data (npy: fast and lossless)
roc_data = {
    "fpr": fpr,
    "tpr": tpr,
    "thresholds": roc_thr,
    "auc": float(roc_auc),
    "y_true": y_true,
    "y_prob": y_prob,
    "weights": weights if weights is not None else None,
}
roc_npy_path = os.path.join(work_dir, f"roc_external_stage_{load_stage}.npy")
np.save(roc_npy_path, roc_data)
print(f"[SAVE] {roc_npy_path}")


# ======================
# PR curve (weighted)
# ======================
precision, recall, pr_thr = precision_recall_curve(y_true, y_prob, sample_weight=weights)
ap = average_precision_score(y_true, y_prob, sample_weight=weights)  # AUPRC / AP

pr_data = {
    "precision": precision,
    "recall": recall,
    "thresholds": pr_thr,     # Length is usually one less than precision/recall
    "ap": float(ap),
    "y_true": y_true,
    "y_prob": y_prob,
    "weights": weights if weights is not None else None,
}

pr_npy_path = os.path.join(work_dir, f"pr_external_stage_{load_stage}.npy")
np.save(pr_npy_path, pr_data)
print(f"[SAVE] {pr_npy_path}")

# %%
# 4) Plot ROC and save the figure
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC (Internal) stage {load_stage}")
ax.legend(loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

roc_png_path = os.path.join(work_dir, f"roc_external_stage_{load_stage}.png")
#plt.savefig(roc_png_path, dpi=600, bbox_inches="tight")
#plt.close(fig)
print(f"[SAVE] {roc_png_path}")

# %%
# ======================
# Plot PR and save the figure
# ======================
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot(recall, precision, label=f"AP={ap:.4f}")

# baseline: positive-class proportion (precision level of a random classifier)
pos_rate = y_true.mean()
ax.hlines(pos_rate, 0, 1, linestyles="--", linewidth=1, label=f"Baseline={pos_rate:.3f}")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"PR (Internal) stage {load_stage}")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.legend(loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

pr_png_path = os.path.join(work_dir, f"pr_external_stage_{load_stage}.png")
# plt.savefig(pr_png_path, dpi=600, bbox_inches="tight")
# plt.close(fig)
print(f"[SAVE] {pr_png_path}")

# %%
