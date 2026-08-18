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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score
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
    def __init__(self, df, text_embed_data, pro_esm_dict, pro_esmfold_dict):
        self.df = df.reset_index(drop=True)
        self.text_embed_data = text_embed_data
        self.pro_esm_dict = pro_esm_dict
        self.pro_esmfold_dict = pro_esmfold_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Fetch the text embedding
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


    def forward(self, x_embed, pro_seq_embed, pro_str_embed):
        """
        x_embed       : [B, x_dim]
        pro_seq_embed : [B, pro_seq_dim]  (ESM2 输出等)
        pro_str_embed : [B, pro_str_dim]  (ESMFold 或结构特征)
        """

        # --------- Text branch: unchanged ---------
        x_feat = self.x_mlp(x_embed)        # [B, hidden_dim]
        
        # --------- Cross-attention: unchanged ---------
        x_tok   = x_feat.unsqueeze(1)   # [B, 1, hidden_dim]
        
        # Keep the rest unchanged and use only the text modality
        attn_out, _ = self.attn(
            query=x_tok,
            key=x_tok,
            value=x_tok,
            need_weights=False
        )
        
        fused_feature = x_tok + attn_out          # [B, 1, hidden_dim]
        fused_feature = fused_feature.squeeze(1)  # [B, hidden_dim]
        
        # --------- Classification output: unchanged ---------
        logits = self.classifier(fused_feature).squeeze(-1)   # [B]

        # Return logits + gate for later analysis of gate usage
        return logits, None

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
    
    precision = precision_score(y_true, y_pred, sample_weight=sw, zero_division=0)
    recall = recall_score(y_true, y_pred, sample_weight=sw, zero_division=0)
    
    return auroc, aupr, f1, precision, recall, acc, probs

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
    auroc_05, aupr_05, f1_05, precision_05, recall_05, acc_05, probs = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=0.5)
    # New: fixed threshold 0.25
    auroc_025, aupr_025, f1_025, precision_025, recall_025, acc_025, _ = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=0.25)

    base_dict = {
        "probs": probs,
        "y_true": y_all,
        "weights": w_all,
        "metrics@0.5":  {"AUROC": auroc_05,  "AUPRC": aupr_05,  "F1": f1_05, "Precision": precision_05, "Recall": recall_05,  "ACC": acc_05,  "thr": 0.5},
        "metrics@0.25": {"AUROC": auroc_025, "AUPRC": aupr_025, "F1": f1_025, "Precision": precision_025, "Recall": recall_025, "ACC": acc_025, "thr": 0.25},
    }

    if auto_find:
        best_thr, best_f1 = find_best_threshold(probs, y_all, sample_weight=sw)
        auroc_b, aupr_b, f1_b, precision_b, recall_b, acc_b, _ = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=best_thr)
        base_dict["metrics@best_on_test"] = {
            "AUROC": auroc_b, "AUPRC": aupr_b, "F1": f1_b, "Precision": precision_b, "Recall": recall_b, "ACC": acc_b, "thr": best_thr
        }

    # Use the given threshold
    auroc_t, aupr_t, f1_t, precision_t, recall_t, acc_t, _ = _eval_cls(logits_all, y_all, sample_weight=sw, thresh=threshold)
    base_dict["metrics@thr"] = {"AUROC": auroc_t, "AUPRC": aupr_t, "F1": f1_t, "Precision": precision_t, "Recall": recall_t, "ACC": acc_t, "thr": threshold}
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate classifier on test set")
    # Path parameter
    parser.add_argument("--save_dir", type=str, default="./output", help="目录中通常包含 checkpoint 与 history")
     # Threshold strategy
    parser.add_argument("--threshold", type=str, default="from_history",
                        choices=["0.5", "auto", "from_history"],
                        help="阈值策略：0.5 / auto(在测试集上寻优) / from_history(从history取阈值)")

    # DataLoader
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)

    # Compatible with running directly in Jupyter
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        print("Running in Jupyter/IPython environment, using default arguments.")
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # -------- Data loading --------
    print("Loading data...")
    # Read the first file
    with open("../protein_embedding/protein_embeddings_all_esm.pkl", "rb") as f:
        pro_esm_dict = pickle.load(f)
        
   
    # Check the array dimension of the first entry
    first_key = next(iter(pro_esm_dict))  # Get the first key
    array_shape = pro_esm_dict[first_key].shape
    print(f"Array shape (pro_esm_dict) for {first_key}: {array_shape}")

   
    # Read the first file
    with open("../protein_embedding/protein_embeddings_all_esmfold.pkl", "rb") as f:
        pro_esmfold_dict = pickle.load(f)
        
    
    # Check the array dimension of the first entry
    first_key = next(iter(pro_esmfold_dict))  # Get the first key
    array_shape = pro_esmfold_dict[first_key].shape
    print(f"Array shape (pro_esmfold_dict) for {first_key}: {array_shape}")

    text_embed_data = np.load("../text_embedding/text_embeddings_nonfill.npy")  # Replace with the file path
    print(f"Shape for text embedding: {text_embed_data.shape}")
    
    x_dim = text_embed_data.shape[1]
    pro_esm_dim = next(iter(pro_esm_dict.values())).shape[0]
    pro_esmfold_dim = next(iter(pro_esmfold_dict.values())).shape[0]
    
    
    # --------- Prepare test set and DataLoader (build once) ---------
    test_in_df  = pd.read_csv("data/basic_plasma_human_high_test.csv", keep_default_na=False, na_values=[''])

    test_in_loader = DataLoader(
        EmbeddingPairDataset(test_in_df,  text_embed_data, pro_esm_dict, pro_esmfold_dict),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # --------- Scan available stages N ---------
    # ==============================
    # New: treat save_dir as the root and iterate over all its subfolders
    # ==============================
    root_dir = args.save_dir  # Should now be ./output
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"--save_dir is not a directory: {root_dir}")

    # List all subfolders (first level only, ignore hidden directories)
    subdirs = sorted(
        d.name for d in os.scandir(root_dir)
        if d.is_dir() and not d.name.startswith('.')
    )
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under {root_dir}")

    print(f"\nRoot: {root_dir}")
    print(f"Found subfolders: {subdirs}")

    for sub in subdirs:
        work_dir = os.path.join(root_dir, sub)
        print(f"\n==================== Evaluating subfolder: {work_dir} ====================")

        # --------- Scan available stages N (prefer history_stage_N.pt, then saved_model_stage_N.pt and history_stage_N.npy) ---------
        stage_nums = set()
        for name in os.listdir(work_dir):
            m = re.match(r"history_stage_(\d+)\.pt$", name)
            if m: stage_nums.add(int(m.group(1)))
        if not stage_nums:
            for name in os.listdir(work_dir):
                m = re.match(r"saved_model_stage_(\d+)\.pt$", name)
                if m: stage_nums.add(int(m.group(1)))
            for name in os.listdir(work_dir):
                m = re.match(r"history_stage_(\d+)\.npy$", name)
                if m: stage_nums.add(int(m.group(1)))

        stage_nums = sorted(stage_nums)
        if not stage_nums:
            print(f"⚠️  No stage files found in {work_dir}. Skip.")
            continue
        print(f"Discovered stages in {sub}: {stage_nums}")

        sub_summary = {}
        
        # --------- Evaluate stage by stage ---------
        for load_stage in stage_nums:
            ckpt_path = os.path.join(work_dir, f"saved_model_stage_{load_stage}.pt")
            if not os.path.exists(ckpt_path):
                print(f"⚠️  Checkpoint not found for stage {load_stage}: {ckpt_path}  -> skip this stage")
                continue

            # Reload weights at each stage (the model structure is the same)
            model = CrossAttentionClassifierGated(x_dim, pro_esm_dim, pro_esmfold_dim).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"\n--------- Load from stage {load_stage} ---------")
            print(f"Loaded checkpoint: {ckpt_path}")
            
            
            history_path = os.path.join(work_dir, f"history_stage_{load_stage}.npy")
            thr_history, best_idx, best_metric = load_threshold_from_history(history_path, load_stage)

            # Internal set
            results_in = evaluate_on_loader(model, test_in_loader, thr_history)
            print("\n========== Internal Test ==========")
            
            m025 = results_in["metrics@0.25"]
            print(f"[@0.25]  AUROC={m025['AUROC']:.4f} | AUPRC={m025['AUPRC']:.4f} | F1={m025['F1']:.4f} | Precision={m025['Precision']:.4f} | Recall={m025['Recall']:.4f} | ACC={m025['ACC']:.4f}")
            
            m05 = results_in["metrics@0.5"]
            print(f"[@0.5]   AUROC={m05['AUROC']:.4f} | AUPRC={m05['AUPRC']:.4f} | F1={m05['F1']:.4f} | Precision={m05['Precision']:.4f} | Recall={m05['Recall']:.4f} |ACC={m05['ACC']:.4f}")
            
            mt = results_in["metrics@thr"]
            print(f"[@Thr ]  thr={mt['thr']:.2f} | AUROC={mt['AUROC']:.4f} | AUPRC={mt['AUPRC']:.4f} | F1={mt['F1']:.4f} | Precision={mt['Precision']:.4f} | Recall={mt['Recall']:.4f} | ACC={mt['ACC']:.4f}")

            mb = results_in["metrics@best_on_test"]
            print(f"[@Best]  thr={mb['thr']:.2f} | AUROC={mb['AUROC']:.4f} | AUPRC={mb['AUPRC']:.4f} | F1={mb['F1']:.4f} | Precision={mb['Precision']:.4f} | Recall={mb['Recall']:.4f} | ACC={mb['ACC']:.4f}")
        
            
            sub_summary[str(load_stage)] = {
                "config": {
                    "stage": load_stage,
                    "checkpoint": ckpt_path,
                    #"threshold_mode": args.threshold,
                    #"threshold_used": float(thr_to_use),
                },
                "internal": {
                    "at_0p25": _to_py(results_in["metrics@0.25"]),
                    "at_0p5": _to_py(results_in["metrics@0.5"]),
                    "at_thr": _to_py(results_in["metrics@thr"]),
                    "best_on_test": _to_py(results_in["metrics@best_on_test"]),
                },
            }
            
        # --- Save JSON ---
        sub_json_path = os.path.join(work_dir, "eval_results.json")
        with open(sub_json_path, "w", encoding="utf-8") as f:
            json.dump(_to_py(sub_summary), f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {sub_json_path}")
        
        
    # ----------------------
    # Read all eval_results.json
    # ----------------------

    root_dir = "./output"   # Consistent with training output
    summary_rows = []

    for sub in sorted(os.listdir(root_dir)):
        sub_dir = os.path.join(root_dir, sub)
        json_path = os.path.join(sub_dir, "eval_results.json")

        if not os.path.isfile(json_path):
            continue

        # Read JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data: { "1": {...}, "2": {...}, ... }
        # Iterate over each stage
        for stage, content in data.items():
            score_thr =  content["internal"]["at_thr"]

            # Compute the mean of the two metrics
            mean_score = (score_thr["AUROC"] + score_thr["F1"] ) / 2.0
        
            summary_rows.append({
                "framework": sub,      # e.g. stage_1345
                "stage": int(stage),   # Stage number of the specific checkpoint
                "AUROC": score_thr["AUROC"],
                "AUPRC": score_thr["AUPRC"],
                "F1": score_thr["F1"],
                "Precision": score_thr["Precision"],
                "Recall": score_thr["Recall"],
                "ACC": score_thr["ACC"],
                "Mean_Score (F1&AUROC)": mean_score,
                "thr": score_thr["thr"],
            })

    # ----------------------
    # Generate DataFrame
    # ----------------------
    df_summary = pd.DataFrame(summary_rows)

    # Sort by Mean_Score at external at_thr
    df_sorted = df_summary.sort_values(by="Mean_Score (F1&AUROC)", ascending=False)

    print("\n===== Sorted by at_thr Mean_Score =====")
    print(df_sorted.head(20))

    # Save as CSV
    output_csv = "eval_summary_internal.csv"
    output_dir = os.path.join(root_dir, output_csv)
    df_sorted.to_csv(output_dir, index=False, encoding="utf-8")

    print(f"\n[SAVED] {output_dir}")
    
# %%
