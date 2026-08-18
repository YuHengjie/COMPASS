# %%
import pickle
import sys
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import argparse
# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机数种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     
    torch.cuda.manual_seed(seed)                

set_seed(42)

# %%
# ================================
# Step 2: 构建 Dataset
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
                 x_dim,              # 文本 / 纳米材料特征维度
                 pro_seq_dim,        # 蛋白序列特征维度（如 ESM2: 2560）
                 pro_str_dim,        # 蛋白结构特征维度（如 ESMFold: 384）
                 hidden_dim=1024, 
                 dropout=0.3):
        super().__init__()

        # --------- 文本侧：保持不变 ---------
        self.x_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --------- 蛋白侧：seq/struct 各自投影到 hidden_dim，再融合 ---------

        # 序列 & 结构各自投影到 hidden_dim（和 x_mlp 对齐）
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

        # 维度级残差门控：输入 concat([seq_h, str_h]) ∈ R^{B×2H}
        # 输出 gate g ∈ (0,1)^{B×H}
        self.pro_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 融合后的蛋白向量，再过一层 BN + ReLU + Dropout
        # 注意：这里没有 Linear，保持“只有一层映射到 hidden_dim”的设定
        self.pro_mlp  = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --------- 注意力：保持不变 ---------
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout/2
        )
        
        # --------- 分类头：保持不变 ---------
        self.classifier  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, hidden_dim//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//16, 1)  # 二分类 logits（配合 BCEWithLogitsLoss）
        )
 
        # 初始化参数 
        self._init_weights()
 
    def _init_weights(self):
        # 原本的 x_mlp / pro_mlp / classifier 里的 Linear
        for module in [self.x_mlp, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.1)

        # 新增的 proj_seq / proj_str / pro_gate 里的 Linear
        for m in [self.proj_seq, self.proj_str] + \
                 [l for l in self.pro_gate if isinstance(l, nn.Linear)]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

        # 可选：把门控最后一层 bias 初始化为略偏向“序列”
        last_linear = [l for l in self.pro_gate if isinstance(l, nn.Linear)][-1]
        nn.init.constant_(last_linear.bias, -1.0)  # sigmoid(-1)≈0.27，初始更偏 seq

    def forward(self, x_embed, pro_seq_embed, pro_str_embed):
        """
        x_embed       : [B, x_dim]
        pro_seq_embed : [B, pro_seq_dim]  (ESM2 输出等)
        pro_str_embed : [B, pro_str_dim]  (ESMFold 或结构特征)
        """

        # --------- 文本侧：保持不变 ---------
        x_feat = self.x_mlp(x_embed)        # [B, hidden_dim]

        # --------- 蛋白序列+结构的融合 ---------
        # 1) 各自投影到 hidden_dim
        seq_h = self.proj_seq(pro_seq_embed)   # [B, hidden_dim]
        str_h = self.proj_str(pro_str_embed)   # [B, hidden_dim]

        # 2) 维度级残差门控
        #    g ∈ (0,1)^{B×hidden_dim}，每个维度一个 gate
        g = self.pro_gate(torch.cat([seq_h, str_h], dim=-1))      # [B, hidden_dim]

        # 3) 残差形式：从 seq_h 出发，用结构做纠偏
        #    g ≈ 0 → 接近 seq_h； g ≈ 1 → 接近 str_h
        pro_fused = seq_h + g * (str_h - seq_h)                   # [B, hidden_dim]

        # 4) 再过 BN + ReLU + Dropout（和 x_mlp 风格一致）
        pro_feat = self.pro_mlp(pro_fused)                        # [B, hidden_dim]
        
        # --------- 交叉注意力：保持不变 ---------
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
        
        # --------- 分类输出：保持不变 ---------
        logits = self.classifier(fused_feature).squeeze(-1)   # [B]

        # 返回 logits + gate，方便后面分析 gate 的使用情况
        return logits, g

# %%
def print_model_params_count(model):
    # 遍历模型的每一个子模块
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if num_params > 0:  # 只打印有参数的模块
            print(f"Layer: {name}, Number of parameters: {num_params}")

# %%
def finetune_head_only(model: nn.Module):
    # 1) 全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 2) 只解冻分类头
    for p in model.classifier.parameters():
        p.requires_grad = True

    # 3) 冻结部分设为 eval：锁 BN running stats + 关 dropout
    #    只让 classifier 处于 train（其内部 LayerNorm/Linear/Dropout 正常训练）
    model.eval()
    model.classifier.train()

    # 4) 打印可训练参数量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Finetune head-only] Trainable params: {trainable}/{total}")
    
    
# %%
@torch.no_grad()
def _eval_cls(logits_all, y_all, sample_weight=None, thresh=0.5):
    probs = torch.sigmoid(torch.tensor(logits_all)).numpy()
    y_true = torch.tensor(y_all).numpy().astype(int)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    auroc = roc_auc_score(y_true, probs, sample_weight=sw) if len(np.unique(y_true)) > 1 else np.nan
    aupr  = average_precision_score(y_true, probs, sample_weight=sw) if len(np.unique(y_true)) > 1 else np.nan
    y_pred = (probs >= thresh).astype(int)
    f1  = f1_score(y_true, y_pred, sample_weight=sw, zero_division=0)
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

def train_and_validate_cls(model, train_loader, val_loader, epochs=100, lr=1e-4,
                           log_interval=1, pos_weight=None, save_dir='./train_result_cls'):
    device = next(model.parameters()).device
    # BCEWithLogitsLoss；若正负样本不均衡，传入 pos_weight=torch.tensor([w]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.3*lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.3*lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, min_lr=lr*0.01, cooldown=1)

    best_aupr = -1
    history = {'train_loss':[], 'val_loss':[], 'auroc':[], 'aupr':[], 'f1':[], 'acc':[], 'best_thresh':[]}

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        # ---- train ----
        model.eval()
        model.classifier.train()
        
        total_loss = 0.0
        for text, pro_seq, pro_str, y, w in train_loader:
            text = text.to(device, non_blocking=True)
            pro_seq = pro_seq.to(device, non_blocking=True)
            pro_str = pro_str.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(text, pro_seq, pro_str)
            
            # 逐样本权重 + 正例权重（pos_weight）
            loss = F.binary_cross_entropy_with_logits(
                logits, y, weight=w, pos_weight=pos_weight, reduction='mean'
            )
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        logits_all, y_all, w_all = [], [], []
        with torch.no_grad():
            for text, pro_seq, pro_str, y, w in val_loader:
                text = text.to(device, non_blocking=True)
                pro_seq = pro_seq.to(device, non_blocking=True)
                pro_str = pro_str.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                
                logits, _ = model(text, pro_seq, pro_str)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y, weight=w, pos_weight=pos_weight, reduction='mean'
                )
                val_loss += loss.item()

                logits_all.append(logits.detach().cpu())
                y_all.append(y.detach().cpu())
                w_all.append(w.detach().cpu())

        avg_val_loss = val_loss / len(val_loader)
        logits_all = torch.cat(logits_all).numpy()
        y_all = torch.cat(y_all).numpy().astype(int)
        w_all = torch.cat(w_all).numpy().astype(float)
        
        # 加权评估
        # 先用默认0.5算指标
        auroc, aupr, f1, acc, probs = _eval_cls(logits_all, y_all, sample_weight=w_all, thresh=0.5)
        best_t, best_f1 = find_best_threshold(probs, y_all, sample_weight=w_all)

        if epoch % log_interval == 0:
            cur_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} | "
                  f"train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f} | "
                  f"AUROC {auroc:.4f} | AUPRC {aupr:.4f} | F1@0.5 {f1:.4f} | ACC@0.5 {acc:.4f} | "
                  f"best_thr {best_t:.2f} | best_F1 {best_f1:.4f} | lr {cur_lr:.6f}")

        # 记录
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['auroc'].append(auroc)
        history['aupr'].append(aupr)
        history['f1'].append(best_f1)
        history['acc'].append(acc)
        history['best_thresh'].append(best_t)

        # 以 AUPRC 作为主早停/调度指标（常用于正负不均衡）
        if aupr > best_aupr:
            best_aupr = aupr
            torch.save(model.state_dict(), os.path.join(save_dir, "saved_model_ft.pt"))
            print(f"✅ Saved best (AUPRC={best_aupr:.4f}) at epoch {epoch} (thr={best_t:.2f})")

        scheduler.step(aupr)

    # 保存曲线
    np.save(os.path.join(save_dir, 'history_ft.npy'), history, allow_pickle=True)
    print(f"Training complete. Best AUPRC: {best_aupr:.4f}")
    return history

# %%
# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning Model Training Script")
    # 可配置参数
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--save_dir', type=str, default='./output/fine_tuning', help="Directory to save model and history")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loader workers")

    # 检查是否在 Jupyter/IPython 环境中
    if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
        # 如果在 Jupyter 中，则解析一个空列表，使用默认参数
        print("Running in Jupyter/IPython environment, using default arguments.")
        args = parser.parse_args([])
    else:
        # 否则，解析命令行参数
        args = parser.parse_args()

    # --- 4.1 数据加载 ---
    print("Loading data...")
    # 读取第一个文件
    with open("../data/protein_embeddings.pkl", "rb") as f:
        pro_esm_dict = pickle.load(f)
        
   
    # 检查第一个条目的数组维度
    first_key = next(iter(pro_esm_dict))  # 获取第一个键
    array_shape = pro_esm_dict[first_key].shape
    print(f"Array shape (pro_esm_dict) for {first_key}: {array_shape}")

   
    # 读取第一个文件
    with open("../data/esmfold_protein_embeddings.pkl", "rb") as f:
        pro_esmfold_dict = pickle.load(f)
        
    
    # 检查第一个条目的数组维度
    first_key = next(iter(pro_esmfold_dict))  # 获取第一个键
    array_shape = pro_esmfold_dict[first_key].shape
    print(f"Array shape (pro_esmfold_dict) for {first_key}: {array_shape}")

    
    text_embed_data = np.load("../data/text_embeddings.npy")  # 替换为文件路径
    print(f"Shape for text embedding: {text_embed_data.shape}")


    train_df = pd.read_csv(f"data/train.csv",keep_default_na=False, na_values=[''])
    val_df = pd.read_csv(f"data/val.csv",keep_default_na=False, na_values=[''])
    
    
    # --- 4.2 数据集和 DataLoader ---
    train_dataset = EmbeddingPairDataset(train_df, text_embed_data, pro_esm_dict, pro_esmfold_dict)
    val_dataset = EmbeddingPairDataset(val_df, text_embed_data, pro_esm_dict, pro_esmfold_dict)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # --- 4.3 模型初始化 ---
    x_dim = text_embed_data.shape[1]
    pro_esm_dim = next(iter(pro_esm_dict.values())).shape[0]
    pro_esmfold_dim = next(iter(pro_esmfold_dict.values())).shape[0]

    model = CrossAttentionClassifierGated(x_dim, pro_esm_dim, pro_esmfold_dim).to(device)
    print_model_params_count(model)

    # ====== 加载预训练权重（用已有的stage-1最优模型）======
    pretrained_ckpt = os.path.join("../../train_basic_10/output/stage_12345/saved_model_stage_5.pt")
    if os.path.exists(pretrained_ckpt):
        print(f"Loading pretrained weights from: {pretrained_ckpt}")
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=device), strict=True)
    else:
        print(f"⚠️ pretrained checkpoint not found: {pretrained_ckpt} (will train from scratch)")

    # ====== 微调：只训分类头 ======
    finetune_head_only(model)

    # --- 4.4 计算 pos_weight ---
    class_counts = train_df['Protein corona composition'].value_counts()
    num_neg = class_counts.get(0, 0)
    num_pos = class_counts.get(1, 0)
    
    pos_weight_val = 2 * num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}, pos_weight: {pos_weight_val:.4f}")
    
    # --- 4.5 启动训练 ---
    print('\n***********Start Training Stage-1***********')
    history = train_and_validate_cls(
        model, train_loader, val_loader,
        epochs=args.epochs, 
        lr=args.lr,
        pos_weight=pos_weight,
        save_dir=args.save_dir
    )
    
# %%