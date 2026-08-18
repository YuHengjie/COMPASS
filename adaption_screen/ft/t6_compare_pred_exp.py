# %%
import pandas as pd
import numpy as np
from IPython.display import display
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# %%
df_raw = pd.read_csv("exper_stat.csv")
df_raw

# %%
df_pred = pd.read_csv("pred_stat.csv")
df_pred

# %%
df_pred.rename(columns={"Unnamed: 0": "sample"}, inplace=True)
df_pred

# %%
df_raw = df_raw.sort_values(by='coverage', ascending=False).reset_index(drop=True)
df_raw

# %%
df_pred = df_pred.sort_values(by='gt_threshold_ratio', ascending=False).reset_index(drop=True)
df_pred

# %%
# First generate sorted numbering for the sample column
# Current order
sample_raw = df_raw['sample'].tolist()
sample_pred = df_pred['sample'].tolist()
print("df_raw 中的 sample 顺序:")
print(sample_raw)
print("\ndf_pred 中的 sample 顺序:")
print(sample_pred)

# %%
# Generate rank mapping (starting from 1)
raw_rank = {s: i+1 for i, s in enumerate(df_raw['sample'])}
pred_rank = {s: i+1 for i, s in enumerate(df_pred['sample'])}

# Create the comparison table
rank_df = pd.DataFrame({
    'sample': sorted(raw_rank.keys())
})

rank_df['raw_rank'] = rank_df['sample'].map(raw_rank)
rank_df['pred_rank'] = rank_df['sample'].map(pred_rank)

rank_df

# %%
corr, p_value = spearmanr(rank_df['raw_rank'], rank_df['pred_rank'])
print(f"Spearman 相关系数: {corr:.3f}, p-value: {p_value:.3g}")

# %%
n = len(rank_df)

rank_df['raw_group'] = pd.qcut(rank_df['raw_rank'], q=3, labels=['High','Medium','Low'])
rank_df['pred_group'] = pd.qcut(rank_df['pred_rank'], q=3, labels=['High','Medium','Low'])
rank_df

# %%
agreement_table = pd.crosstab(
    rank_df['raw_group'],
    rank_df['pred_group'],
    rownames=['raw_group'],
    colnames=['pred_group']
)

agreement_table

# %%
agreement_rate = np.trace(agreement_table.values) / agreement_table.values.sum()

print(f"Agreement rate: {agreement_rate:.2%}")

# %%
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(
    rank_df['raw_group'],
    rank_df['pred_group']
)

print(f"Cohen's kappa: {kappa:.3f}")

# %%
same_group = rank_df['raw_group'] == rank_df['pred_group']

print("Same group samples:", same_group.sum())
print("Total samples:", len(rank_df))
print(f"Agreement rate: {same_group.mean():.2%}")

# %%
high_low_samples = rank_df[
    ((rank_df['raw_group'] == 'High') & (rank_df['pred_group'] == 'Low')) |
    ((rank_df['raw_group'] == 'Low') & (rank_df['pred_group'] == 'High'))
]

high_low_samples[['sample', 'raw_group', 'pred_group']]

# %%
# Merge coverage values from df_raw
rank_df = rank_df.merge(
    df_raw[['sample', 'coverage']],
    on='sample',
    how='left'
)

# Merge gt_threshold_ratio values from df_pred
rank_df = rank_df.merge(
    df_pred[['sample', 'gt_threshold_ratio']],
    on='sample',
    how='left'
)

rank_df


# %%

# Pearson correlation
r, p = pearsonr(
    rank_df['coverage'],
    rank_df['gt_threshold_ratio']
)

print(f"Pearson r = {r:.3f}")
print(f"p-value = {p:.3g}")


# %%
K = 10

true_top_k = set(
    rank_df.sort_values('raw_rank').head(K)['sample']
)

pred_top_k = set(
    rank_df.sort_values('pred_rank').head(K)['sample']
)

overlap = true_top_k & pred_top_k

precision_at_k = len(overlap) / K
recall_at_k = len(overlap) / K

print(f"Top-{K} overlap: {len(overlap)}/{K}")
print(f"Precision@{K}: {precision_at_k:.2%}")
print(f"Recall@{K}: {recall_at_k:.2%}")
print("Hit samples:", overlap)

# %%
# Plot heatmap
fig, ax =  plt.subplots(figsize=(4, 3))
hm = sns.heatmap(
    agreement_table,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    ax=ax
)

# axis labels
ax.set_xlabel(
    "Predicted rank",
    fontproperties=prop,
    fontsize=10
)

ax.set_ylabel(
    "Experimental rank",
    fontproperties=prop,
    fontsize=10
)

# x tick labels
for label in ax.get_xticklabels():
    label.set_fontproperties(prop)

# y tick labels
for label in ax.get_yticklabels():
    label.set_fontproperties(prop)

# heatmap numbers
for text in ax.texts:
    text.set_fontproperties(prop)

# colorbar font
cbar = hm.collections[0].colorbar
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(prop)

plt.tight_layout()
plt.savefig(
    "output/agreement_table_heatmap.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# %%
# %% Option 1: Agreement Heatmap (fix fonts and style)
fig, ax = plt.subplots(figsize=(4.5, 3.5))
hm = sns.heatmap(
    agreement_table,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    ax=ax,
    annot_kws={"fontproperties": prop, "fontsize": 10}
)

ax.set_xlabel("Predicted rank", fontproperties=prop, fontsize=11)
ax.set_ylabel("Experimental rank", fontproperties=prop, fontsize=11)

# Set tick fonts consistently
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(prop)

cbar = hm.collections[0].colorbar
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(prop)

plt.tight_layout()
plt.savefig("output/agreement_table_heatmap.png", dpi=600, bbox_inches="tight")
plt.show()


# %%
from matplotlib.lines import Line2D
df_plot = rank_df.copy()

# Sort by predicted rank
df_plot = df_plot.sort_values(
    by='pred_rank'
).reset_index(drop=True)


# convert rank to height
max_rank = max(
    df_plot['raw_rank'].max(),
    df_plot['pred_rank'].max()
)

# smaller rank corresponds to greater height
df_plot['raw_height'] = (
    max_rank - df_plot['raw_rank'] + 1
)

df_plot['pred_height'] = (
    max_rank - df_plot['pred_rank'] + 1
)


x_pos = np.arange(len(df_plot))


# =========================
# Group colors by rank
# =========================

rank_colors = {
    "top": "#fb6a4b",       # rank 1-12
    "middle": "#55A868",    # rank 13-24
    "bottom": "#4b89bf"     # rank 25-36
}


def rank_to_color(rank):

    if rank <= 12:
        return rank_colors["top"]

    elif rank <= 24:
        return rank_colors["middle"]

    else:
        return rank_colors["bottom"]



# color by raw rank
raw_colors = [
    rank_to_color(r)
    for r in df_plot['raw_rank']
]


# color by predicted rank
pred_colors = [
    rank_to_color(r)
    for r in df_plot['pred_rank']
]



# =========================
# Plot
# =========================

fig, ax = plt.subplots(
    figsize=(15, 4.5)
)


# --------------------------------
# Predicted rank
# --------------------------------

# stem
ax.vlines(
    x_pos - 0.18,
    ymin=0,
    ymax=df_plot['pred_height'],
    colors=pred_colors,
    linestyles='dashed',
    linewidth=1.8,
    alpha=0.9,
    zorder=1
)


# point
# Outer circle: colored edge
ax.scatter(
    x_pos - 0.18,
    df_plot['pred_height'],
    s=120,
    facecolors='none',
    edgecolors=pred_colors,
    linewidths=2.0,
    marker='D',
    zorder=3
)

# Inner circle: white isolation background
ax.scatter(
    x_pos - 0.18,
    df_plot['pred_height'],
    s=90,
    facecolors='white',
    edgecolors='none',
    marker='D',
    zorder=4
)

# Inner point: solid colored
ax.scatter(
    x_pos - 0.18,
    df_plot['pred_height'],
    s=45,
    color=pred_colors,
    marker='D',
    zorder=5
)


# --------------------------------
# Experimental rank
# --------------------------------

# stem
ax.vlines(
    x_pos + 0.18,
    ymin=0,
    ymax=df_plot['raw_height'],
    colors=raw_colors,
    linestyles='solid',
    linewidth=1.5,
    alpha=0.85,
    zorder=1
)


# point
# Outer circle: colored edge
ax.scatter(
    x_pos + 0.18,
    df_plot['raw_height'],
    s=150,
    facecolors='none',
    edgecolors=raw_colors,
    linewidths=2.0,
    marker='o',
    zorder=3
)

# Inner circle: white isolation background
ax.scatter(
    x_pos + 0.18,
    df_plot['raw_height'],
    s=95,
    facecolors='white',
    edgecolors='none',
    marker='o',
    zorder=4
)

# Inner point: solid colored
ax.scatter(
    x_pos + 0.18,
    df_plot['raw_height'],
    s=45,
    color=raw_colors,
    marker='o',
    zorder=5
)

# =========================
# Axes
# =========================


ax.set_xticks(x_pos)

ax.set_xticklabels(
    df_plot['sample'],
    rotation=45,
    ha='right',
    fontproperties=prop,
    fontsize=12
)



# =========================
# y-axis shows only 0, 12, 24, 36
# =========================

rank_ticks = [0, 12, 24, 36]

# Convert to bar-height coordinates
ax.set_yticks(
    max_rank - np.array(rank_ticks) + 1
)

ax.set_yticklabels(
    rank_ticks,
    fontproperties=prop,
    fontsize=12
)



ax.set_ylabel(
    "Rank",
    fontproperties=prop,
    fontsize=12
)


#ax.set_xlabel("Sample", fontproperties=prop, fontsize=12)



# =========================
# Styling
# =========================

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')


ax.tick_params(
    axis='x',
    length=4
)


ax.tick_params(
    axis='y',
    length=4
)


ax.set_ylim(
    0,
    max_rank + 4
)


ax.set_xlim(
    -0.8,
    len(df_plot)-0.2
)



# =========================
# Legend
# =========================


raw_handle = Line2D(
    [0],
    [0],
    linestyle='--',
    color='black',
    marker='o',
    markerfacecolor='white',
    markeredgecolor='black',
    markersize=8,
    linewidth=1.5
)


pred_handle = Line2D(
    [0],
    [0],
    linestyle='-',
    color='black',
    marker='D',
    markerfacecolor='black',
    markeredgecolor='black',
    markersize=7,
    linewidth=1.5
)



rank1_handle = Line2D(
    [0],
    [0],
    marker='o',
    linestyle='None',
    markerfacecolor=rank_colors["top"],
    markeredgecolor=rank_colors["top"],
    markersize=8
)


rank2_handle = Line2D(
    [0],
    [0],
    marker='o',
    linestyle='None',
    markerfacecolor=rank_colors["middle"],
    markeredgecolor=rank_colors["middle"],
    markersize=8
)


rank3_handle = Line2D(
    [0],
    [0],
    marker='o',
    linestyle='None',
    markerfacecolor=rank_colors["bottom"],
    markeredgecolor=rank_colors["bottom"],
    markersize=8
)



leg = ax.legend(
    [
        pred_handle,
        raw_handle,
        rank1_handle,
        rank2_handle,
        rank3_handle
    ],
    [
        "Predicted rank",
        "Experimental rank",
        "High-performing",
        "Medium-performing",
        "Low-performing"
    ],
    frameon=False,
    loc='upper right',
    prop=prop,
    
)

for text in leg.get_texts():
    text.set_fontsize(12)

plt.tight_layout()


plt.savefig(
    "output/rank_lollipop_comparison.png",
    dpi=600,
    bbox_inches='tight'
)


plt.show()

# %% Option 2: Rank-Order Scatter (fully fix FixedFormatter warning)
fig, ax = plt.subplots(figsize=(5, 5))

# 1. Draw scatter points
ax.scatter(
    rank_df['raw_rank'], 
    rank_df['pred_rank'], 
    s=40, alpha=0.7, 
    color='#4C72B0', edgecolors='white', linewidth=0.5
)

# 2. Diagonal
max_rank = len(rank_df) + 1
ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5, lw=1.5)

# 3. Annotate Spearman
corr, p_value = spearmanr(rank_df['raw_rank'], rank_df['pred_rank'])
textstr = f'Spearman $\\rho = {corr:.3f}$\n$p = {p_value:.2g}$'
props_box = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='none')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props_box, fontproperties=prop)

# 4. Safe axis and font settings
ax.set_xlabel("Experimental Rank", fontproperties=prop, fontsize=12)
ax.set_ylabel("Predicted Rank", fontproperties=prop, fontsize=12)
ax.set_xlim(0, max_rank)
ax.set_ylim(0, max_rank)

# Avoid the set_xticklabels warning
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(10)

ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("output/nature_rank_scatter.png", dpi=600, bbox_inches="tight")
plt.show()



# %% Option 3: Bland-Altman Plot (safe plotting with normalization)
fig, ax = plt.subplots(figsize=(6, 4.5))

# If the numeric scales differ, uncomment the two lines below for Z-score normalization
# exp_val = (rank_df['coverage'] - rank_df['coverage'].mean()) / rank_df['coverage'].std()
# pred_val = (rank_df['gt_threshold_mean'] - rank_df['gt_threshold_mean'].mean()) / rank_df['gt_threshold_mean'].std()

exp_val = rank_df['coverage']
pred_val = rank_df['gt_threshold_ratio']

mean_vals = (exp_val + pred_val) / 2
diff_vals = exp_val - pred_val

md = np.mean(diff_vals)
sd = np.std(diff_vals, ddof=1)

ax.scatter(mean_vals, diff_vals, s=30, alpha=0.6, color='#DD8452', edgecolors='white', linewidth=0.5)
ax.axhline(md, color='black', linestyle='-', lw=1.5, label=f'Bias = {md:.3f}')
ax.axhline(md + 1.96*sd, color='gray', linestyle='--', lw=1, label=f'+1.96 SD = {md+1.96*sd:.3f}')
ax.axhline(md - 1.96*sd, color='gray', linestyle='--', lw=1, label=f'-1.96 SD = {md-1.96*sd:.3f}')

ax.set_xlabel("Mean of Experimental & Predicted", fontproperties=prop, fontsize=12)
ax.set_ylabel("Difference (Exp - Pred)", fontproperties=prop, fontsize=12)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(10)

ax.legend(prop=prop, frameon=False, fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig("output/nature_bland_altman.png", dpi=600, bbox_inches="tight")
plt.show()


# %% Option 4: Top-K Cumulative Hit Rate (fix alignment and formatting)
fig, ax = plt.subplots(figsize=(6, 4.5))

K_max = len(rank_df)
hits = []
for k in range(1, K_max + 1):
    pred_top_k = set(rank_df.sort_values('pred_rank').head(k)['sample'])
    true_top_k = set(rank_df.sort_values('raw_rank').head(k)['sample'])
    hits.append(len(pred_top_k & true_top_k) / k)

# 1. Draw the model step plot (where='post' keeps K=1 at x=1)
ax.step(range(1, K_max + 1), hits, where='post', color='#4C72B0', lw=2, label='Model')

# 2. Random baseline
random_baseline = [k / K_max for k in range(1, K_max + 1)]
ax.plot(range(1, K_max + 1), random_baseline, 'k--', alpha=0.4, lw=1, label='Random Baseline')

ax.set_xlabel("Top-K schemes", fontproperties=prop, fontsize=12)
ax.set_ylabel("Cumulative hit rate (Precision@K)", fontproperties=prop, fontsize=12)
ax.set_xlim(1, K_max)
ax.set_ylim(0, 1.05)

# Set tick fonts safely; avoid manual set_yticklabels formatting to prevent misalignment
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(prop)
    label.set_fontsize(10)

ax.legend(prop=prop, frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("output/nature_topk_hitrate.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
