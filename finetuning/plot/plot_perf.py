# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
font_path = "/home/yuhengjie/.fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

# %%
train_nano_scratch_np = '../train_nano/output/from_scratch/eval_summary_external.csv'
train_nano_ft_np = '../train_nano/output/ft/eval_summary_external.csv'

df_nano_scratch = pd.read_csv(train_nano_scratch_np)
df_nano_scratch

# %%
df_nano_ft = pd.read_csv(train_nano_ft_np)
df_nano_ft

# %%
# Ensure sorting by framework (10,20,...,100)
df_nano_scratch   = df_nano_scratch.sort_values('framework')
df_nano_ft = df_nano_ft.sort_values('framework')
df_nano_scratch

# %%
mean_gain = (
    df_nano_ft['AUROC'].values
    - df_nano_scratch['AUROC'].values
).mean()

mean_gain


# %%
fig, ax = plt.subplots(figsize=(6.5, 3))

# First line: nano scratch
ax.plot(
    df_nano_scratch['framework'],
    df_nano_scratch['AUROC'],
    marker='o',
    linewidth=2,
    markersize=6,
    color="#75A0AF",
    label='From scratch'
)

# Second line: nano fine-tune
ax.plot(
    df_nano_ft['framework'],
    df_nano_ft['AUROC'],
    marker='s',
    linewidth=2,
    markersize=6,
    color='#DF8389',
    label='Finetuning'
)

ax.margins(y=0)  # Remove automatic padding in the y direction
ax.set_ylim((0.88,0.93))

# ---- Guide line: connect the AUROC of the first df_nano_ft point to the df_nano_scratch curve, then drop to the x-axis ----

# 1) Take the first df_nano_ft point
x0 = df_nano_ft['framework'].iloc[0]
y0 = df_nano_ft['AUROC'].iloc[0]

# 2) Find x corresponding to AUROC=y0 on the df_nano_scratch curve (linear interpolation)
xs = df_nano_scratch['framework'].to_numpy(dtype=float)
ys = df_nano_scratch['AUROC'].to_numpy(dtype=float)

# First find the adjacent interval containing y0 (allow increasing or decreasing)
idx = None
for i in range(len(ys) - 1):
    y1, y2 = ys[i], ys[i+1]
    if (y1 - y0) * (y2 - y0) <= 0 and (y1 != y2):
        idx = i
        break

if idx is None:
    print("y0 不在 df_nano_scratch 的 AUROC 范围内，无法插值求交点。")
else:
    x1, x2 = xs[idx], xs[idx+1]
    y1, y2 = ys[idx], ys[idx+1]

    # Linear interpolation: y0 = y1 + (y2-y1)*(x-x1)/(x2-x1)
    x_cross = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)

    # 3) Draw guide lines
    # Horizontal line: from the first df_nano_ft point to the scratch curve intersection
    ax.hlines(y=y0, xmin=x0, xmax=x_cross, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # Vertical line: from the intersection to the x-axis (use the current lower y limit)
    y_bottom = ax.get_ylim()[0]
    ax.vlines(x=x_cross, ymin=y_bottom, ymax=y0, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # Optionally mark the intersection point
    ax.scatter(
        [x_cross], [y0],
        s=40,                      # Slightly larger for clarity
        color='#DF8389',      # Edge color
        linewidths=1.5,
        zorder=5,
        alpha=0.7,
    )
    # Optionally mark x_cross
    ax.annotate(f"{x_cross:.1f}",
                xy=(x_cross, y_bottom),
                xytext=(15, 10),
                textcoords="offset points",
                ha='center', va='top', color='#D54952',
                fontproperties=prop, fontsize=10)

ax.text(
    0.75, 0.25,
    f"Mean AUROC gain = +{mean_gain:.3f}\n(at the same training data proportion) ",
    transform=ax.transAxes,
    ha='center', va='top',
    fontproperties=prop,
    color = '#D54952',
    fontsize=10
)

# from scratch first point
x_s = df_nano_scratch['framework'].iloc[0]
y_s = df_nano_scratch['AUROC'].iloc[0]

# finetuning first point
x_f = df_nano_ft['framework'].iloc[0]
y_f = df_nano_ft['AUROC'].iloc[0]

ax.annotate(
    "",                      # Draw only arrows, no text
    xy=(x_f, y_f),           # Arrow end (finetuning)
    xytext=(x_s, y_s),       # Arrow start (scratch)
    arrowprops=dict(
        arrowstyle="->",
        color="#DF8389",
        linewidth=1.5,
        alpha=0.8,
        shrinkA=2,
        shrinkB=2
    ),
    zorder=4
)

# Arrow midpoint
x_mid = (x_s + x_f) / 2
y_mid = (y_s + y_f) / 2

ax.text(
    x_mid+0.5, y_mid,
    f"+{(y_f - y_s):.3f}",
    color="#D54952",
    fontproperties=prop,
    fontsize=10,
    ha='left',
    va='center',
    zorder=5
)

# Axis labels
ax.set_xlabel("Fraction of training data (%)", fontproperties=prop, fontsize=12)
ax.set_ylabel("AUROC", fontproperties=prop, fontsize=12)

# x-axis ticks: 10 to 100, step 10
ax.set_xticks(np.arange(10, 101, 10))
for tick in ax.get_xticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# y-axis ticks
for tick in ax.get_yticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# Legend
leg = ax.legend(prop=prop, fontsize=11, frameon=False)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

plt.tight_layout()
plt.savefig(
    "Nano_scratch_ft.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()






# %%
# %%
train_protein_scratch_np = '../train_protein/output/from_scratch/eval_summary_external.csv'
train_protein_ft_np = '../train_protein/output/ft/eval_summary_external.csv'

df_protein_scratch = pd.read_csv(train_protein_scratch_np)
df_protein_scratch

# %%
df_protein_ft = pd.read_csv(train_protein_ft_np)
df_protein_ft

# %%
# Ensure sorting by framework (10,20,...,100)
df_protein_scratch   = df_protein_scratch.sort_values('framework')
df_protein_ft = df_protein_ft.sort_values('framework')
df_protein_scratch

# %%
mean_gain = (
    df_protein_ft['AUROC'].values
    - df_protein_scratch['AUROC'].values
).mean()

mean_gain

# %%
fig, ax = plt.subplots(figsize=(6.5, 3))

# First line: protein scratch
ax.plot(
    df_protein_scratch['framework'],
    df_protein_scratch['AUROC'],
    marker='o',
    linewidth=2,
    markersize=6,
    color="#75A0AF",
    label='From scratch'
)

# Second line: protein fine-tune
ax.plot(
    df_protein_ft['framework'],
    df_protein_ft['AUROC'],
    marker='s',
    linewidth=2,
    markersize=6,
    color='#DF8389',
    label='Finetuning'
)

ax.margins(y=0)  # Remove automatic padding in the y direction
ax.set_ylim((0.83,0.93))

# ---- Guide line: connect the AUROC of the first df_protein_ft point to the df_protein_scratch curve, then drop to the x-axis ----

# 1) Take the first df_protein_ft point
x0 = df_protein_ft['framework'].iloc[0]
y0 = df_protein_ft['AUROC'].iloc[0]

# 2) Find x corresponding to AUROC=y0 on the df_protein_scratch curve (linear interpolation)
xs = df_protein_scratch['framework'].to_numpy(dtype=float)
ys = df_protein_scratch['AUROC'].to_numpy(dtype=float)

# First find the adjacent interval containing y0 (allow increasing or decreasing)
idx = None
for i in range(len(ys) - 1):
    y1, y2 = ys[i], ys[i+1]
    if (y1 - y0) * (y2 - y0) <= 0 and (y1 != y2):
        idx = i
        break

if idx is None:
    print("y0 不在 df_protein_scratch 的 AUROC 范围内，无法插值求交点。")
else:
    x1, x2 = xs[idx], xs[idx+1]
    y1, y2 = ys[idx], ys[idx+1]

    # Linear interpolation: y0 = y1 + (y2-y1)*(x-x1)/(x2-x1)
    x_cross = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)

    # 3) Draw guide lines
    # Horizontal line: from the first df_protein_ft point to the scratch curve intersection
    ax.hlines(y=y0, xmin=x0, xmax=x_cross, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # Vertical line: from the intersection to the x-axis (use the current lower y limit)
    y_bottom = ax.get_ylim()[0]
    ax.vlines(x=x_cross, ymin=y_bottom, ymax=y0, colors='#DF8389', linestyles='--', alpha=0.7, linewidth=1.5)

    # Optionally mark the intersection point
    ax.scatter(
        [x_cross], [y0],
        s=40,                      # Slightly larger for clarity
        color='#DF8389',      # Edge color
        linewidths=1.5,
        zorder=5,
        alpha=0.7,
    )
    # Optionally mark x_cross
    ax.annotate(f"{x_cross:.1f}",
                xy=(x_cross, y_bottom),
                xytext=(15, 10),
                textcoords="offset points",
                ha='center', va='top', color='#D54952',
                fontproperties=prop, fontsize=10)

ax.text(
    0.75, 0.25,
    f"Mean AUROC gain = +{mean_gain:.3f}\n(at the same training data proportion) ",
    transform=ax.transAxes,
    ha='center', va='top',
    fontproperties=prop,
    color = '#D54952',
    fontsize=10
)

# from scratch first point
x_s = df_protein_scratch['framework'].iloc[0]
y_s = df_protein_scratch['AUROC'].iloc[0]

# finetuning first point
x_f = df_protein_ft['framework'].iloc[0]
y_f = df_protein_ft['AUROC'].iloc[0]

ax.annotate(
    "",                      # Draw only arrows, no text
    xy=(x_f, y_f),           # Arrow end (finetuning)
    xytext=(x_s, y_s),       # Arrow start (scratch)
    arrowprops=dict(
        arrowstyle="->",
        color="#DF8389",
        linewidth=1.5,
        alpha=0.8,
        shrinkA=2,
        shrinkB=2
    ),
    zorder=4
)

# Arrow midpoint
x_mid = (x_s + x_f) / 2
y_mid = (y_s + y_f) / 2

ax.text(
    x_mid+0.5, y_mid,
    f"+{(y_f - y_s):.3f}",
    color="#D54952",
    fontproperties=prop,
    fontsize=10,
    ha='left',
    va='center',
    zorder=5
)


# Axis labels
ax.set_xlabel("Fraction of training data (%)", fontproperties=prop, fontsize=12)
ax.set_ylabel("AUROC", fontproperties=prop, fontsize=12)

# x-axis ticks: 10 to 100, step 10
ax.set_xticks(np.arange(10, 101, 10))
for tick in ax.get_xticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# y-axis ticks
for tick in ax.get_yticklabels():
    tick.set_fontproperties(prop)
    tick.set_fontsize(10)

# Legend
leg = ax.legend(prop=prop, fontsize=11, frameon=False)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

plt.tight_layout()
plt.savefig(
    "Protein_scratch_ft.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

# %%
