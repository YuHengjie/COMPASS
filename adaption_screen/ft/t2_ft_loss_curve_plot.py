# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# Font
font_path = "/path/to/your/fonts/ArialMdm.ttf"
prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# %%
# =========================
# Read history
# =========================

history = np.load(
    "output/fine_tuning/history_ft.npy",
    allow_pickle=True
).item()


print(history.keys())


train_loss = history["train_loss"]
val_loss = history["val_loss"]

epochs = np.arange(1, len(train_loss)+1)


# %%
# =========================
# Plot
# =========================

fig, ax = plt.subplots(
    figsize=(5, 3.5)
)


ax.plot(
    epochs,
    train_loss,
    linewidth=2,
    label="Training loss"
)


ax.plot(
    epochs,
    val_loss,
    linewidth=2,
    label="Validation loss"
)



# =========================
# Axes
# =========================

ax.set_xlabel(
    "Epoch",
    fontproperties=prop,
    fontsize=12
)


ax.set_ylabel(
    "Loss",
    fontproperties=prop,
    fontsize=12
)



# tick font

for label in ax.get_xticklabels():
    label.set_fontproperties(prop)

for label in ax.get_yticklabels():
    label.set_fontproperties(prop)



# =========================
# Styling
# =========================

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.legend(
    frameon=False,
    prop=prop,
    fontsize=10
)


plt.tight_layout()


plt.savefig(
    "output/fine_tuning_loss_curve.png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()

# %%
