# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re

# %%
df = pd.read_csv("data/protein_external_plasma_human.csv",keep_default_na=False, na_values=[''])
df

# %%
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Accession"]
)

print(len(train_df), len(test_df))

# %%
test_df.to_csv('data/protein_external_plasma_human_test.csv', index=False)

# %%
train_df_100, val_df_100 = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["Accession"]
)

print(len(train_df_100), len(val_df_100))

# %%
train_df_100.to_csv('data/protein_external_plasma_human_train_100.csv', index=False)
val_df_100.to_csv('data/protein_external_plasma_human_val_100.csv', index=False)


# %% Obtain training sets with specific fractions
for p in np.arange(0.1, 1.0, 0.1):  # 0.1 ~ 0.9
    # 1) Stratified sampling to obtain train_df_sub
    train_df_sub, _ = train_test_split(
        train_df,
        test_size=1 - p,
        random_state=42,
        stratify=train_df["Accession"]
    )

    print(f"p={p:.1f}, ratio={len(train_df_sub)/len(train_df):.4f}, n={len(train_df_sub)}")

    # 2) Random split into train/val without stratification
    train_df_final, val_df_final = train_test_split(
        train_df_sub,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_df_final.to_csv(f"data/protein_external_plasma_human_train_{int(p*100)}.csv", index=False)
    val_df_final.to_csv(f"data/protein_external_plasma_human_val_{int(p*100)}.csv", index=False)

    print(f"train: {len(train_df_final)}, val: {len(val_df_final)}")

# %%