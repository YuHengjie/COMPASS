# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re

# %%
df = pd.read_csv("../data/all_curated_nonfill.csv",keep_default_na=False, na_values=[''])
df

# %%
df["RPA"].hist(bins=200)

# %%
rpa_values = df['RPA'].values
threshold = 1e-6
count_below_threshold = (rpa_values < threshold).sum()  # Count samples below threshold
total_count = len(rpa_values)  # Total sample count
percentage_below_threshold = count_below_threshold / total_count * 100  # Compute the proportion

print(f"Number of samples with RPA < threshold: {count_below_threshold}")
print(f"Total number of samples: {total_count}")
print(f"Percentage of samples with RPA < threshold: {percentage_below_threshold:.2f}%")

# %%
df['Protein corona composition'] = df['RPA'].apply(lambda x: 0 if x < threshold else 1)
sum(df['Protein corona composition'])

# %%
df['Incubation protein source'].unique()

# %%
df['Protein source organism'].unique()

# %%
mask = ~df['Incubation protein source'].str.contains(
    'serum|plasma|blood', case=False, na=False
)
df_not_blood = df[mask]
df_not_blood['Incubation protein source'].unique()
print(f"筛选到 {len(df_not_blood)} 条记录。")

# %%
mask_source = df['Incubation protein source'].str.contains(
    'serum|plasma|blood', case=False, na=False
)
mask_not_human = ~df['Protein source organism'].str.contains(
    'human', case=False, na=False
)

df_blood_nonhuman = df[mask_source & mask_not_human]

print(df_blood_nonhuman['Protein source organism'].unique())
print(f"筛选到 {len(df_blood_nonhuman)} 条记录。")


# %%
mask_serum = df['Incubation protein source'].str.contains(
    'serum', case=False, na=False
)
mask_human = df['Protein source organism'].str.contains(
    'human', case=False, na=False
)

df_serum_human = df[mask_serum & mask_human]

print(df_serum_human['Protein source organism'].unique())
print(f"筛选到 {len(df_serum_human)} 条记录。")

# %%
mask_plasma = df['Incubation protein source'].str.contains(
    'plasma', case=False, na=False
)
mask_human = df['Protein source organism'].str.contains(
    'human', case=False, na=False
)

df_plasma_human = df[mask_plasma & mask_human]

print(df_plasma_human['Protein source organism'].unique())
print(f"筛选到 {len(df_plasma_human)} 条记录。")

# %%
prob_label = pd.read_csv("../data/problematic_labels.csv",)
prob_label

# %%
# Extract unique Label values from prob_label
problematic_labels = prob_label['Label'].dropna().unique()

# %%
df_plasma_human_low = df_plasma_human[df_plasma_human['Label'].isin(problematic_labels)]
print(f"筛选到 {len(df_plasma_human_low)} 条记录。")

# %%
df_plasma_human_high = df_plasma_human[~df_plasma_human['Label'].isin(problematic_labels)]
print(f"筛选到 {len(df_plasma_human_high)} 条记录。")

# %%
# Split by backslash and find the part containing '10.' in each row
doi_series = (
    df_plasma_human_high['Label']
    .str.split('\\')                                   # Split path
    .apply(lambda parts: next((p for p in parts if isinstance(p, str) and p.startswith('10.')), None))
)

# Show the first rows and the number of unique DOIs
print(doi_series)
print("Unique DOI count:", doi_series.nunique())

# %%
# Use Series.str.replace(pat, repl, n=1) to replace only the first '_'
processed_doi_series = doi_series.astype(str).str.replace('_', '/', n=1)
# Get unique DOIs and remove None/NaN values
unique_dois = processed_doi_series.dropna().unique()
print(len(unique_dois))
# Define the output filename
output_filename = 'unique_processed_dois.txt'

# Write unique DOI values to a file, one per line
try:
    with open(output_filename, 'w') as f:
        # Write with join and newline characters
        f.write('\n'.join(unique_dois))
    
    print(f"\n✅ 成功将 {len(unique_dois)} 个独特 DOI 保存到文件: {output_filename}")
    
except Exception as e:
    print(f"\n❌ 写入文件时发生错误: {e}")
    

# %%
# Shuffle df_plasma_human_high
df_plasma_human_high = df_plasma_human_high.sample(frac=1, random_state=42)

# Randomly sample 15% of df_plasma_human_high as the test set
df_plasma_human_high_test = df_plasma_human_high.sample(frac=0.15, random_state=42)

# Randomly sample 15% of the remaining 85% as the validation set
df_plasma_human_high_remaining = df_plasma_human_high.drop(df_plasma_human_high_test.index)
df_plasma_human_high_val = df_plasma_human_high_remaining.sample(frac=0.1764706, random_state=42)  # 15% of the original dataset

# The remaining 70% is the training set
df_plasma_human_high_train = df_plasma_human_high_remaining.drop(df_plasma_human_high_val.index)

# %%
# Shuffle df_not_blood
df_not_blood = df_not_blood.sample(frac=1, random_state=42)

# %%
# Shuffle df_blood_nonhuman
df_blood_nonhuman = df_blood_nonhuman.sample(frac=1, random_state=42)

# %% Merge and shuffle df_serum_human_plasma_low
df_serum_human_plasma_low = pd.concat([df_serum_human, df_plasma_human_low], )
df_serum_human_plasma_low = df_serum_human_plasma_low.sample(frac=1, random_state=42)
df_serum_human_plasma_low


# %%
#df_plasma_human_high_final = df_plasma_human_high_train[df_plasma_human_high_train['Overall data quality'] > 0.7]
df_plasma_human_high_final = df_plasma_human_high_train.sample(
    frac=0.75, 
    random_state=42 # Use a fixed random seed
)

# %%
# Split by backslash and find the part containing '10.' in each row
doi_series = (
    df_plasma_human_high_final['Label']
    .str.split('\\')                                   # Split path
    .apply(lambda parts: next((p for p in parts if isinstance(p, str) and p.startswith('10.')), None))
)

# Show the first rows and the number of unique DOIs
print(doi_series)
print("Unique DOI count:", doi_series.nunique())

# %%
# Ensure the data folder exists
os.makedirs("data", exist_ok=True)

# Save the indices of each DataFrame as a dictionary
index_dict = {
    "not_blood": df_not_blood.index.tolist(),
    "blood_nonhuman": df_blood_nonhuman.index.tolist(),
    "serum_human_plasma_low": df_serum_human_plasma_low.index.tolist(),
    
    "plasma_human_high": df_plasma_human_high.index.tolist(),
    "plasma_human_high_train": df_plasma_human_high_train.index.tolist(),
    "plasma_human_high_val": df_plasma_human_high_val.index.tolist(),
    "plasma_human_high_test": df_plasma_human_high_test.index.tolist(),

    "plasma_human_high_final": df_plasma_human_high_final.index.tolist()
}

# Save as a JSON file
with open("data/data_split_indices.json", "w", encoding="utf-8") as f:
    json.dump(index_dict, f, ensure_ascii=False, indent=4)

print("✅ 索引已保存到 data/data_split_indices.json")

# %%
for name, idx_list in index_dict.items():
    print(f"{name}: {len(idx_list)}")


# %%
# Read indices from the JSON file
with open("data/data_split_indices.json", "r", encoding="utf-8") as f:
    index_dict = json.load(f)

# %%
# Select subsets from df in batch according to index_dict
df_not_blood = df.loc[index_dict["not_blood"]]
df_blood_nonhuman = df.loc[index_dict["blood_nonhuman"]]
df_serum_human_plasma_low = df.loc[index_dict["serum_human_plasma_low"]]

df_plasma_human_high = df.loc[index_dict["plasma_human_high"]]
df_plasma_human_high_train = df.loc[index_dict["plasma_human_high_train"]]
df_plasma_human_high_val = df.loc[index_dict["plasma_human_high_val"]]
df_plasma_human_high_test = df.loc[index_dict["plasma_human_high_test"]]

df_plasma_human_high_final = df.loc[index_dict["plasma_human_high_final"]]

# %%
# Print the size of each dataset for confirmation
for name, subset in {
    "not_blood": df_not_blood,
    "blood_nonhuman": df_blood_nonhuman,
    "serum_human_plasma_low": df_serum_human_plasma_low,
    "plasma_human_high": df_plasma_human_high,
    "plasma_human_high_train": df_plasma_human_high_train,
    "plasma_human_high_val": df_plasma_human_high_val,
    "plasma_human_high_test": df_plasma_human_high_test,
    "plasma_human_high_final": df_plasma_human_high_final
}.items():
    print(f"{name}: {len(subset)}")
    
# %%
# Ensure the target directory exists
os.makedirs("data", exist_ok=True)

combined_dfs = {}
for key in index_dict.keys():
    df_name = f"df_{key}"

    if df_name in globals():
        df_to_save = globals()[df_name]
        combined_dfs[key] = df_to_save # 
        save_path = f"data/basic_{key}.csv"
        df_to_save.to_csv(save_path, index=False)
        print(f"{key}: 保存 df_{key}，共 {len(df_to_save)} 条记录，已保存到 {save_path}")


# %% Add 100% of the far-domain data
# Copy not_blood data
df_nb = combined_dfs["not_blood"].copy()

# Randomly sample an equal number of records from plasma_human_high
n_nb = len(df_nb)
# Use combined_dfs["plasma_human_high"] as the sampling pool to keep the data consistent
df_phh_nb = combined_dfs["plasma_human_high"].sample(
    n=n_nb, replace=False, random_state=42
).copy()

# Merge and shuffle
df_nb_addhigh = pd.concat([df_nb, df_phh_nb], axis=0, ignore_index=True)
df_nb_addhigh = df_nb_addhigh.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
save_path_nb = "data/basic_not_blood_addhigh100.csv"
df_nb_addhigh.to_csv(save_path_nb, index=False)

print(f"✅ 已保存 {save_path_nb}，"
      f"not_blood 原始 {len(df_nb)} 条，"
      f"plasma_human_high 抽样 {len(df_phh_nb)} 条，"
      f"合并后共 {len(df_nb_addhigh)} 条。")

train_nb, val_nb = train_test_split(
    df_nb_addhigh, test_size=0.15, random_state=42, shuffle=True
)

train_nb.to_csv("data/basic_not_blood_addhigh100_train.csv", index=False)
val_nb.to_csv("data/basic_not_blood_addhigh100_val.csv", index=False)

print(f"✅ not_blood_addhigh 划分完成：train={len(train_nb)}, val={len(val_nb)}")

# %% Add 100% of the far-domain data
# Copy blood_nonhuman data
df_bn = combined_dfs["blood_nonhuman"].copy()

# Randomly sample an equal number of records from plasma_human_high
n = len(df_bn)
df_phh = combined_dfs["plasma_human_high"].sample(
    n=n, replace=False, random_state=42
).copy()

# Merge and shuffle
df_bn_addhigh = pd.concat([df_bn, df_phh], axis=0, ignore_index=True)
df_bn_addhigh = df_bn_addhigh.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
save_path = "data/basic_blood_nonhuman_addhigh100.csv"
df_bn_addhigh.to_csv(save_path, index=False)

print(f"✅ 已保存 data/basic_blood_nonhuman_addhigh100.csv，"
      f"blood_nonhuman 原始 {len(df_bn)} 条，"
      f"plasma_human_high 抽样 {len(df_phh)} 条，"
      f"合并后共 {len(df_bn_addhigh)} 条。")

# === Split blood_nonhuman_addhigh 85/15 ===
train_bn, val_bn = train_test_split(
    df_bn_addhigh, test_size=0.15, random_state=42, shuffle=True
)

train_bn.to_csv("data/basic_blood_nonhuman_addhigh100_train.csv", index=False)
val_bn.to_csv("data/basic_blood_nonhuman_addhigh100_val.csv", index=False)

print(f"✅ blood_nonhuman_addhigh 划分完成：train={len(train_bn)}, val={len(val_bn)}")


# %% Add 100% of the near-domain data
# Copy serum_human_plasma_low data
df_shpl = combined_dfs["serum_human_plasma_low"].copy()
df_phh = combined_dfs["plasma_human_high"].copy()
# Randomly sample 100% of serum from plasma_human_high
m = len(df_shpl)
k = m
high_count = len(df_phh)

# 1. Check whether the target exceeds the total sample size
if k > high_count:
    # Target sample size exceeds the population: use sampling with replacement (replace=True)
    print(f"Warning: Target sample size ({k}) exceeds total population ({high_count}). Switching to replacement sampling.")
    df_phh_sample = df_phh.sample(
        n=k, 
        replace=True,  # Sampling with replacement (oversampling)
        random_state=42
    ).copy()
else:
    # Target sample size is at most the population: use sampling without replacement (replace=False)
    print(f"Target sample size ({k}) is safe. Using non-replacement sampling.")
    df_phh_sample = df_phh.sample(
        n=k, 
        replace=False, # Sampling without replacement (normal sampling)
        random_state=42
    ).copy()

# Merge and shuffle
df_shpl_addhigh100 = pd.concat([df_shpl, df_phh_sample], axis=0, ignore_index=True)
df_shpl_addhigh100 = df_shpl_addhigh100.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
save_path = "data/basic_serum_human_plasma_low_addhigh100.csv"
df_shpl_addhigh100.to_csv(save_path, index=False)

print(
    f"✅ 已保存 {save_path}，"
    f"serum_human_plasma_low 原始 {len(df_shpl)} 条，"
    f"plasma_human_high 抽样 {len(df_phh_sample)} 条，"
    f"合并后共 {len(df_shpl_addhigh100)} 条。"
)

# === Split serum_human_plasma_low_addhigh100 85/15 ===
train_shpl, val_shpl = train_test_split(
    df_shpl_addhigh100, test_size=0.15, random_state=42, shuffle=True
)

train_shpl.to_csv("data/basic_serum_human_plasma_low_addhigh100_train.csv", index=False)
val_shpl.to_csv("data/basic_serum_human_plasma_low_addhigh100_val.csv", index=False)

print(f"✅ serum_human_plasma_low_addhigh100 划分完成：train={len(train_shpl)}, val={len(val_shpl)}")

# %%
