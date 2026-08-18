# %%
# %%
import pickle

# %%
# Read the first file
with open("protein_embeddings_all.pkl", "rb") as f:
    dict1 = pickle.load(f)

# %%
# Read the second file
with open("protein_embeddings_unseen.pkl", "rb") as f:
    dict2 = pickle.load(f)

# %%
# Merge two dictionaries
# If two dictionaries share a key, the later one overwrites the earlier one
merged_dict = {**dict1, **dict2}

# Save the merged dictionary
with open("protein_embeddings_all_esm.pkl", "wb") as f:
    pickle.dump(merged_dict, f)




# %%
# Read the first file
with open("esmfold_protein_embeddings.pkl", "rb") as f:
    dict1 = pickle.load(f)

# %%
# Read the second file
with open("esmfold_protein_embeddings_unseen.pkl", "rb") as f:
    dict2 = pickle.load(f)

# %%
# Merge two dictionaries
# If two dictionaries share a key, the later one overwrites the earlier one
merged_dict = {**dict1, **dict2}

# Save the merged dictionary
with open("protein_embeddings_all_esmfold.pkl", "wb") as f:
    pickle.dump(merged_dict, f)

# %%
