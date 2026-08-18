# %%
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
import pandas as pd
# Use GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
df = pd.read_excel("protein_seq_20250418.xlsx", index_col=0)
df = df.iloc[17000:,:]  # Process only the first 1000 records
df

# %% Load tokenizer and model
model_path = "/yuhengjie/backup/pretrainedmodel/esm2_t36_3B_UR50D"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model = model.to(device)
model.eval()

# %%
# Set batch size
batch_size = 8  # Adjust according to GPU memory

# Dictionary for saved results
embedding_dict = {}

# Iterate over batches
for i in tqdm(range(0, len(df), batch_size), desc="Encoding"):
    batch_df = df.iloc[i:i+batch_size]
    accessions = batch_df["Accession"].tolist()
    sequences = batch_df["Sequence"].tolist()

    # tokenize
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # attention mask-based mean pooling
        attention_mask = inputs["attention_mask"]
        embeddings = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

    # Save into dict
    for acc, emb in zip(accessions, embeddings.cpu()):
        embedding_dict[acc] = emb.numpy()  # Or emb.detach()
        
    # Clear GPU memory
    del inputs
    del outputs
    del last_hidden
    del embeddings
    torch.cuda.empty_cache()  # Free GPU memory

# %%
# Save to file
with open("protein_embeddings_2.pkl", "wb") as f:
    pickle.dump(embedding_dict, f)
    