# %%
import torch
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
import multiprocessing as mp
import pickle # Import pickle
import os # Import os for checking file existence

# %%
# Use GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================
# Test GPU memory usage
# ==============================
def test_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} 显存分配情况：")
        print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  保留: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

# test_gpu_memory()

# %%
# ==============================
# Piecewise function with overlap
# ==============================
max_len = 1024   # Maximum length per segment
overlap = 64

def chunk_sequence(seq, chunk_size=max_len, overlap=overlap):
    """把序列切成若干段，每段有 overlap 个氨基酸与前一段重叠。"""
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    step = chunk_size - overlap
    chunks = [seq[i:i + chunk_size] for i in range(0, len(seq), step)]
    return chunks

# %%
# Define the global save path
EMBEDDING_FILE = "esmfold_protein_embeddings_unseen.pkl"

def clean_sequence(seq):
    seq = seq.strip().upper()
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join([aa if aa in valid_aas else "X" for aa in seq])
    return cleaned

def compute_protein_embedding_single_gpu(seq, model, tokenizer, chunk_size, overlap, device):
    """单序列切片处理函数 (与你原代码的 compute_protein_embedding 相似)"""
    # ... (reuse the single-sequence handling logic from the original compute_protein_embedding)
    # The single-sequence handling logic from the original code is reused here; no batching is needed
    seq = clean_sequence(seq)
    chunks = chunk_sequence(seq, chunk_size, overlap)
    all_embeddings = []

    with torch.no_grad():
        for chunk in chunks:
            tokenized_input = tokenizer([chunk], return_tensors="pt", 
                                        add_special_tokens=False,padding=True,)["input_ids"].to(device)
            # Call the model directly because there is no DataParallel
            output = model(tokenized_input) 
            # Ensure output["states"] has shape [1, L, 384] or similar
            last_layer = output["states"][-1, 0] 
            chunk_emb = last_layer.mean(dim=0)
            all_embeddings.append(chunk_emb.cpu().to(torch.float32))
            
            # Clear GPU memory
            del tokenized_input, output, last_layer
            torch.cuda.empty_cache()

    all_embeddings = torch.stack(all_embeddings)
    protein_embedding = all_embeddings.mean(dim=0)
    return protein_embedding.cpu().numpy()

def gpu_worker(rank, df_subset, model_path, tokenizer_path, chunk_size, overlap, result_queue):
    """每个 GPU 进程执行的函数"""
    device = torch.device(f"cuda:{rank}")
    print(f"Worker {rank}: Loading model on {device}")
    
    # Load model and tokenizer onto their GPUs
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EsmForProteinFolding.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    
    # Inference
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"GPU {rank} Progress"): # Add tqdm for subprocesses
        accession = row["Accession"]
        seq = row["Sequence"]
        
        try:
            # Call the single-sequence processing function
            emb = compute_protein_embedding_single_gpu(seq, model, tokenizer, chunk_size, overlap, device)
            result_queue.put((accession, emb))
        except Exception as e:
            print(f"Error processing {accession} on GPU {rank}: {e}")
            # If an error occurs, still send None; the main process ignores it, but execution continues here
            result_queue.put((accession, None)) 
            
    # Key change: send the end signal
    result_queue.put(('SENTINEL', None)) 
            
    # Explicit cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"Worker {rank}: Finished.")
    
# ==============================
# Main process: coordinate and save
# ==============================
def run_parallel_inference(df, model_path, tokenizer_path, chunk_size, overlap, num_gpus):

    # Step A: Load existing results
    embedding_dict = {}
    if os.path.exists(EMBEDDING_FILE):
        try:
            with open(EMBEDDING_FILE, "rb") as f:
                embedding_dict = pickle.load(f)
            print(f"Loaded checkpoint with {len(embedding_dict)} embeddings.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            embedding_dict = {}

    processed_accessions = set(embedding_dict.keys())
    df_unprocessed = df[~df["Accession"].isin(processed_accessions)]

    if len(df_unprocessed) == 0:
        print("All sequences are already encoded.")
        return embedding_dict

    print(f"Total sequences to process: {len(df_unprocessed)}")

    # Step B: Split data
    num_gpus = min(num_gpus, len(df_unprocessed))
    df_splits = np.array_split(df_unprocessed, num_gpus)
    result_queue = mp.Queue()

    # Step C: Start subprocesses
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, df_splits[rank], model_path, tokenizer_path, chunk_size, overlap, result_queue),
        )
        p.start()
        processes.append(p)

    # Step D: Main process collects results and saves them
    active_processes = num_gpus
    SAVE_INTERVAL = max(100, num_gpus)
    pbar = tqdm(total=len(df_unprocessed), desc="Overall Progress")

    while active_processes > 0:
        try:
            accession, emb = result_queue.get(timeout=1)

            if accession == "SENTINEL":
                active_processes -= 1
                continue

            if emb is not None:
                embedding_dict[accession] = emb
                pbar.update(1)

                # Periodic save (atomic write)
                if len(embedding_dict) % SAVE_INTERVAL == 0:
                    tmp_file = EMBEDDING_FILE + ".tmp"
                    with open(tmp_file, "wb") as f:
                        pickle.dump(embedding_dict, f)
                    os.replace(tmp_file, EMBEDDING_FILE)
                    pbar.set_postfix({"Saved": len(embedding_dict)})

        except Exception:
            pass  # Queue is currently empty

    for p in processes:
        p.join()

    # Step E: Final save
    tmp_file = EMBEDDING_FILE + ".tmp"
    with open(tmp_file, "wb") as f:
        pickle.dump(embedding_dict, f)
    os.replace(tmp_file, EMBEDDING_FILE)

    pbar.close()
    print(f"✅ Finished encoding. Total proteins: {len(embedding_dict)} saved to {EMBEDDING_FILE}")
    return embedding_dict


# %%
# =========================================================
# Main entry point
# =========================================================
if __name__ == '__main__':
    # 1. Set the multiprocessing start method here to avoid duplicate setup
    try:
        # force=True ensures the setting takes effect
        mp.set_start_method('spawn', force=True) 
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Could not set start method: {e}")
        
    # 2. Global parameters and data loading (execute once in the main process)
    # Base model (ESMFold): download from Hugging Face or ModelScope into the folder below, then update this hard-coded path.
    model_path = "/path/to/your/pretrained_model/esmfold_v1"
    tokenizer_path = model_path
    num_gpus = 8
    max_len = 1024
    overlap = 64
    
    # Assume df is already loaded
    # Ensure pd.read_excel is also inside the if __name__ == '__main__': block
    df = pd.read_excel("protein_seq_2504_unseen.xlsx", )

    # 3. Run parallel inference (execute once in the main process)
    print(f"Starting parallel inference on {num_gpus} GPUs...")
    embedding_dict = run_parallel_inference(df, model_path, tokenizer_path, max_len, overlap, num_gpus)

    print(f"Total proteins encoded: {len(embedding_dict)}")
    
# %%
