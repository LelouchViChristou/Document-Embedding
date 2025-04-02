import os
import time
import gc
import logging
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict

# === Logging configuration ===
logging.basicConfig(
    filename='embedding_log_final.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# === Configuration ===
# Folder where your processed (chunked) files are stored.
DATA_DIR = r"C:\Users\chris\Desktop\Document Embeddings\Document-Embedding\processed_data"
BATCH_SIZE = 64

# === Load the SentenceTransformer model once ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
logger.info(f"Using device: {device}")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
model.eval()

# === Function to encode one article's chunks ===
def encode_article(chunks):
    """
    Encode a list of text chunks (for one article) in mini-batches.
    Returns a list of embeddings as standard Python lists.
    """
    embeddings = []
    if not isinstance(chunks, list) or len(chunks) == 0:
        return []
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            emb = model.encode(
                batch,
                convert_to_tensor=True,
                batch_size=BATCH_SIZE,
                show_progress_bar=False
            ).cpu().numpy()
            embeddings.extend(emb.tolist())
    return embeddings

# === Function to process one file and replace original ===
def process_file(fname):
    input_path = os.path.join(DATA_DIR, fname)
    # Use a temporary output file name; once successful, replace the original file.
    temp_output = os.path.join(DATA_DIR, fname.replace("_chunked.parquet", "_embedded_temp.parquet"))
    final_output = os.path.join(DATA_DIR, fname.replace("_chunked.parquet", "_embedded.parquet"))
    
    print(f"\n=== Processing {fname} ===")
    logger.info(f"=== Processing {fname} ===")
    
    # Load the chunked DataFrame
    df = pd.read_parquet(input_path, engine="fastparquet")
    
    # Filter out rows with invalid or empty text_chunks
    df = df[df["text_chunks"].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
    print(f"Valid rows: {len(df)}")
    logger.info(f"Valid rows: {len(df)}")
    
    # Encode each article's chunks using a progress bar
    tqdm.pandas(desc="Embedding rows")
    df["embeddings"] = df["text_chunks"].progress_apply(encode_article)
    
    # Save the new DataFrame to a temporary file
    df.to_parquet(temp_output, engine="fastparquet")
    print(f"Saved temporary output to: {temp_output}")
    logger.info(f"Saved temporary output to: {temp_output}")
    
    # Clean up memory
    del df
    gc.collect()
    
    # Replace the original file with the new one:
    # Remove the original file, then rename the temporary file.
    os.remove(input_path)
    os.rename(temp_output, final_output)
    print(f"Replaced original file with: {final_output}")
    logger.info(f"Replaced original file with: {final_output}")

# === Main loop: Process each file with retry mechanism ===
files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("wiki_chunk_") and f.endswith("_chunked.parquet"))
print(f"Found {len(files)} files to process.")
logger.info(f"Found {len(files)} files to process.")

for fname in files:
    success = False
    while not success:
        try:
            process_file(fname)
            success = True
            print(f"Processing of {fname} succeeded.")
            logger.info(f"Processing of {fname} succeeded.")
        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")
            logger.error(f"[ERROR] Failed to process {fname}: {e}")
            print("Retrying in 5 seconds...")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)
