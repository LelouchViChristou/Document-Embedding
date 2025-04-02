from datasets import load_dataset
import os

# === CONFIGURATION ===
SAVE_DIR = "/home/lenos/Desktop/Document embedding/data"
NUM_FILES = 30
FILE_PREFIX = "wiki_chunk"  # files like wiki_chunk_0.parquet, wiki_chunk_1.parquet

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# === 1. Load dataset (non-streaming) ===
print("Loading full Wikipedia dataset...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# === 2. Compute split size ===
total_len = len(dataset)
chunk_size = total_len // NUM_FILES

# === 3. Split and save to parquet ===
for i in range(NUM_FILES):
    start_idx = i * chunk_size
    # last file takes the remainder too
    end_idx = (i + 1) * chunk_size if i < NUM_FILES - 1 else total_len

    chunk = dataset.select(range(start_idx, end_idx))

    file_path = os.path.join(SAVE_DIR, f"{FILE_PREFIX}_{i}.parquet")
    chunk.to_parquet(file_path)

    print(f"Saved chunk {i + 1}/{NUM_FILES} -> {file_path} ({end_idx - start_idx} examples)")
