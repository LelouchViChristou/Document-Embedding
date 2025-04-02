import os
import pandas as pd
import re
import gc
from tqdm import tqdm
import time
import multiprocessing

# === Configuration ===
DATA_DIR = r"C:\Users\chris\Desktop\Document Embeddings\Document-Embedding\data"
OUTPUT_DIR = r"C:\Users\chris\Desktop\Document Embeddings\Document-Embedding\processed_data"
MAX_WORDS = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Chunking function (regex-based, no NLTK) ===
def split_text_to_chunks(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return []
        if len(text) > 100_000:
            return []
        sentence_splitter = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_splitter.split(text.strip())
        chunks = []
        current_chunk = []
        current_word_count = 0
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            if current_word_count + word_count > MAX_WORDS:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count
            else:
                current_chunk.append(sentence)
                current_word_count += word_count
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    except Exception as e:
        print(f"[Chunking Error]: {e}")
        return []

# === Function to process one file ===
def process_file(fname):
    input_path = os.path.join(DATA_DIR, fname)
    output_path = os.path.join(OUTPUT_DIR, fname.replace(".parquet", "_chunked.parquet"))
    print(f"\n=== Chunking {fname} ===")
    try:
        df = pd.read_parquet(input_path, engine="fastparquet")
        # Filter out rows with no valid text
        df = df[df["text"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)].reset_index(drop=True)
        tqdm.pandas(desc=f"Chunking {fname}")
        df["text_chunks"] = df["text"].progress_apply(split_text_to_chunks)
        df.drop(columns=["text"], inplace=True)
        df.to_parquet(output_path, engine="fastparquet")
        print(f"✅ Saved to: {output_path}")
        # Explicitly clear memory
        del df
        gc.collect()
    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")
        # Propagate the error so the process exits with nonzero code.
        raise e

if __name__ == "__main__":
    # Get a sorted list of files that match the pattern.
    input_files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("wiki_chunk_") and f.endswith(".parquet"))
    print(f"Found {len(input_files)} files to process.")

    # Process each file in its own process and restart on failure.
    for fname in input_files:
        success = False
        while not success:
            p = multiprocessing.Process(target=process_file, args=(fname,))
            p.start()
            p.join()
            if p.exitcode == 0:
                success = True
                print(f"Processing of {fname} succeeded.")
            else:
                print(f"Processing {fname} failed (exit code {p.exitcode}). Restarting after 5 seconds...")
                time.sleep(5)
