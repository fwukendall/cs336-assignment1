import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# Adjust import path if needed based on where you save this script
try:
    from cs336_basics.bpe import BytePairEncoder
except ImportError:
    from bpe import BytePairEncoder

def get_chunk_boundaries(file_path, chunk_size_bytes):
    """Safely chunk a text file by seeking to target size and advancing to next whitespace."""
    file_size = os.path.getsize(file_path)
    boundaries = [0]
    with open(file_path, 'rb') as f:
        while boundaries[-1] < file_size:
            target = boundaries[-1] + chunk_size_bytes
            if target >= file_size:
                boundaries.append(file_size)
                break
            f.seek(target)
            char = f.read(1)
            # Advance until we hit a safe breaking point (space, newline, tab)
            while char and char not in b' \n\t':
                char = f.read(1)
            boundaries.append(f.tell())
    return boundaries

def tokenize_chunk(args):
    chunk_id, file_path, start_byte, end_byte, vocab_path, merges_path, special_tokens, tmp_dir = args

    # 1. Instantiate the BPE inside the worker so we don't incur pickling overhead
    bpe = BytePairEncoder.from_files(vocab_path, merges_path, special_tokens=special_tokens)

    # 2. Read exact byte boundaries
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        raw_bytes = f.read(end_byte - start_byte)
        text = raw_bytes.decode('utf-8', errors='ignore')

    # 3. Tokenize
    tokens = bpe.encode(text)
    
    # 4. Cast to uint16 (safe for vocabs up to 65,535) and save to disk to save RAM
    arr = np.array(tokens, dtype=np.uint16)
    out_path = os.path.join(tmp_dir, f"chunk_{chunk_id}.npy")
    np.save(out_path, arr)
    
    return out_path, len(arr)

def process_dataset(input_file, vocab_path, merges_path, output_file, special_tokens, num_workers):
    print(f"\n🚀 Starting: {input_file}")
    
    # 10MB chunks to prevent the Linux OOM Killer from crashing WSL
    chunk_size = 10 * 1024 * 1024 
    boundaries = get_chunk_boundaries(input_file, chunk_size)
    
    tmp_dir = output_file + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    tasks = [
        (i, input_file, boundaries[i], boundaries[i+1], vocab_path, merges_path, special_tokens, tmp_dir)
        for i in range(len(boundaries)-1)
    ]

    print(f"Divided into {len(tasks)} chunks. Spawning {num_workers} processes...")
    
    # Use imap_unordered so we can yield results to tqdm as soon as any worker finishes
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(tokenize_chunk, tasks), 
            total=len(tasks), 
            desc="Tokenizing Chunks",
            unit="chunk",
            smoothing=0.1 # Keeps the ETA stable
        ))

    # Re-order chunks mathematically since imap_unordered returns them out of sync
    results.sort(key=lambda x: int(os.path.basename(x[0]).split('_')[1].split('.')[0]))

    print(f"Tokenization complete. Streaming chunks into {output_file}...")
    total_tokens = sum(count for _, count in results)
    
    # memmap lets us write the final binary file out-of-core
    final_arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total_tokens,))

    idx = 0
    # Add a second progress bar for the disk writing phase
    for chunk_path, count in tqdm(results, desc="Writing Bin File", unit="chunk"):
        arr = np.load(chunk_path)
        final_arr[idx:idx+count] = arr
        idx += count
        os.remove(chunk_path) # Cleanup temp files as we go

    final_arr.flush()
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
        
    print(f"✅ Success! Saved {total_tokens:,} tokens to {output_file}.")

if __name__ == '__main__':
    # Throttled to 8 cores to protect WSL RAM limits. 
    # If your PC has 32GB+ RAM, you can safely bump this to 12 or 16.
    NUM_CORES = 6
    SPECIAL_TOKENS = ['<|endoftext|>']

    # --- OWT Paths ---
    OWT_TRAIN_TXT = 'data/owt_train.txt'
    OWT_VALID_TXT = 'data/owt_valid.txt'
    OWT_VOCAB = 'data/results/train_bpe_owt_train_vocab.pkl'
    OWT_MERGES = 'data/results/train_bpe_owt_train_merges.pkl'
    OWT_TRAIN_OUT = 'data/results/owt_train.bin'
    OWT_VALID_OUT = 'data/results/owt_valid.bin'

    # --- TinyStories Paths ---
    TINY_TRAIN_TXT = 'data/TinyStoriesV2-GPT4-train.txt'
    TINY_VALID_TXT = 'data/TinyStoriesV2-GPT4-valid.txt'
    TINY_VOCAB = 'data/results/train_bpe_tiny_train_vocab.pkl'
    TINY_MERGES = 'data/results/train_bpe_tiny_train_merges.pkl'
    TINY_TRAIN_OUT = 'data/results/tiny_train.bin'
    TINY_VALID_OUT = 'data/results/tiny_valid.bin'

    # Process TinyStories
    if os.path.exists(TINY_VOCAB) and os.path.exists(TINY_TRAIN_TXT):
        process_dataset(TINY_VALID_TXT, TINY_VOCAB, TINY_MERGES, TINY_VALID_OUT, SPECIAL_TOKENS, NUM_CORES)
        process_dataset(TINY_TRAIN_TXT, TINY_VOCAB, TINY_MERGES, TINY_TRAIN_OUT, SPECIAL_TOKENS, NUM_CORES)
    else:
        print(f"⚠️ Skipping TinyStories: Missing vocab/merges or text file.")
        
    # Process OWT
    if os.path.exists(OWT_VOCAB) and os.path.exists(OWT_TRAIN_TXT):
        process_dataset(OWT_TRAIN_TXT, OWT_VOCAB, OWT_MERGES, OWT_TRAIN_OUT, SPECIAL_TOKENS, NUM_CORES)
        process_dataset(OWT_VALID_TXT, OWT_VOCAB, OWT_MERGES, OWT_VALID_OUT, SPECIAL_TOKENS, NUM_CORES)
    else:
        print(f"⚠️ Skipping OWT: Missing vocab/merges or text file.")

    print("\n🎉 All distributed tokenization jobs finished!")