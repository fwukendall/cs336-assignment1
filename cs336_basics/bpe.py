import os
from typing import BinaryIO
import regex as re
from typing import Tuple, Dict, List
from collections import Counter
from functools import partial
from itertools import chain
import multiprocessing as mp
import json
import numpy as np
import pandas as pd

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_chunk_size: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    desired_num_chunks = file_size // desired_chunk_size + 1
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    stlen = len(split_special_token)
    rewind_length = stlen - 1
    tail = b""

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = tail + file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
            tail = mini_chunk[-rewind_length:]

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(chunk, pt_counts=None, special_tokens=None) -> Dict[Tuple, int]:
    if pt_counts is None:
        pt_counts = Counter()
    ssr = '|'.join([re.escape(st.decode('utf-8')) for st in special_tokens])
    last_end = 0
    assert special_tokens is not None
    for match in re.finditer(ssr, chunk):
        sub_chunk = chunk[last_end:match.start()]
        last_end = match.end()
        pt_counts[match.group()] += 1
        if not sub_chunk:
            continue
        for pt in re.finditer(PAT, sub_chunk):
            pt_counts[pt.group()] += 1
    # take care of tail
    sub_chunk = chunk[last_end:]
    if sub_chunk:
        for pt in re.finditer(PAT, sub_chunk):
            pt_counts[pt.group()] += 1
    return pt_counts

def mp_pt(chunk_start, chunk_end, input_path, special_tokens=None):
    with open(input_path, 'rb') as f:
        f.seek(chunk_start)
        chunk = f.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")
    chunk_pt_count = pre_tokenize(chunk, special_tokens=special_tokens)
    return chunk_pt_count

## Usage
def get_pretoken_counts(input_path, desired_chunk_size=int(1e6), num_chunks=None, num_processes=8,
                        special_tokens=None, special_split_token='<|endoftext|>'):
    pt_counts = None
    chunk_count = 0
    special_split_token = special_split_token.encode('utf-8')
    if special_tokens is None:
        special_tokens = [special_split_token]
    else:
        special_tokens = special_tokens + [special_split_token]
        special_tokens = sorted(set(special_tokens))

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunk_size, special_split_token)
    if num_processes == 1:
        # run serial
        with open(input_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                pt_counts = pre_tokenize(chunk, pt_counts=pt_counts, special_tokens=special_tokens)
                chunk_count += 1
                if num_chunks is not None and chunk_count >= num_chunks:
                    break
    else:
        pt_call = partial(mp_pt, input_path=input_path, special_tokens=[special_split_token])
        n_proc = min(num_processes, len(boundaries) - 1)
        with mp.Pool(n_proc) as pool:
            # print(f'Spawning {n_proc} processes!')
            chunk_pt_counts = pool.starmap(pt_call, zip(boundaries[:-1], boundaries[1:]))
        pt_counts = sum(chunk_pt_counts, start=Counter())
    return pt_counts

def count_subpts(substats, start_len=2, width=20, counter=None):
    if counter is None:
        counter = Counter()
    for pt, num_occur in substats.items():
        bstr = pt.encode('utf-8')
        for cur_len in range(start_len, start_len+width):
            if len(bstr) < cur_len:
                continue
            for start_pos in range(0, len(bstr)-cur_len+1):
                counter[tuple([bytes([c]) for c in bstr[start_pos:(start_pos+cur_len)]])] += num_occur
    return counter

def broadcast_merge(pt_split: List[bytes], merge: tuple[bytes, bytes], num_occur, counter=None):
    if counter is None:
        counter = Counter()
    m1, m2 = merge
    m3 = m1 + m2
    if m3 not in b''.join(pt_split) or m1 not in pt_split:
        return (pt_split, counter)
    can_find = True
    new_split = []
    sub_split = pt_split[:]
    while can_find:
        next_i = sub_split.index(m1)
        new_split += sub_split[:next_i]
        if len(sub_split) > next_i + 1 and sub_split[next_i+1] == m2:
            new_split.append(m3)
            sub_split = sub_split[next_i+2:]
        else:
            new_split.append(m1)
            sub_split = sub_split[next_i+1:]
        can_find = m1 in sub_split
    new_split += sub_split

    # if len(new_split) != len(pt_split):
    #     for i in range(len(new_split) - 1):
    #         counter[(new_split[i], new_split[i+1])] += num_occur

    #     for i in range(len(pt_split) - 1):
    #         counter[(pt_split[i], pt_split[i+1])] -= num_occur

    for i, g in enumerate(new_split):
        if g == m3:
            if i > 0 and new_split[i-1] != m3:
                counter[(new_split[i-1], m3)] += num_occur
                counter[(new_split[i-1], m1)] -= num_occur

            if i < len(new_split) - 1:
                counter[(m3, new_split[i+1])] += num_occur
                if new_split[i+1] != m3:
                    counter[(m2, new_split[i+1])] -= num_occur
                else:
                    counter[(m2, m1)] -= num_occur
    return new_split, counter

def pt_batch_merge(
    pt_stats: Dict[str, int],
    pt_splits: Dict[str, List[bytes]],
    merge: tuple[bytes, bytes],
) -> Tuple[Dict[str, List[bytes]], Dict[Tuple[bytes, bytes], int]]:
    change_counter = Counter()
    updated_splits = {}
    after_merge = b''.join(merge)
    for pt, num in pt_stats.items():
        if after_merge not in b''.join(pt_splits[pt]):
            continue
        updated_splits[pt], change_counter = broadcast_merge(pt_splits[pt], merge, num, counter=change_counter)
    pt_splits.update(updated_splits)
    return pt_splits, change_counter

def safe_counter_add(d1: Counter, d2: Counter):
    for k, v in d2.items():
        d1[k] += v
        # if v > 0:
        #     d1[k] += v
        # elif k in d1:
        #     d1[k] += v
    return

class BytePairEncoder:
    def __init__(self):
        return
    
    def init_train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        pt_chunk_size=int(1e6),
        pt_num_chunks=None,
        pt_path=None,
        num_processes=16,
        special_split_token: str='<|endoftext|>',
    ):
        if special_split_token not in special_tokens:
            special_tokens.append(special_split_token)
        self.special_strings = special_tokens
        self.special_split_token = special_split_token

        special_tokens = [st.encode('utf-8') for st in special_tokens]
        special_tokens = sorted(set(special_tokens))

        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pt_chunk_size = pt_chunk_size
        self.pt_num_chunks = pt_num_chunks
        self.pt_path = pt_path
        self.num_processes = num_processes
        self.covered_len = 0
        self.trained_vocab = {}
        self.trained_rocab = {}
        self.trained_merges = []
        return
    
    def train(self):
        pt_stats = self.get_pt_stats()
        substats_splits = [df.num.to_dict() for _, df in pt_stats.groupby('procgroup')]
        pt_stats = pt_stats.num.to_dict()
        n_proc = len(substats_splits)
        procgroup_splits = [
            {
                pt: [bytes([c]) for c in list(pt.encode('utf-8'))]
                for pt in sss
            }
            for sss in substats_splits
        ]
        all_splits = {k: v for pgs in procgroup_splits for k, v in pgs.items()}

        vocab = {}
        rocab = {}
        merges = []
        base = len(vocab)
        for i in range(256):
            t = bytes([i])
            vocab[i] = t
            rocab[t] = i
        base = len(vocab)

        for i, st in enumerate(self.special_tokens):
            if st in rocab:
                continue
            vocab[base] = st
            rocab[st] = base
            base += 1
        base = len(vocab)
        
        num_remain = self.vocab_size - base
        
        sptc = count_subpts(pt_stats, 2, 1)
        # sort on count first descendingly, then on token-candidate lexigraphy ascendingly
        candidates = sorted(sptc.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)[:num_remain]

        while base < self.vocab_size and candidates and candidates[0][1] > 0:
            new_merge, _ = candidates[0]
            _ = sptc.pop(new_merge)

            merges.append(new_merge)
            new_token = new_merge[0] + new_merge[1]
            vocab[base] = new_token
            rocab[new_token] = base
            base += 1
            num_remain -= 1

            merge_call = partial(pt_batch_merge, merge=new_merge)
            # If I switch on multiprocessing, the performance degrades rapidly!
            # if n_proc > 1:
            #     with mp.Pool(n_proc) as pool:
            #         results = pool.starmap(merge_call, zip(substats_splits, procgroup_splits))
            # else:
            #     results = [merge_call(substats_splits[0], procgroup_splits[0])]
            results = [merge_call(pt_stats, all_splits)]
            procgroup_splits = [r[0] for r in results]
            for r in results:
                safe_counter_add(sptc, r[1])

            # result = pt_batch_merge(pt_stats, all_splits, new_merge)
            # safe_counter_add(sptc, result[1])
            # all_splits = result[0]

            candidates = sorted(sptc.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)[:num_remain]
            
            if base % 1000 == 0:
                print(f'Onto token number {base}')
        self.sptc = sptc
        self.candidates = candidates
        self.trained_vocab = vocab
        self.trained_rocab = rocab
        self.trained_merges = merges
        return

    
    def get_pt_stats(self):
        input_path = self.input_path
        vocab_size = self.vocab_size
        special_tokens = self.special_tokens
        pt_chunk_size = self.pt_chunk_size
        pt_num_chunks = self.pt_num_chunks
        pt_path = self.pt_path
        num_processes = self.num_processes
        special_split_token = self.special_split_token

        if pt_path is not None:
            print('Loading dumped pre-token counts!')
            with open(pt_path, 'r') as pf:
                pt_counts = json.load(pf)
        else:
            pt_counts = get_pretoken_counts(
                input_path=input_path, 
                desired_chunk_size=pt_chunk_size,
                num_chunks=pt_num_chunks,
                num_processes=num_processes,
                special_tokens=special_tokens,
                special_split_token=special_split_token,
            )
        
        # pre-processing
        pt_stats = pd.DataFrame({
            'charlen': [len(pt.encode('utf-8')) for pt in pt_counts],
            'num': [pt_counts[pt] for pt in pt_counts]
        }, index=pt_counts.keys())
        pt_stats = pt_stats.loc[[pt for pt in pt_stats.index if pt not in self.special_strings]].copy()
        
        np.random.seed(sum(pt_counts.values()))
        ishuffle = np.arange(pt_stats.shape[0]).astype(int)
        np.random.shuffle(ishuffle)

        pt_stats = pt_stats.iloc[ishuffle].copy()
        cum_charlen = pt_stats.charlen.cumsum()
        splitlen = pt_stats.charlen.sum() // num_processes
        cum_splitlen = np.arange(0, num_processes * splitlen, splitlen)
        split_indices = np.searchsorted(cum_charlen.values, cum_splitlen)
        pt_procgroups = np.zeros(pt_stats.shape[0])
        for si in split_indices[1:]:
            pt_procgroups[si:] += 1
        pt_stats['procgroup'] = pt_procgroups.astype(int)
        return pt_stats


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    pt_chunk_size=int(1e6),
    pt_num_chunks=None,
    pt_path=None,
    num_processes=16,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bpencoder = BytePairEncoder()
    bpencoder.init_train(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        pt_chunk_size=pt_chunk_size,
        pt_num_chunks=pt_num_chunks,
        pt_path=pt_path,
        num_processes=num_processes,
        special_split_token='<|endoftext|>',
    )
    bpencoder.train()
    return (bpencoder.trained_vocab, bpencoder.trained_merges)

if __name__ == '__main__':
    test_path = '../tests/fixtures/corpus.en'
    bpeobj = BytePairEncoder()
    bpeobj.init_train(
        # input_path=valid_path,
        input_path=test_path,
        vocab_size=300,
        special_tokens=['<|endoftext|>'],
        pt_chunk_size=int(1e6),
        # pt_path='data/pt_counts.json',
        num_processes=1,
    )
    bpeobj.train()