from typing import List, Tuple
import numpy as np


def load_vocab(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_nyt_counts(data_path: str) -> List[List[Tuple[int, int]]]:
    """Load NYT data: 'idx:count,idx:count,...' -> [(idx, count), ...]."""
    docs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs = [
                (int(k), int(v)) for k, v in (kv.split(":") for kv in line.split(","))
            ]
            docs.append(pairs)
    return docs


def expand_counts_to_tokens(pairs: List[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.empty(0, dtype=np.int32)
    vs = [np.full(c, v, dtype=np.int32) for v, c in pairs if c > 0]
    return np.concatenate(vs) if vs else np.empty(0, dtype=np.int32)


def sample_document_tokens(
    tokens: np.ndarray, N: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample N tokens from document."""
    L = tokens.shape[0]
    if L == N:
        return tokens
    if L < N:
        return tokens[rng.integers(0, L, size=N)]
    return tokens[rng.choice(L, size=N, replace=False)]


def build_corpus_tokens(
    docs_counts: List[List[Tuple[int, int]]],
    N: int = 200,
    min_len: int = 150,
    seed: int = 0,
) -> np.ndarray:
    """Build corpus matrix [M, N] by sampling N tokens from each document."""
    rng = np.random.default_rng(seed)
    sampled = []
    for pairs in docs_counts:
        raw = expand_counts_to_tokens(pairs)
        if raw.shape[0] >= min_len:
            sampled.append(sample_document_tokens(raw, N, rng))
    if not sampled:
        raise ValueError("No documents left.")
    return np.stack(sampled, axis=0).astype(np.int32)  # [M, N]
