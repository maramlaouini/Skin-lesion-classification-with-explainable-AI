"""Hash perceptuel, distance de Hamming, composantes de quasi-doublons (union-find)."""
import hashlib, numpy as np, imagehash
from PIL import Image
from tqdm import tqdm


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def phash_int(path):
    return int(str(imagehash.phash(Image.open(path).convert("RGB"))), 16)


def popcount64(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def near_duplicate_pairs(hashes: np.ndarray, thr: int = 4):
    pairs = []
    for i in tqdm(range(len(hashes) - 1), desc="near-dups"):
        d = popcount64(hashes[i + 1:] ^ hashes[i])
        for j in np.nonzero(d <= thr)[0]:
            pairs.append((i, i + 1 + int(j)))
    return pairs


def min_dist_to_reference(h_query: np.ndarray, h_ref: np.ndarray) -> np.ndarray:
    return np.array([popcount64(h_ref ^ h).min() for h in h_query])
