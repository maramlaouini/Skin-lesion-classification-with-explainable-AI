"""Construit data/splits.csv : doublons exacts, quasi-doublons, groupes, split train/val/test sans fuite.
Usage : python -m src.data.prepare_splits --raw data/raw --out data/splits.csv
"""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from src.common import CLASS2IDX
from src.data.dedup import md5, phash_int, near_duplicate_pairs, UnionFind

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--out", default="data/splits.csv")
ap.add_argument("--ham_thr", type=int, default=4, help="distance de Hamming max (phash 64 bits)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = pd.read_csv(glob.glob(f"{a.raw}/**/HAM10000_metadata.csv", recursive=True)[0])
paths = {os.path.splitext(os.path.basename(p))[0]: p
         for p in glob.glob(f"{a.raw}/**/*.jpg", recursive=True)}
df["path"] = df.image_id.map(paths)
assert df.path.notna().all(), "images manquantes"
df["label"] = df.dx.map(CLASS2IDX)

df["md5"] = [md5(p) for p in tqdm(df.path, desc="md5")]
df["phash"] = [phash_int(p) for p in tqdm(df.path, desc="phash")]
print("doublons exacts (md5) :", df.md5.duplicated().sum())

# groupes : lesion_id + quasi-doublons (union-find)
uf = UnionFind(len(df))
for _, idx in df.groupby("lesion_id").indices.items():
    for j in idx[1:]: uf.union(idx[0], j)
pairs = near_duplicate_pairs(df.phash.values.astype(np.uint64), a.ham_thr)
for i, j in pairs: uf.union(i, j)
df["group"] = [uf.find(i) for i in range(len(df))]
print(f"lesion_id uniques : {df.lesion_id.nunique()} | groupes finaux : {df.group.nunique()} "
      f"| paires quasi-doublons : {len(pairs)}")
mixed = df.groupby("group").dx.nunique()
print("groupes à labels mixtes :", int((mixed > 1).sum()))

# 7 folds -> 1 test (~15%), 1 val (~15%), 5 train
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=a.seed)
folds = [te for _, te in sgkf.split(df, df.label, df.group)]
df["split"] = "train"
df.loc[folds[0], "split"] = "test"
df.loc[folds[1], "split"] = "val"

g = {s: set(df.group[df.split == s]) for s in ["train", "val", "test"]}
assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"])
print(df.groupby(["split", "dx"]).size().unstack(0))
df.to_csv(a.out, index=False)
print("sauvegardé :", a.out)
