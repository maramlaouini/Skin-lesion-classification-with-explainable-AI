"""EDA : classes, métadonnées, grille d'images, test de biais de couleur.  python -m src.data.eda"""
import argparse, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from src.common import CLASSES

ap = argparse.ArgumentParser(); ap.add_argument("--splits", default="data/splits.csv")
ap.add_argument("--out", default="reports/eda"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.splits)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
df.dx.value_counts().plot.bar(ax=ax[0, 0], title="Classes")
df.dx_type.value_counts().plot.bar(ax=ax[0, 1], title="Vérité terrain (dx_type)")
df.sex.value_counts().plot.bar(ax=ax[0, 2], title="Sexe")
df.age.plot.hist(bins=20, ax=ax[1, 0], title="Âge")
df.localization.value_counts().plot.barh(ax=ax[1, 1], title="Localisation")
df.groupby("lesion_id").size().value_counts().sort_index().plot.bar(ax=ax[1, 2], title="Images par lesion_id")
plt.tight_layout(); plt.savefig(f"{a.out}/distributions.png", dpi=120); plt.close()
pd.crosstab(df.dx, df.dx_type, normalize="index").round(3).to_csv(f"{a.out}/dx_vs_dxtype.csv")

fig, ax = plt.subplots(len(CLASSES), 8, figsize=(16, 14))
for i, c in enumerate(CLASSES):
    for j, p in enumerate(df[df.dx == c].path.sample(8, random_state=0)):
        ax[i, j].imshow(Image.open(p).resize((150, 112))); ax[i, j].axis("off")
    ax[i, 0].set_title(c, loc="left")
plt.tight_layout(); plt.savefig(f"{a.out}/grid.png", dpi=90); plt.close()

# biais d'acquisition : des statistiques de couleur seules (dont les bords) prédisent-elles la classe ?
sub = df.groupby("dx").sample(min(300, df.dx.value_counts().min()), random_state=0)
feats = []
for p in sub.path:
    x = np.asarray(Image.open(p).convert("RGB").resize((64, 48)), dtype=np.float32) / 255
    feats.append(np.r_[x.mean((0, 1)), x.std((0, 1)), x[:8].mean((0, 1)), x[-8:].mean((0, 1))])
pred = cross_val_predict(LogisticRegression(max_iter=2000), np.array(feats), sub.dx,
                         groups=sub.group, cv=GroupKFold(5))
print("Balanced acc. (stats couleur seules, hasard = 0.143) :",
      round(balanced_accuracy_score(sub.dx, pred), 3))
vc = df.dx.value_counts(); print(vc, "\nratio max/min :", round(vc.max() / vc.min(), 1))
