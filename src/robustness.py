"""Robustesse : perturbations synthétiques, sous-groupes, test externe (avec retrait des doublons HAM10000).
python -m src.robustness perturb  --run RUN
python -m src.robustness subgroups --run RUN
python -m src.robustness external --run RUN --ext_csv ext.csv   (colonnes : path, dx avec codes HAM)"""
import argparse, io, json
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import f1_score, balanced_accuracy_score
from src.common import CLASSES, CLASS2IDX, K, device, softmax
from src.data.dataset import make_loader
from src.data.dedup import phash_int, min_dist_to_reference
from src.evaluate import predict_logits
from src.models import load_run


def _noise(s):
    def f(img):
        x = np.asarray(img, np.float32); rng = np.random.default_rng(0)
        return Image.fromarray(np.clip(x + rng.normal(0, s, x.shape), 0, 255).astype(np.uint8))
    return f

def _jpeg(q):
    def f(img):
        b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    return f

def _occ(frac):
    def f(img):
        img = img.copy(); w, h = img.size; s = int(frac * min(w, h)); cx, cy = w // 2, h // 2
        img.paste((0, 0, 0), (cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2)); return img
    return f

PERTURB = {
    "blur": [(r, (lambda r: lambda im: im.filter(ImageFilter.GaussianBlur(r)))(r)) for r in [0, 1, 2, 4, 8]],
    "noise": [(s, _noise(s)) for s in [0, 5, 10, 20, 40]],
    "jpeg_quality": [(q, _jpeg(q)) for q in [95, 50, 20, 10, 5]],
    "brightness": [(b, (lambda b: lambda im: ImageEnhance.Brightness(im).enhance(b))(b)) for b in [1.0, 0.7, 0.5, 1.5, 2.0]],
    "contrast": [(c, (lambda c: lambda im: ImageEnhance.Contrast(im).enhance(c))(c)) for c in [1.0, 0.7, 0.5, 1.5, 2.0]],
    "occlusion": [(o, _occ(o)) for o in [0, 0.1, 0.2, 0.3, 0.4]],
}


def metrics_quick(logits, y, T):
    p = softmax(logits / T).argmax(1); m = y == CLASS2IDX["mel"]
    return {"f1_macro": f1_score(y, p, average="macro", labels=range(K), zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, p), "mel_sens": float((p[m] == CLASS2IDX["mel"]).mean()) if m.any() else np.nan}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["perturb", "subgroups", "external"])
    ap.add_argument("--run", required=True); ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--ext_csv"); ap.add_argument("--ham_thr", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    dev = device(); model, cfg = load_run(a.run, dev); run = Path(a.run)
    T = json.load(open(run / "temperature.json"))["T"]
    df = pd.read_csv(a.splits); te = df[df.split == "test"].reset_index(drop=True); y = te.label.values

    if a.mode == "perturb":
        rows = []
        for pname, levels in PERTURB.items():
            for lvl, fn in levels:
                lg = predict_logits(model, make_loader(te, False, cfg["size"], 64, workers=a.workers, pil_fn=fn), dev)
                rows.append({"perturbation": pname, "level": lvl, **metrics_quick(lg, y, T)}); print(rows[-1])
        r = pd.DataFrame(rows); r.to_csv(run / "robustness_perturb.csv", index=False)
        fig, ax = plt.subplots(2, 3, figsize=(14, 7))
        for axx, (pn, g) in zip(ax.ravel(), r.groupby("perturbation", sort=False)):
            g = g.sort_values("level"); axx.plot(g.level.astype(str), g.f1_macro, "o-", label="F1 macro")
            axx.plot(g.level.astype(str), g.mel_sens, "s--", label="Sens. mel"); axx.set_title(pn); axx.set_ylim(0, 1)
        ax[0, 0].legend(); plt.tight_layout(); plt.savefig(run / "robustness_perturb.png", dpi=120)

    elif a.mode == "subgroups":
        lg = np.load(run / "test_logits.npy"); pred = softmax(lg / T).argmax(1); te["pred"] = pred
        te["age_bin"] = pd.cut(te.age, [0, 30, 45, 60, 75, 120]).astype(str); rows = []
        for col in ["sex", "age_bin", "localization", "dx_type"]:
            for v, g in te.groupby(col):
                if len(g) < 30: continue
                rows.append({"attribut": col, "valeur": v, "n": len(g),
                             "F1 macro": f1_score(g.label, g.pred, average="macro", labels=range(K), zero_division=0),
                             "accuracy": (g.label == g.pred).mean()})
        r = pd.DataFrame(rows); r.to_csv(run / "subgroups.csv", index=False); print(r.round(3).to_string())

    else:
        ext = pd.read_csv(a.ext_csv); ext["label"] = ext.dx.map(CLASS2IDX); ext = ext.dropna(subset=["label"]).copy()
        ext["label"] = ext.label.astype(int)
        h_ext = np.array([phash_int(p) for p in ext.path], dtype=np.uint64)
        h_ref = df.phash.values.astype(np.uint64)
        d = min_dist_to_reference(h_ext, h_ref); keep = d > a.ham_thr
        print(f"images externes : {len(ext)} | retirées (présentes dans HAM10000) : {(~keep).sum()}")
        ext = ext[keep].reset_index(drop=True)
        lg = predict_logits(model, make_loader(ext, False, cfg["size"], 64, workers=a.workers), dev)
        res = metrics_quick(lg, ext.label.values, T); pred = softmax(lg / T).argmax(1)
        res["sensitivity"] = {c: float((pred[ext.label.values == k] == k).mean())
                              for k, c in enumerate(CLASSES) if (ext.label.values == k).any()}
        json.dump(res, open(run / "external.json", "w"), indent=1); print(res)


if __name__ == "__main__":
    main()
