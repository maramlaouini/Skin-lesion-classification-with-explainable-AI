"""Évaluation sur le TEST : F1 macro, sens/spec par classe, AUROC, matrice de confusion, calibration
(ECE, Brier, temperature scaling ajusté sur VAL), IC bootstrap par groupe.
python -m src.evaluate --run runs/efficientnet_b0_ce_s0"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score,
                             accuracy_score)
from src.common import CLASSES, CLASS2IDX, MALIGNANT, K, device, softmax
from src.data.dataset import make_loader
from src.models import load_run


@torch.no_grad()
def predict_logits(model, loader, dev):
    out = [model(x.to(dev)).float().cpu().numpy() for x, _ in loader]
    return np.concatenate(out)


def nll(logits, y, T=1.0):
    p = softmax(logits / T); return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def fit_temperature(logits, y):
    r = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 10), method="bounded")
    return float(r.x)


def ece_score(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(p, y):
    oh = np.eye(K)[y]; return float(((p - oh) ** 2).sum(1).mean())


def sens_spec(cm):
    sens, spec = {}, {}
    for k, c in enumerate(CLASSES):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp; tn = cm.sum() - tp - fn - fp
        sens[c] = float(tp / (tp + fn)) if tp + fn else float("nan")
        spec[c] = float(tn / (tn + fp)) if tn + fp else float("nan")
    return sens, spec


def compute_metrics(logits, y, T=1.0):
    p = softmax(logits / T); pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(K)); sens, spec = sens_spec(cm)
    auc = {c: (roc_auc_score(y == k, p[:, k]) if 0 < (y == k).sum() < len(y) else float("nan"))
           for k, c in enumerate(CLASSES)}
    mal = [CLASS2IDX[c] for c in MALIGNANT]; yb = np.isin(y, mal); sb = p[:, mal].sum(1)
    pb = sb >= 0.5
    return {
        "f1_macro": f1_score(y, pred, average="macro"), "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred), "sensitivity": sens, "specificity": spec, "auroc": auc,
        "auroc_macro": float(np.nanmean(list(auc.values()))),
        "binary_malignant": {"auroc": roc_auc_score(yb, sb), "sensitivity": float((pb & yb).sum() / yb.sum()),
                             "specificity": float((~pb & ~yb).sum() / (~yb).sum())},
        "ece": ece_score(p, y), "brier": brier(p, y), "nll": nll(logits, y, T),
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(logits, y, groups, T, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}; pred = softmax(logits / T).argmax(1)
    res = {"f1_macro": [], "balanced_acc": [], "mel_sensitivity": []}
    for _ in range(n):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
        res["f1_macro"].append(f1_score(y[idx], pred[idx], average="macro"))
        res["balanced_acc"].append(balanced_accuracy_score(y[idx], pred[idx]))
        m = y[idx] == CLASS2IDX["mel"]; res["mel_sensitivity"].append((pred[idx][m] == CLASS2IDX["mel"]).mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}


def plot_cm(cm, path):
    cmn = np.array(cm, float); cmn /= cmn.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(K)); ax.set_yticklabels(CLASSES)
    for i in range(K):
        for j in range(K): ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                                   color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai"); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_reliability(items, y, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--")
    for name, p in items:
        conf, acc = p.max(1), (p.argmax(1) == y); edges = np.linspace(0, 1, n_bins + 1); xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 5: xs.append(conf[m].mean()); ys.append(acc[m].mean())
        ax.plot(xs, ys, "o-", label=f"{name} (ECE={ece_score(p, y):.3f})")
    ax.set_xlabel("Confiance"); ax.set_ylabel("Précision"); ax.legend(); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--splits", default="data/splits.csv"); ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    dev = device(); model, cfg = load_run(a.run, dev); run = Path(a.run)
    df = pd.read_csv(a.splits); va, te = df[df.split == "val"], df[df.split == "test"]
    lv = predict_logits(model, make_loader(va, False, cfg["size"], 64, workers=a.workers), dev)
    lt = predict_logits(model, make_loader(te, False, cfg["size"], 64, workers=a.workers), dev)
    yv, yt = va.label.values, te.label.values
    T = fit_temperature(lv, yv)                      # ajusté sur VAL uniquement
    np.save(run / "test_logits.npy", lt); np.save(run / "val_logits.npy", lv)
    json.dump({"T": T}, open(run / "temperature.json", "w"))
    m_raw, m_cal = compute_metrics(lt, yt, 1.0), compute_metrics(lt, yt, T)
    m_raw["ci95"] = bootstrap_ci(lt, yt, te.group.values, 1.0, a.n_boot)
    json.dump({"T": T, "raw": m_raw, "calibrated": m_cal, "f1_macro": m_raw["f1_macro"],
               "balanced_acc": m_raw["balanced_acc"], "auroc_macro": m_raw["auroc_macro"],
               "ece_before": m_raw["ece"], "ece_after": m_cal["ece"],
               "mel_sens": m_raw["sensitivity"]["mel"], "config": cfg},
              open(run / "metrics.json", "w"), indent=1)
    plot_cm(m_raw["confusion_matrix"], run / "confusion_matrix.png")
    plot_reliability([("avant TS", softmax(lt)), (f"après TS (T={T:.2f})", softmax(lt / T))], yt,
                     run / "reliability.png")
    print(f"F1 macro={m_raw['f1_macro']:.3f} IC95={m_raw['ci95']['f1_macro']} | bal.acc={m_raw['balanced_acc']:.3f} "
          f"| AUROC macro={m_raw['auroc_macro']:.3f} | ECE {m_raw['ece']:.3f} -> {m_cal['ece']:.3f} (T={T:.2f})")
    print("sensibilité :", {k: round(v, 2) for k, v in m_raw["sensitivity"].items()})
    print("binaire malin :", {k: round(v, 3) for k, v in m_raw["binary_malignant"].items()})


if __name__ == "__main__":
    main()
