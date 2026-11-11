"""Explicabilité : Grad-CAM / Grad-CAM++ / ScoreCAM + évaluation quantitative.
 - deletion / insertion (AUC)
 - saillance hors lésion + pointing game (masque Otsu heuristique ; préférez les masques ISIC 2018 Task 1)
 - test de sanité (randomisation en cascade des couches, corrélation de Spearman)
python -m src.explain --run runs/efficientnet_b0_ce_s0 --method gradcam --n 140"""
import argparse, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.stats import spearmanr
from skimage import filters, measure, morphology
import torchvision.transforms.functional as TF
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.common import CLASSES, MEAN, STD, device, seed_everything
from src.data.dataset import SkinDataset, get_transforms
from src.models import load_run, target_layer

METHODS = {"gradcam": GradCAM, "gradcampp": GradCAMPlusPlus, "scorecam": ScoreCAM}


def denorm(x):
    m, s = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def compute_cam(model, name, method, x, cls):
    with METHODS[method](model=model, target_layers=[target_layer(model, name)]) as cam:
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]


def lesion_mask(rgb):
    g = rgb.mean(2); m = g < filters.threshold_otsu(g)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, structure=morphology.disk(3)))
    lab = measure.label(m)
    if lab.max() == 0: return m
    H, W = m.shape; c = np.bincount(lab[H // 4:3 * H // 4, W // 4:3 * W // 4].ravel(), minlength=lab.max() + 1)
    c[0] = 0
    return lab == c.argmax() if c.max() > 0 else m


@torch.no_grad()
def deletion_insertion(model, x, heat, cls, steps=20):
    H, W = heat.shape; n = H * W; order = torch.from_numpy(np.argsort(-heat.ravel()).copy()).to(x.device)
    blur = TF.gaussian_blur(x, 51, 20.0); dels, inss = [], []
    for s in range(steps + 1):
        m = torch.zeros(n, device=x.device); m[order[:int(n * s / steps)]] = 1; m = m.view(1, 1, H, W)
        dels.append(x * (1 - m) + blur * m); inss.append(blur * (1 - m) + x * m)
    auc = lambda p: float(((p[:-1] + p[1:]) / 2).sum() / steps)
    pd_ = F.softmax(model(torch.cat(dels)), 1)[:, cls].cpu().numpy()
    pi_ = F.softmax(model(torch.cat(inss)), 1)[:, cls].cpu().numpy()
    return auc(pd_), auc(pi_)          # deletion : plus BAS = mieux ; insertion : plus HAUT = mieux


def sanity_check(model, name, method, x, cls, ref_cam):
    """Randomise les couches de la sortie vers l'entrée ; la carte doit se dégrader."""
    m = copy.deepcopy(model); rows = []
    for lname, child in reversed(list(m.named_children())):
        for mod in child.modules():
            if hasattr(mod, "reset_parameters"): mod.reset_parameters()
        cam = compute_cam(m.eval(), name, method, x, cls)
        rows.append((lname, float(spearmanr(ref_cam.ravel(), cam.ravel())[0])))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--method", default="gradcam", choices=list(METHODS))
    ap.add_argument("--n", type=int, default=140); ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    seed_everything(a.seed); dev = device(); model, cfg = load_run(a.run, dev)
    name, size = cfg["model"], cfg["size"]
    te = pd.read_csv(a.splits).query("split == 'test'")
    te = te.sample(frac=1, random_state=a.seed).groupby("label").head(max(1, a.n // 7))
    ds = SkinDataset(te, get_transforms(False, size)); out = Path(a.run) / f"explain_{a.method}"; out.mkdir(exist_ok=True)
    rows, shown = [], {"ok": [], "err": []}
    for i in range(len(ds)):
        x, y = ds[i]; x = x[None].to(dev)
        with torch.no_grad(): pred = int(model(x).argmax(1))
        heat = compute_cam(model, name, a.method, x, pred)
        rgb = denorm(x[0]); mask = lesion_mask(rgb); h = heat / (heat.sum() + 1e-12)
        d_auc, i_auc = deletion_insertion(model, x, heat, pred)
        rows.append({"label": CLASSES[y], "pred": CLASSES[pred], "correct": pred == y,
                     "deletion_auc": d_auc, "insertion_auc": i_auc,
                     "outside_lesion_mass": float(h[~mask].sum()), "lesion_area_frac": float(mask.mean()),
                     "pointing_hit": bool(mask[np.unravel_index(heat.argmax(), heat.shape)])})
        k = "ok" if pred == y else "err"
        if len(shown[k]) < 6: shown[k].append((show_cam_on_image(rgb.astype(np.float32), heat, use_rgb=True), mask, y, pred))
    r = pd.DataFrame(rows); r.to_csv(out / "per_sample.csv", index=False)
    summ = r.groupby("correct")[["deletion_auc", "insertion_auc", "outside_lesion_mass", "lesion_area_frac", "pointing_hit"]].mean()
    print(summ.round(3)); summ.to_csv(out / "summary.csv")
    print("à comparer : outside_lesion_mass vs (1 - lesion_area_frac) attendu si la carte était uniforme")

    fig, ax = plt.subplots(4, 6, figsize=(18, 12))
    for row, k in enumerate(["ok", "err"]):
        for j, (ov, mask, y, p) in enumerate(shown[k]):
            ax[2 * row, j].imshow(ov); ax[2 * row, j].set_title(f"vrai={CLASSES[y]} préd={CLASSES[p]}", fontsize=9)
            ax[2 * row + 1, j].imshow(mask, cmap="gray"); ax[2 * row + 1, j].set_title("masque lésion (Otsu)", fontsize=8)
    for axx in ax.ravel(): axx.axis("off")
    plt.tight_layout(); plt.savefig(out / "examples.png", dpi=90); plt.close()

    x, y = ds[0]; x = x[None].to(dev); cls = int(model(x).argmax(1))
    ref = compute_cam(model, name, a.method, x, cls); s = sanity_check(model, name, a.method, x, cls, ref)
    pd.DataFrame(s, columns=["couche_randomisée_cascade", "spearman"]).to_csv(out / "sanity.csv", index=False)
    print("test de sanité (Spearman ; doit chuter vers 0) :", [(n, round(v, 2)) for n, v in s])


if __name__ == "__main__":
    main()
