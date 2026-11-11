"""Entraînement.  python -m src.train --model efficientnet_b0 --loss wce --seed 0
models : baseline | resnet50 | efficientnet_b0 | efficientnet_b3 | densenet121 | convnext_tiny
loss   : ce | wce (poids 'effective number') | sampler | focal | ls (label smoothing)"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from src.common import K, seed_everything, device, img_size
from src.data.dataset import make_loader
from src.models import build_model, head_params


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma, self.weight = gamma, weight
    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1)
        ce = F.nll_loss(logp, y, weight=self.weight, reduction="none")
        pt = logp.gather(1, y[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loss(kind, counts, dev):
    if kind in ("ce", "sampler"): return nn.CrossEntropyLoss()
    if kind == "ls": return nn.CrossEntropyLoss(label_smoothing=0.1)
    beta = 0.999
    w = (1 - beta) / (1 - beta ** counts); w = torch.tensor(w / w.sum() * K, dtype=torch.float32, device=dev)
    if kind == "wce": return nn.CrossEntropyLoss(weight=w)
    if kind == "focal": return FocalLoss(2.0)
    raise ValueError(kind)


@torch.no_grad()
def eval_f1(model, loader, dev):
    model.eval(); P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(dev)).argmax(1).cpu()); Y.append(y)
    return f1_score(torch.cat(Y), torch.cat(P), average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0"); ap.add_argument("--loss", default="ce")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--head_epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    seed_everything(a.seed); dev = device()
    df = pd.read_csv(a.splits)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    size = img_size(a.model); pretrained = a.model != "baseline"
    counts = np.bincount(tr.label, minlength=K)
    sampler = None
    if a.loss == "sampler":
        w = 1.0 / counts[tr.label.values]
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(tr), replacement=True)
    tl = make_loader(tr, True, size, a.bs, sampler, a.workers)
    vl = make_loader(va, False, size, a.bs, None, a.workers)

    model = build_model(a.model, pretrained=pretrained).to(dev)
    crit = make_loss(a.loss, counts, dev)
    hp = {id(p) for p in head_params(model)}
    bb = [p for p in model.parameters() if id(p) not in hp]
    groups = [{"params": head_params(model), "lr": a.lr_head}]
    if bb: groups.append({"params": bb, "lr": a.lr if pretrained else a.lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run = Path(a.out) / f"{a.model}_{a.loss}_s{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    json.dump({**vars(a), "size": size}, open(run / "config.json", "w"), indent=1)
    best, bad, hist = -1, 0, []
    for ep in range(a.epochs):
        freeze = pretrained and ep < a.head_epochs        # phase 1 : tête seule
        for p in bb: p.requires_grad = not freeze
        model.train(); tot = 0; t0 = time.time()
        for x, y in tqdm(tl, desc=f"ep{ep}", leave=False):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); tot += loss.item()
        sched.step()
        f1 = eval_f1(model, vl, dev)
        hist.append({"epoch": ep, "loss": tot / len(tl), "val_f1_macro": f1})
        print(f"ep{ep} loss={tot/len(tl):.4f} val_f1_macro={f1:.4f} ({time.time()-t0:.0f}s)")
        if f1 > best:
            best, bad = f1, 0; torch.save(model.state_dict(), run / "best.pt")
        else:
            bad += 1
            if ep >= a.head_epochs and bad >= a.patience: print("early stopping"); break
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    print("meilleur val F1 macro :", round(best, 4), "->", run)


if __name__ == "__main__":
    main()
