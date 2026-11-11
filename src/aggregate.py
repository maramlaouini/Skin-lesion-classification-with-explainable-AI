"""Agrège runs/*/metrics.json -> reports/results.md (moyenne ± écart-type sur les seeds)."""
import json, glob, pandas as pd
rows = []
for f in glob.glob("runs/*/metrics.json"):
    m = json.load(open(f)); c = m["config"]
    rows.append({"model": c["model"], "loss": c["loss"], "seed": c["seed"], "F1 macro": m["f1_macro"],
                 "Bal. acc": m["balanced_acc"], "AUROC macro": m["auroc_macro"], "ECE (avant)": m["ece_before"],
                 "ECE (après TS)": m["ece_after"], "Sens. mel": m["mel_sens"],
                 "Sens. akiec": m["raw"]["sensitivity"]["akiec"], "Sens. bcc": m["raw"]["sensitivity"]["bcc"],
                 "Sens. df": m["raw"]["sensitivity"]["df"], "Sens. vasc": m["raw"]["sensitivity"]["vasc"]})
df = pd.DataFrame(rows); g = df.drop(columns="seed").groupby(["model", "loss"])
mu, sd, n = g.mean(), g.std(), g.size()
out = mu.round(3).astype(str) + " ± " + sd.fillna(0).round(3).astype(str); out["n_seeds"] = n
open("reports/results.md", "w").write(out.to_markdown()); print(out.to_markdown())
