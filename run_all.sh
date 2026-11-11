#!/bin/bash
# Pipeline complet (~ plusieurs heures sur un GPU T4/P100). Ajustez --epochs si besoin.
set -e
python -m src.data.prepare_splits --raw data/raw --out data/splits.csv
python -m src.data.eda
for seed in 0 1 2; do
  for m in baseline resnet50 efficientnet_b0; do
    python -m src.train --model $m --loss ce --seed $seed
  done
  for loss in wce sampler focal ls; do                      # ablation déséquilibre
    python -m src.train --model efficientnet_b0 --loss $loss --seed $seed
  done
done
for r in runs/*/; do python -m src.evaluate --run "$r"; done
python -m src.aggregate
BEST=$(ls -d runs/efficientnet_b0_wce_s0)                   # à remplacer par votre meilleur run
for meth in gradcam gradcampp scorecam; do python -m src.explain --run $BEST --method $meth; done
python -m src.robustness perturb --run $BEST
python -m src.robustness subgroups --run $BEST
# python -m src.robustness external --run $BEST --ext_csv data/external.csv
