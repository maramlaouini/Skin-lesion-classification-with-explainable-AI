# Classification de lésions cutanées par apprentissage profond et IA explicable

> ⚠️ **Prototype de recherche. Ce projet n'est pas un dispositif médical et ne doit pas être utilisé pour établir un diagnostic clinique.**

Ce dépôt propose un pipeline reproductible pour classer des images dermatoscopiques de lésions cutanées (jeu de données **HAM10000**) et étudier, au-delà de la performance brute, **l'explicabilité, la calibration, la robustesse et les biais** des prédictions.

## Problématique

Un classifieur de lésions cutanées peut afficher une bonne *accuracy* tout en étant peu fiable : les classes sont très déséquilibrées, plusieurs images d'une même lésion peuvent fuiter entre entraînement et test, les probabilités sont souvent mal calibrées, et les cartes d'attention peuvent révéler des raccourcis (règles, bords, marqueurs) plutôt que la lésion elle-même. Le projet cherche donc à répondre à :

1. Comment évaluer honnêtement un modèle sur des classes déséquilibrées, sans fuite de données ?
2. Les architectures par transfert d'apprentissage (ResNet, EfficientNet) apportent-elles un gain réel face à un CNN de référence ?
3. Les probabilités prédites sont-elles fiables, et peut-on les calibrer ?
4. Sur quelles régions de l'image le modèle s'appuie-t-il, et ces explications sont-elles fiables ?
5. Comment le modèle se comporte-t-il face aux perturbations, aux sous-groupes de patients et aux données d'une autre source ?

## Données

**HAM10000** : 10 015 images dermatoscopiques, 7 classes.

| Code | Signification | Images |
|---|---|---|
| nv | Naevus mélanocytaire | 6 705 |
| mel | Mélanome | 1 113 |
| bkl | Kératose bénigne | 1 099 |
| bcc | Carcinome basocellulaire | 514 |
| akiec | Kératose actinique / maladie de Bowen | 327 |
| vasc | Lésion vasculaire | 142 |
| df | Dermatofibrome | 115 |

Le déséquilibre est d'environ **58:1** entre la classe majoritaire (nv) et la plus rare (df). Les images proviennent de deux centres (Vienne, Queensland). La vérité terrain combine histopathologie, suivi, consensus d'experts et microscopie confocale (champ `dx_type`).

Source : [Kaggle – HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), [ISIC Archive](https://www.isic-archive.com/).

## Méthodologie

### 1. Doublons et séparation sans fuite
HAM10000 ne fournit pas d'identifiant patient, mais un `lesion_id` : une même lésion peut avoir plusieurs images. La séparation est donc faite **par groupes** :
- détection des doublons exacts (MD5) et des quasi-doublons (hash perceptuel, distance de Hamming ≤ 4) ;
- fusion des images d'une même lésion et des quasi-doublons en groupes (union-find) ;
- découpage stratifié par groupe (`StratifiedGroupKFold`) : ~70 % entraînement, 15 % validation, 15 % test, avec vérification automatique qu'aucun groupe n'est partagé entre ensembles.

Mesures obtenues sur HAM10000 : 7 470 lésions uniques, 7 283 groupes finaux, 2 doublons exacts, 425 paires de quasi-doublons ; ensembles de 7 153 images (entraînement), 1 431 (validation) et 1 431 (test).
*Limite :* le groupement par lésion est un proxy du groupement par patient, qui reste impossible sans identifiant patient. 15 groupes contiennent des labels différents (fusion par quasi-doublon ou incohérence des données) et sont signalés.

### 2. Modèles
- **CNN de référence** entraîné de zéro (4 blocs conv-BN-ReLU, global average pooling).
- **Transfert d'apprentissage** : ResNet50 et EfficientNet-B0/B3 (poids ImageNet, via `timm`), avec fine-tuning en deux phases (tête seule puis réseau complet). DenseNet121 et ConvNeXt-T sont aussi disponibles.
- AdamW, ordonnancement cosinus, arrêt anticipé sur le **F1 macro de validation**.

### 3. Gestion du déséquilibre
Comparaison, à architecture fixée, de : entropie croisée standard, entropie croisée pondérée (« effective number of samples »), sur-échantillonnage (`WeightedRandomSampler`), *focal loss* et *label smoothing*.

### 4. Évaluation
Calculée sur le test uniquement, avec les hyperparamètres choisis sur la validation :
- **F1 macro** (métrique principale), accuracy équilibrée ;
- **sensibilité et spécificité par classe** ;
- **AUROC** un-contre-tous par classe et macro, plus une tâche binaire malin/pré-malin vs bénin (`akiec`, `bcc`, `mel` contre le reste) ;
- **matrice de confusion** normalisée ;
- **calibration** : ECE, score de Brier, diagramme de fiabilité, *temperature scaling* ajusté sur la validation ;
- **intervalles de confiance à 95 %** par bootstrap au niveau des groupes.

### 5. Explicabilité
- Cartes **Grad-CAM, Grad-CAM++ et ScoreCAM** (`pytorch-grad-cam`).
- Évaluation **quantitative** des cartes : courbes *deletion/insertion*, part de la saillance **hors de la lésion**, *pointing game*, et **test de sanité** (randomisation en cascade des couches, corrélation de Spearman).
- Le masque de lésion est obtenu par seuillage d'Otsu (heuristique) ; les masques ISIC 2018 Task 1 sont recommandés pour une évaluation rigoureuse.

### 6. Robustesse et biais
- Perturbations synthétiques : flou, bruit, compression JPEG, luminosité, contraste, occlusion.
- Performance par **sous-groupes** : sexe, tranche d'âge, localisation, type de vérité terrain.
- **Test externe** (ISIC 2019, BCN20000) avec retrait automatique des images déjà présentes dans HAM10000 (hash perceptuel).
- Analyse exploratoire du biais d'acquisition : un classifieur utilisant uniquement des statistiques de couleur (dont les bords) atteint une accuracy équilibrée de **0,34** (hasard : 0,14), ce qui indique un signal lié à l'acquisition indépendant de la lésion.

### 7. Prototype
Application **Gradio** : image dermatoscopique en entrée ; classe la plus probable, probabilités calibrées pour les 7 classes, carte Grad-CAM superposée, avertissement de faible confiance, et **avertissement non-diagnostic affiché en permanence**.

## Structure du dépôt

```
skin-xai/
├── src/
│   ├── common.py            # classes, constantes, utilitaires
│   ├── models.py            # CNN de référence, modèles timm, couches cibles Grad-CAM
│   ├── train.py             # entraînement (modèles, pertes, seeds)
│   ├── evaluate.py          # métriques, calibration, bootstrap
│   ├── explain.py           # Grad-CAM + évaluation quantitative
│   ├── robustness.py        # perturbations, sous-groupes, test externe
│   ├── aggregate.py         # tableaux moyenne ± écart-type sur les seeds
│   └── data/                # dedup.py, prepare_splits.py, dataset.py, eda.py
├── app/app.py               # prototype Gradio
├── notebooks/               # notebook Kaggle/Colab
├── run_all.sh               # pipeline complet
└── requirements.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate    # Windows : .venv\Scripts\activate
pip install -r requirements.txt
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip
```
`data/raw` doit contenir `HAM10000_metadata.csv` et les images `.jpg` (les sous-dossiers sont acceptés). Un GPU est recommandé (Kaggle ou Colab suffisent).

## Utilisation

Toutes les commandes se lancent depuis la racine du dépôt.

```bash
# 1. Splits sans fuite + analyse exploratoire
python -m src.data.prepare_splits --raw data/raw --out data/splits.csv
python -m src.data.eda

# 2. Entraînement (modèles : baseline, resnet50, efficientnet_b0, efficientnet_b3 ;
#    pertes : ce, wce, sampler, focal, ls)
python -m src.train --model efficientnet_b0 --loss wce --seed 0

# 3. Évaluation sur le test
python -m src.evaluate --run runs/efficientnet_b0_wce_s0

# 4. Explicabilité
python -m src.explain --run runs/efficientnet_b0_wce_s0 --method gradcam

# 5. Robustesse
python -m src.robustness perturb   --run runs/efficientnet_b0_wce_s0
python -m src.robustness subgroups --run runs/efficientnet_b0_wce_s0

# 6. Prototype
RUN_DIR=runs/efficientnet_b0_wce_s0 python app/app.py
```
`./run_all.sh` enchaîne l'ensemble (3 seeds, plusieurs modèles et stratégies) et `python -m src.aggregate` produit le tableau récapitulatif dans `reports/results.md`.

Pour un test rapide sur un sous-échantillon, créez un `data/splits_mini.csv` et ajoutez `--splits data/splits_mini.csv --epochs 2 --workers 0` aux commandes.

## Résultats

*(à compléter avec les résultats des expériences)*

| Modèle | Perte | F1 macro | Sens. mélanome | AUROC macro | ECE (avant → après TS) |
|---|---|---|---|---|---|
| … | … | … | … | … | … |

Figures à ajouter : matrice de confusion, diagramme de fiabilité, exemples Grad-CAM, courbes de robustesse.

## Limites et biais

- Population dominée par des **peaux claires** ; peu d'information sur les phototypes foncés.
- **Deux centres** d'acquisition : la généralisation à d'autres dermatoscopes n'est pas garantie.
- Pas d'identifiant patient : risque résiduel de fuite au niveau patient.
- Labels **non tous confirmés par histologie**, surtout pour les naevus.
- Artefacts d'acquisition (règles, marqueurs, vignettage) susceptibles de servir de raccourcis.
- Classes fortement déséquilibrées ; évaluation **rétrospective**, sans validation clinique.
- Les cartes Grad-CAM indiquent où le modèle « regarde », pas pourquoi : elles ne constituent pas une preuve de raisonnement clinique.

## Références

- Tschandl et al. (2018). *The HAM10000 dataset.* Scientific Data.
- Selvaraju et al. (2017). *Grad-CAM.* ICCV.
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Adebayo et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS.
- Bissoto et al. (2019). *Constructing Normative Models…* / biais et artefacts dans les datasets de lésions cutanées (à vérifier avant citation).
- Daneshjou et al. (2021–2022). Biais en IA dermatologique.
- Jang & Park (2024). *Development and Validation of Adaptable Skin Cancer Classification System Using Dynamically Expandable Representation.* Healthcare Informatics Research.
- Miętkiewicz et al. (2025). *The Skin Game: Revolutionizing Standards for AI Dermatology Model Comparison.* arXiv:2502.02500.

## Avertissement

Ce dépôt est fourni à des fins de **recherche et d'enseignement**. Il n'est pas un dispositif médical, n'a fait l'objet d'aucune validation clinique et ne doit pas être utilisé pour poser un diagnostic ou orienter une décision de soin. Pour toute lésion cutanée, consultez un professionnel de santé.
