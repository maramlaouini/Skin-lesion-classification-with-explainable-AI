import json, timm, torch, torch.nn as nn
from pathlib import Path
from src.common import K


class SimpleCNN(nn.Module):
    """Baseline from scratch : 4 blocs (2 conv-BN-ReLU + maxpool) + GAP + FC."""
    def __init__(self, n_classes=K, widths=(32, 64, 128, 256)):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                       nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(c, n_classes))

    def forward(self, x): return self.classifier(self.features(x))


def build_model(name: str, n_classes=K, pretrained=True):
    if name == "baseline":
        return SimpleCNN(n_classes)
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes, drop_rate=0.2)


def head_params(model):
    return list((model.classifier if isinstance(model, SimpleCNN) else model.get_classifier()).parameters())


def target_layer(model, name: str):
    """Dernière couche convolutive pour Grad-CAM."""
    if name == "baseline": return model.features[-2]          # ReLU avant le dernier maxpool
    if name.startswith("resnet"): return model.layer4[-1]
    if name.startswith("efficientnet"): return model.blocks[-1][-1]
    if name.startswith("densenet"): return model.features.norm5
    if name.startswith("convnext"): return model.stages[-1]
    raise ValueError(f"target_layer inconnu pour {name}")


def load_run(run_dir, dev):
    run = Path(run_dir)
    cfg = json.load(open(run / "config.json"))
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    return model.to(dev).eval(), cfg
