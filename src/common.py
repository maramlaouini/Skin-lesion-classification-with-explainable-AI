import random, numpy as np, torch

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
FULL_NAMES = {
    "akiec": "Kératose actinique / maladie de Bowen",
    "bcc": "Carcinome basocellulaire",
    "bkl": "Kératose bénigne",
    "df": "Dermatofibrome",
    "mel": "Mélanome",
    "nv": "Naevus mélanocytaire",
    "vasc": "Lésion vasculaire",
}
# regroupement binaire "malin / pré-malin" vs bénin (akiec est pré-malin : à documenter)
MALIGNANT = ["akiec", "bcc", "mel"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
K = len(CLASSES)


def img_size(model_name: str) -> int:
    return 300 if model_name == "efficientnet_b3" else 224


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
