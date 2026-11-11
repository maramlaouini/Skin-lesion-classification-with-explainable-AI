import pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from src.common import MEAN, STD


def get_transforms(train: bool, size: int):
    norm = T.Normalize(MEAN, STD)
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(30),
            T.ColorJitter(0.15, 0.15, 0.1, 0.02),   # modéré : la couleur est diagnostique
            T.ToTensor(), norm, T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf, pil_fn=None):
        self.df, self.tf, self.pil_fn = df.reset_index(drop=True), tf, pil_fn

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        if self.pil_fn is not None:
            img = self.pil_fn(img)           # perturbation (robustesse)
        return self.tf(img), int(r.label)


def make_loader(df, train, size, bs=32, sampler=None, workers=4, pil_fn=None):
    ds = SkinDataset(df, get_transforms(train, size), pil_fn)
    return DataLoader(ds, batch_size=bs, shuffle=(train and sampler is None), sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=train)
