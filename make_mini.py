import pandas as pd

df = pd.read_csv("data/splits.csv")
parts = []
for s, n in [("train", 60), ("val", 20), ("test", 20)]:
    d = df[df.split == s].sample(frac=1, random_state=0)   # mélange
    parts.append(d.groupby("dx").head(n))                   # n images max par classe
mini = pd.concat(parts)
mini.to_csv("data/splits_mini.csv", index=False)
print(mini.groupby(["split", "dx"]).size().unstack(0))