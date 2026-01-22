import os
from fever_loader import FeverDataset

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

train_df = FeverDataset("train").to_dataframe()
val_df = FeverDataset("labelled_dev").to_dataframe()

train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")
