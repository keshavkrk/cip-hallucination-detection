from datasets import load_dataset
import os

DATA_DIR = "data/raw/fever"
os.makedirs(DATA_DIR, exist_ok=True)

dataset = load_dataset(
    "fever",
    "v1.0",
    cache_dir=DATA_DIR
)

print(dataset)
