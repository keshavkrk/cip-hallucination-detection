import pandas as pd
from pathlib import Path

# resolve project root automatically
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

FILES = ["train.csv", "val.csv"]

print("=" * 60)
print("DATASET VALIDATION REPORT")
print("=" * 60)

for file in FILES:
    path = DATA_DIR / file

    if not path.exists():
        print(f"\n❌ {file} NOT FOUND at {path}")
        continue

    df = pd.read_csv(path)

    print(f"\n📄 File: {file}")
    print("-" * 60)

    print("Total samples:", len(df))

    if "label" in df.columns:
        print("\nClass distribution:")
        print(df["label"].value_counts())
        print("\nClass ratio:")
        print(df["label"].value_counts(normalize=True))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
    if "claim" in df.columns:
        print("Duplicate claims:", df["claim"].duplicated().sum())

    if "claim" in df.columns:
        df["claim_len"] = df["claim"].str.split().str.len()
        print("\nClaim length stats:")
        print(df["claim_len"].describe())

print("\n✅ Validation complete")
    