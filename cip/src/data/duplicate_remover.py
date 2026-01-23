import pandas as pd

df = pd.read_csv("../data/processed/val.csv")

print("Before:", len(df))

df = df.drop_duplicates()

print("After:", len(df))

df.to_csv("../data/processed/val_cleaned.csv", index=False)
