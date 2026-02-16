import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.module2_preprocess import module2_process
from classifier.feature_extractor import DistilBERTFeatureExtractor


INPUT_PATH = "../data/processed/truthfulqa_pairs.csv"
OUTPUT_X = "../data/processed/X.npy"
OUTPUT_y = "../data/processed/y.npy"


def main():

    df = pd.read_csv(INPUT_PATH)

    extractor = DistilBERTFeatureExtractor()

    features = []
    labels = []

    skipped = 0

    print("Building embedding features (768-d only)...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        question = row.get("question")
        answer = row.get("answer")
        label = row.get("label")

        # ---- Skip invalid rows
        if pd.isna(question) or pd.isna(answer) or pd.isna(label):
            skipped += 1
            continue

        try:
            # Module 2 (no LLM call because answer is provided)
            m2 = module2_process(question, answer)

            # Module 5 → 768-d embedding
            embedding = extractor.extract(
                m2["input_ids"],
                m2["attention_mask"]
            ).cpu().numpy().flatten()

            features.append(embedding)
            labels.append(int(label))

        except Exception as e:
            skipped += 1
            continue

    X = np.array(features)
    y = np.array(labels)

    os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_y, y)

    print("\nDone.")
    print("Final Feature shape:", X.shape)
    print("Final Label shape:", y.shape)
    print("Skipped rows:", skipped)


if __name__ == "__main__":
    main()
