# cip/src/classifier/build_features.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.module2_preprocess import module2_process
from classifier.feature_extractor import DistilBERTFeatureExtractor
from negation.nli_scorer import NLIScorer


INPUT_PATH = "../data/processed/truthfulqa_pairs.csv"
OUTPUT_X = "../data/processed/X.npy"
OUTPUT_y = "../data/processed/y.npy"


def main():

    df = pd.read_csv(INPUT_PATH)

    extractor = DistilBERTFeatureExtractor()
    nli = NLIScorer()

    features = []
    labels = []

    print("Building OFFLINE 770-dim features...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        question = row["question"]
        answer = row["answer"]
        label = row["label"]

        if pd.isna(question) or pd.isna(answer):
            continue

        # -------- Module 2 (no LLM)
        m2 = module2_process(question, answer)

        embedding = extractor.extract(
            m2["input_ids"],
            m2["attention_mask"]
        ).cpu().numpy().flatten()

        # -------- NLI scores (offline)
        entailment = nli.entailment_score(question, answer)
        contradiction = nli.contradiction_score(question, answer)

        vector = np.concatenate([
            embedding,
            np.array([entailment]),
            np.array([contradiction])
        ])

        features.append(vector)
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_y, y)

    print("Done.")
    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)


if __name__ == "__main__":
    main()