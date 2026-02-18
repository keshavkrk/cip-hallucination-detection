# cip/src/classifier/build_features.py

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.module2_preprocess import module2_process
from classifier.feature_extractor import DistilBERTFeatureExtractor
from rephrase.module3.rephrase_consistency import RephraseConsistencyAnalyzer
from negation.negation_probe import NegationProbe


INPUT_PATH = "../data/processed/truthfulqa_pairs.csv"
OUTPUT_X = "../data/processed/X.npy"
OUTPUT_y = "../data/processed/y.npy"

SAVE_EVERY = 100
RETRY_LIMIT = 3
SLEEP_TIME = 1.5  # safer for Groq overnight


def safe_call(func, *args, **kwargs):
    """
    Retry wrapper for unstable API calls (handles 429, timeouts).
    """
    for attempt in range(RETRY_LIMIT):
        try:
            return func(*args, **kwargs)
        except Exception:
            time.sleep(3)
    return None


def main():

    df = pd.read_csv(INPUT_PATH)

    extractor = DistilBERTFeatureExtractor()
    m3 = RephraseConsistencyAnalyzer(num_paraphrases=1)
    m4 = NegationProbe()

    # -------------------------
    # Resume logic
    # -------------------------
    if os.path.exists(OUTPUT_X) and os.path.exists(OUTPUT_y):
        print("Resuming from checkpoint...")
        features = list(np.load(OUTPUT_X))
        labels = list(np.load(OUTPUT_y))
    else:
        features = []
        labels = []

    skipped = 0
    start_index = len(features)

    print("Building FULL 770-dim features (stable overnight mode)...")
    print(f"Starting from index: {start_index}")

    try:
        for idx, row in tqdm(
            df.iloc[start_index:].iterrows(),
            total=len(df) - start_index
        ):

            question = row["question"]
            answer = row["answer"]
            label = row["label"]

            if pd.isna(question) or pd.isna(answer):
                skipped += 1
                continue

            try:
                # -------- Module 2 (no API here)
                m2 = module2_process(question, answer)

                embedding = extractor.extract(
                    m2["input_ids"],
                    m2["attention_mask"]
                ).cpu().numpy().flatten()

                # -------- Consistency
                m3_out = safe_call(m3.run, question, answer)
                if m3_out is None:
                    skipped += 1
                    continue

                consistency = m3_out.get("consistency_score") or 0.0

                # -------- Contradiction
                m4_out = safe_call(m4.run, question, answer)
                if m4_out is None:
                    skipped += 1
                    continue

                contradiction = m4_out.get("contradiction_score", 0.0)

                # -------- Final 770 vector
                vector = np.concatenate([
                    embedding,
                    np.array([consistency]),
                    np.array([contradiction])
                ])

                features.append(vector)
                labels.append(label)

                # -------- Periodic checkpoint
                if len(features) % SAVE_EVERY == 0:
                    np.save(OUTPUT_X, np.array(features))
                    np.save(OUTPUT_y, np.array(labels))
                    print(f"\nCheckpoint saved at {len(features)} samples.")

                time.sleep(SLEEP_TIME)

            except Exception:
                skipped += 1
                continue

    except KeyboardInterrupt:
        print("\nInterrupted manually. Saving progress...")

    # -------------------------
    # Final Save
    # -------------------------
    X = np.array(features)
    y = np.array(labels)

    os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_y, y)

    print("\nFinished.")
    print("Final Feature shape:", X.shape)
    print("Final Label shape:", y.shape)
    print("Skipped rows:", skipped)


if __name__ == "__main__":
    main()
