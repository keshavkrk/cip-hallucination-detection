import os
import pandas as pd

INPUT_PATH = "../data/truthfulQA/TruthfulQA.csv"
OUTPUT_PATH = "../data/processed/truthfulqa_pairs.csv"


def split_answers(text):
    if pd.isna(text):
        return []
    return [a.strip() for a in str(text).split(";") if a.strip()]


def main():
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for _, row in df.iterrows():
        question = str(row["Question"]).strip()

        correct_answers = split_answers(row["Correct Answers"])
        incorrect_answers = split_answers(row["Incorrect Answers"])

        # Add correct answers (label 0)
        for ans in correct_answers:
            rows.append({
                "question": question,
                "answer": ans,
                "label": 0
            })

        # Add incorrect answers (label 1)
        for ans in incorrect_answers:
            rows.append({
                "question": question,
                "answer": ans,
                "label": 1
            })

    final_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Done.")
    print("Total samples:", len(final_df))
    print("Label distribution:")
    print(final_df["label"].value_counts())


if __name__ == "__main__":
    main()
