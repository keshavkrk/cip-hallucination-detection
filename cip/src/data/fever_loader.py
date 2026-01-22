from datasets import load_dataset
import pandas as pd

LABEL_MAP = {
    "SUPPORTS": 1,
    "REFUTES": 0
}

class FeverDataset:
    def __init__(self, split, cache_dir="data/raw/fever"):
        self.dataset = load_dataset(
            "fever",
            "v1.0",
            split=split,
            cache_dir=cache_dir
        )

    def to_dataframe(self):
        rows = []

        for item in self.dataset:
            label = item["label"]
            if label not in LABEL_MAP:
                continue  # drop NEI

            rows.append({
                "id": str(item["id"]),
                "claim": item["claim"],
                "label": LABEL_MAP[label],
                "source": "fever"
            })

        return pd.DataFrame(rows)
