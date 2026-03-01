# cip/src/explainability/lime_explainer.py

import numpy as np
import joblib
import logging

from lime.lime_tabular import LimeTabularExplainer
from fusion.fusion_layer import fuse_prediction

logger = logging.getLogger("CIPExplainer")

# Feature layout:  [0..767] = embedding,  768 = consistency,  769 = negation
_EMB_END = 768
_CONSISTENCY_IDX = 768
_NEGATION_IDX = 769
_TOTAL_FEATURES = 770

_FEATURE_NAMES = [f"emb_{i}" for i in range(_EMB_END)] + ["consistency", "negation"]


class CIPExplainer:
    """
    LIME-based grouped explainability for the CIP pipeline.

    Groups the 770-dimensional feature vector into three interpretable signals:
      • Embedding  (768 dims → aggregated into one score)
      • Consistency (1 dim)
      • Negation    (1 dim)

    Returns contribution weights showing HOW MUCH each signal pushed
    the prediction toward hallucination (positive) or factual (negative).
    """

    def __init__(self, model_path: str, X_background: np.ndarray):
        self.model = joblib.load(model_path)

        self.explainer = LimeTabularExplainer(
            X_background,
            feature_names=_FEATURE_NAMES,
            mode="classification",
            class_names=["Factual", "Hallucination"],
            discretize_continuous=True,
            random_state=42,
        )

    # --------------------------------------------------
    # LIME-compatible prediction wrapper (model + fusion)
    # --------------------------------------------------
    def _predict_with_fusion(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X)[:, 1]

        fused = np.array([
            fuse_prediction(
                p_model=float(probs[i]),
                consistency_score=float(X[i, _CONSISTENCY_IDX]),
                negation_score=float(X[i, _NEGATION_IDX]),
            )
            for i in range(len(X))
        ])

        return np.vstack([1 - fused, fused]).T

    # --------------------------------------------------
    # Main entry point
    # --------------------------------------------------
    def explain_instance(self, instance: np.ndarray, num_features: int = 20) -> dict:
        """
        Explain a single 770-d feature vector.

        Returns dict with:
          embedding_signal   – aggregated LIME weight for embedding features
          consistency_signal – LIME weight for the consistency feature
          negation_signal    – LIME weight for the negation feature
          dominant_signal    – name of the strongest contributor
          raw_weights        – full list of (feature_name, weight) from LIME
        """
        instance = np.asarray(instance).flatten()

        explanation = self.explainer.explain_instance(
            instance,
            self._predict_with_fusion,
            num_features=min(num_features, _TOTAL_FEATURES),
            num_samples=500,
        )

        weights = explanation.as_list()

        embedding_signal = 0.0
        consistency_signal = 0.0
        negation_signal = 0.0

        for feature_desc, weight in weights:
            # LIME returns descriptions like "emb_42 > 0.12" or "consistency <= 0.50"
            name = feature_desc.split()[0] if feature_desc else ""

            if name == "consistency":
                consistency_signal = weight
            elif name == "negation":
                negation_signal = weight
            elif name.startswith("emb_"):
                embedding_signal += weight

        # Determine dominant signal
        signals = {
            "Embedding": abs(embedding_signal),
            "Consistency": abs(consistency_signal),
            "Negation": abs(negation_signal),
        }
        dominant = max(signals, key=signals.get)

        return {
            "embedding_signal": round(embedding_signal, 4),
            "consistency_signal": round(consistency_signal, 4),
            "negation_signal": round(negation_signal, 4),
            "dominant_signal": dominant,
            "raw_weights": weights,
        }