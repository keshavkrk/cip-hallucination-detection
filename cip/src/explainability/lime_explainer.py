# cip/src/explainability/lime_explainer.py

import numpy as np
import joblib

from lime.lime_tabular import LimeTabularExplainer
from fusion.fusion_layer import fuse_prediction


class CIPExplainer:
    """
    LIME-based grouped explainability:
    - Embedding (768 dims)
    - Consistency (1 dim)
    - Negation (1 dim)
    """

    def __init__(self, model_path, X_background):
        self.model = joblib.load(model_path)

        self.explainer = LimeTabularExplainer(
            X_background,
            mode="classification",
            discretize_continuous=True
        )

    def _predict_with_fusion(self, X):
        """
        Wraps model + fusion into LIME-compatible predict function
        """
        probs = self.model.predict_proba(X)[:, 1]

        fused_probs = []
        for i in range(len(X)):
            p_model = probs[i]
            c_score = X[i, 768]
            n_score = X[i, 769]

            fused = fuse_prediction(p_model, c_score, n_score)
            fused_probs.append(fused)

        fused_probs = np.array(fused_probs)

        return np.vstack([1 - fused_probs, fused_probs]).T

    def explain_instance(self, instance, num_features=20):
        """
        Returns grouped explanation dictionary
        """

        explanation = self.explainer.explain_instance(
            instance,
            self._predict_with_fusion,
            num_features=num_features
        )

        weights = explanation.as_list()

        embedding_contribution = 0.0
        consistency_contribution = 0.0
        negation_contribution = 0.0

        for feature, weight in weights:

            # Extract feature index
            if "feature" in feature:
                continue

            try:
                idx = int(feature.split(" ")[0])
            except:
                continue

            if idx < 768:
                embedding_contribution += weight
            elif idx == 768:
                consistency_contribution += weight
            elif idx == 769:
                negation_contribution += weight

        return {
            "embedding_signal": embedding_contribution,
            "consistency_signal": consistency_contribution,
            "negation_signal": negation_contribution,
            "raw_explanation": weights
        }