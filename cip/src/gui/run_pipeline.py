import sys
import os

# -------------------------------------------------
# FIX PYTHON PATH
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# Now imports will work
# -------------------------------------------------

import numpy as np
import joblib

from llm_interface.real_llm import llm_answer
from preprocessing.module2_preprocess import module2_process
from classifier.feature_extractor import DistilBERTFeatureExtractor
from rephrase.module3.rephrase_consistency import RephraseConsistencyAnalyzer
from negation.negation_probe import NegationProbe
from fusion.fusion_layer import fuse_prediction


MODEL_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "data", "processed", "hallucination_model.pkl")
_model = None


def _get_model():
    """Lazy-load model; returns None if file missing."""
    global _model
    if _model is not None:
        return _model
    if os.path.isfile(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        return _model
    return None


extractor = DistilBERTFeatureExtractor()
m3 = RephraseConsistencyAnalyzer(num_paraphrases=1)
m4 = NegationProbe()


def run_cip_pipeline(question: str) -> dict:

    # Step 1: LLM answer
    answer = llm_answer(question)

    # Step 2: Embedding
    m2 = module2_process(question, answer)

    embedding = extractor.extract(
        m2["input_ids"],
        m2["attention_mask"]
    ).cpu().numpy().flatten()

    # Step 3: Consistency
    m3_out = m3.run(question, answer)

    consistency = m3_out.get("consistency_score") or 0.0
    paraphrases = m3_out.get("paraphrases", [])
    rephrased_answers = m3_out.get("rephrased_answers", [])

    # Step 4: Negation
    m4_out = m4.run(question, answer)

    negation = m4_out.get("contradiction_score", 0.0)
    negated_question = m4_out.get("negated_question")
    negated_answer = m4_out.get("negated_answer")

    # Step 5: Feature vector
    vector = np.concatenate([
        embedding,
        np.array([consistency]),
        np.array([negation])
    ]).reshape(1, -1)

    vector = np.nan_to_num(vector)

    # Step 6: Model probability
    model = _get_model()
    if model is not None:
        p_model = model.predict_proba(vector)[0, 1]
    else:
        p_model = 0.5  # neutral fallback when model not trained yet

    # Step 7: Fusion

    final_risk = fuse_prediction(
        p_model=p_model,
        consistency_score=consistency,
        negation_score=negation,
    )

    prediction = "Hallucination" if final_risk > 0.5 else "Factual"

    return {
        "answer": answer,
        "p_model": float(p_model),
        "consistency": float(consistency),
        "negation": float(negation),
        "final_risk": float(final_risk),
        "prediction": prediction,
        "paraphrases": paraphrases,
        "rephrased_answers": rephrased_answers,
        "negated_question": negated_question,
        "negated_answer": negated_answer,
        "model_loaded": model is not None,
    }