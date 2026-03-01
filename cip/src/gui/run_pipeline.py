import sys
import os

# -------------------------------------------------
# FIX PYTHON PATH
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# Imports
# -------------------------------------------------

import logging
import numpy as np
import joblib

from llm_interface.real_llm import llm_answer
from preprocessing.module2_preprocess import module2_process
from classifier.feature_extractor import DistilBERTFeatureExtractor
from rephrase.module3.rephrase_consistency import RephraseConsistencyAnalyzer
from negation.negation_probe import NegationProbe
from fusion.fusion_layer import fuse_prediction, decompose_prediction
from explainability.lime_explainer import CIPExplainer

logger = logging.getLogger("CIPPipeline")

DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data", "processed")
MODEL_PATH = os.path.join(DATA_DIR, "hallucination_model.pkl")
BACKGROUND_PATH = os.path.join(DATA_DIR, "X.npy")
_model = None
_explainer = None


def _get_model():
    """Lazy-load model; returns None if file missing."""
    global _model
    if _model is not None:
        return _model
    if os.path.isfile(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        return _model
    return None


def _get_explainer():
    """Lazy-load LIME explainer; returns None if model/data missing."""
    global _explainer
    if _explainer is not None:
        return _explainer
    if os.path.isfile(MODEL_PATH) and os.path.isfile(BACKGROUND_PATH):
        try:
            X_bg = np.load(BACKGROUND_PATH)
            _explainer = CIPExplainer(MODEL_PATH, X_bg)
            return _explainer
        except Exception as e:
            logger.warning(f"Could not load LIME explainer: {e}")
    return None


def _generate_why_explanation(
    prediction: str,
    consistency: float,
    negation: float,
    decomposition: dict,
) -> str:
    """
    Generate a natural-language explanation of WHY the system made
    its prediction, using the LLM itself for readable output.
    Falls back to a template-based explanation if the LLM call fails.
    """

    dominant = decomposition["dominant_signal"]
    emb_pct = decomposition["embedding"]["percentage"]
    con_pct = decomposition["consistency"]["percentage"]
    neg_pct = decomposition["negation"]["percentage"]

    prompt = (
        f"You are an AI explainability assistant. Explain in 2-3 concise sentences "
        f"why a hallucination detection system classified an LLM response as "
        f'"{prediction}". Use these facts:\n\n'
        f"- Dominant signal: {dominant} ({max(emb_pct, con_pct, neg_pct):.0f}% of decision)\n"
        f"- Embedding model confidence contributed {emb_pct:.0f}%\n"
        f"- Rephrase consistency score: {consistency:.2f} (contributed {con_pct:.0f}%)\n"
        f"- Negation contradiction score: {negation:.2f} (contributed {neg_pct:.0f}%)\n\n"
        f"Be concise. Explain what each signal means in plain English. "
        f"Do NOT use bullet points. Write in third person."
    )

    try:
        return llm_answer(prompt)
    except Exception as e:
        logger.warning(f"LLM explanation failed: {e}")
        # Template fallback
        if prediction == "Hallucination":
            return (
                f"The system flagged this response as a potential hallucination. "
                f"The {dominant.lower()} signal was the strongest contributor "
                f"({max(emb_pct, con_pct, neg_pct):.0f}% of the decision). "
                f"The rephrase consistency score was {consistency:.2f} and the "
                f"negation contradiction score was {negation:.2f}."
            )
        else:
            return (
                f"The system determined this response is likely factual. "
                f"The {dominant.lower()} signal was the dominant factor "
                f"({max(emb_pct, con_pct, neg_pct):.0f}% of the decision). "
                f"The response was consistent across rephrasings ({consistency:.2f}) "
                f"and showed low contradiction ({negation:.2f})."
            )


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
        p_model = 0.3  # lean factual when model not available (innocent until proven guilty)

    # Step 7: Adaptive Fusion
    final_risk = fuse_prediction(
        p_model=p_model,
        consistency_score=consistency,
        negation_score=negation,
    )

    prediction = "Hallucination" if final_risk > 0.5 else "Factual"

    # Step 8: Direct Decomposition (always available)
    decomposition = decompose_prediction(
        p_model=p_model,
        consistency_score=consistency,
        negation_score=negation,
    )

    # Step 9: LIME Explainability (when model exists)
    lime_explanation = None
    explainer = _get_explainer()
    if explainer is not None:
        try:
            lime_explanation = explainer.explain_instance(vector.flatten())
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")

    # Step 10: Natural Language Explanation
    why_explanation = _generate_why_explanation(
        prediction=prediction,
        consistency=consistency,
        negation=negation,
        decomposition=decomposition,
    )

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
        "decomposition": decomposition,
        "explanation": lime_explanation,
        "why_explanation": why_explanation,
    }