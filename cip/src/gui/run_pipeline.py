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

    print("\n" + "=" * 70)
    print("              CIP HALLUCINATION DETECTION PIPELINE")
    print("=" * 70)
    print(f"  Input Question: {question}")
    print("=" * 70)

    # Step 1: LLM answer
    answer = llm_answer(question)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 2 · TEXT PREPROCESSING                                │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Question (clean) : {question[:55]}")
    print(f"│  LLM Answer       : {answer[:55]}{'…' if len(answer) > 55 else ''}")

    # Step 2: Embedding
    m2 = module2_process(question, answer)

    embedding = extractor.extract(
        m2["input_ids"],
        m2["attention_mask"]
    ).cpu().numpy().flatten()

    print(f"│  QA Text           : {m2['qa_text'][:55]}{'…' if len(m2['qa_text']) > 55 else ''}")
    print(f"│  Input IDs shape   : {m2['input_ids'].shape}")
    print(f"│  Attention mask    : {m2['attention_mask'].shape}")
    print("└─────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 5 · TRANSFORMER FEATURE EXTRACTION (DistilBERT)       │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Embedding shape   : ({len(embedding)},)")
    print(f"│  Embedding [0:5]   : {[round(float(x), 4) for x in embedding[:5]]}")
    print(f"│  Embedding norm    : {float(np.linalg.norm(embedding)):.4f}")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 3: Consistency
    m3_out = m3.run(question, answer)

    consistency = m3_out.get("consistency_score") or 0.0
    paraphrases = m3_out.get("paraphrases", [])
    rephrased_answers = m3_out.get("rephrased_answers", [])

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 3 · REPHRASE CONSISTENCY ANALYZER                     │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Consistency Score : {consistency:.4f}")
    print(f"│  Paraphrases ({len(paraphrases)})  :")
    for i, p in enumerate(paraphrases, 1):
        print(f"│    {i}. {p[:58]}{'…' if len(p) > 58 else ''}")
    print(f"│  Rephrased Answers ({len(rephrased_answers)}):")
    for i, a in enumerate(rephrased_answers, 1):
        print(f"│    {i}. {a[:58]}{'…' if len(a) > 58 else ''}")
    if m3_out.get("reason"):
        print(f"│  Reason            : {m3_out['reason']}")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 4: Negation
    m4_out = m4.run(question, answer)

    negation = m4_out.get("contradiction_score", 0.0)
    negated_question = m4_out.get("negated_question")
    negated_answer = m4_out.get("negated_answer")

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 4 · NEGATION PROBE                                    │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Contradiction Flag  : {m4_out.get('antonym_contradiction_flag', 0)}")
    print(f"│  Contradiction Score : {negation:.4f}")
    if negated_question:
        print(f"│  Negated Question    : {negated_question[:55]}{'…' if len(negated_question) > 55 else ''}")
    if negated_answer:
        print(f"│  Negated Answer      : {negated_answer[:55]}{'…' if len(negated_answer) > 55 else ''}")
    if m4_out.get("reason"):
        print(f"│  Reason              : {m4_out['reason']}")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 5: Feature vector
    vector = np.concatenate([
        embedding,
        np.array([consistency]),
        np.array([negation])
    ]).reshape(1, -1)

    vector = np.nan_to_num(vector)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  FEATURE VECTOR CONSTRUCTION                                  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Vector shape       : {vector.shape}")
    print(f"│  [0..767] embedding : DistilBERT CLS ({len(embedding)} dims)")
    print(f"│  [768] consistency  : {consistency:.4f}")
    print(f"│  [769] negation     : {negation:.4f}")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 6: Model probability
    model = _get_model()
    if model is not None:
        p_model = model.predict_proba(vector)[0, 1]
    else:
        p_model = 0.3  # lean factual when model not available (innocent until proven guilty)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 6 · HALLUCINATION CLASSIFIER (Calibrated SVM)         │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Model loaded       : {'✅ Yes' if model is not None else '❌ No (using default p=0.3)'}")
    print(f"│  P(hallucination)   : {p_model:.4f}")
    print(f"│  P(factual)         : {1 - p_model:.4f}")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 7: Adaptive Fusion
    final_risk = fuse_prediction(
        p_model=p_model,
        consistency_score=consistency,
        negation_score=negation,
    )

    prediction = "Hallucination" if final_risk > 0.5 else "Factual"

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 7 · FEATURE FUSION LAYER (Adaptive Weights)           │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Inputs:")
    print(f"│    p_model (embedding)     : {p_model:.4f}")
    print(f"│    consistency_score       : {consistency:.4f}")
    print(f"│    negation_score          : {negation:.4f}")
    print(f"│  Final Fused Risk          : {final_risk:.4f}")
    print(f"│  Prediction                : {'🔴 ' + prediction if prediction == 'Hallucination' else '🟢 ' + prediction}")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 8: Direct Decomposition (always available)
    decomposition = decompose_prediction(
        p_model=p_model,
        consistency_score=consistency,
        negation_score=negation,
    )

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 8 · CONFIDENCE CALIBRATION (Direct Decomposition)     │")
    print("├─────────────────────────────────────────────────────────────────┤")
    emb_d = decomposition["embedding"]
    con_d = decomposition["consistency"]
    neg_d = decomposition["negation"]
    print(f"│  Embedding   → weight: {emb_d['weight']:.3f}  contrib: {emb_d['contribution']:.4f}  ({emb_d['percentage']:.1f}%)")
    print(f"│  Consistency → weight: {con_d['weight']:.3f}  contrib: {con_d['contribution']:.4f}  ({con_d['percentage']:.1f}%)")
    print(f"│  Negation    → weight: {neg_d['weight']:.3f}  contrib: {neg_d['contribution']:.4f}  ({neg_d['percentage']:.1f}%)")
    print(f"│  Dominant Signal     : ⭐ {decomposition['dominant_signal']}")
    print(f"│  Adaptive Weights    : α={decomposition['weights_used']['alpha']:.3f}  β={decomposition['weights_used']['beta']:.3f}  γ={decomposition['weights_used']['gamma']:.3f}")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 9: LIME Explainability (when model exists)
    lime_explanation = None
    explainer = _get_explainer()
    if explainer is not None:
        try:
            lime_explanation = explainer.explain_instance(vector.flatten())
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 9 · EXPLAINABILITY (LIME)                             │")
    print("├─────────────────────────────────────────────────────────────────┤")
    if lime_explanation:
        print(f"│  Embedding Signal    : {lime_explanation['embedding_signal']:+.4f}")
        print(f"│  Consistency Signal  : {lime_explanation['consistency_signal']:+.4f}")
        print(f"│  Negation Signal     : {lime_explanation['negation_signal']:+.4f}")
        print(f"│  Dominant Signal     : ⭐ {lime_explanation['dominant_signal']}")
    else:
        print("│  ⚠️  LIME unavailable (model or background data not found)")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Step 10: Natural Language Explanation
    why_explanation = _generate_why_explanation(
        prediction=prediction,
        consistency=consistency,
        negation=negation,
        decomposition=decomposition,
    )

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  MODULE 10 · NATURAL LANGUAGE EXPLANATION                     │")
    print("├─────────────────────────────────────────────────────────────────┤")
    # Word-wrap the explanation to fit the box
    _why_lines = [why_explanation[i:i+60] for i in range(0, len(why_explanation), 60)]
    for line in _why_lines:
        print(f"│  {line}")
    print("└─────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 70)
    print(f"  FINAL RESULT: {prediction}  |  Risk: {final_risk:.4f}  |  Confidence: {1 - final_risk:.4f}")
    print("=" * 70 + "\n")

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