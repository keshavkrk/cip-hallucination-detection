# cip/src/fusion/fusion_layer.py

"""
Adaptive Fusion Layer
=====================
Dynamically adjusts signal weights based on signal confidence/reliability
instead of using fixed weights. This produces more accurate predictions
because the system emphasizes whichever signal is most informative for
each specific query.

Signals:
  • p_model       — Embedding model probability (from classifier)
  • consistency   — Rephrase consistency score (Module 3)
  • negation      — Negation contradiction score (Module 4)
"""

# -------------------------------------------------
# Base weights (prior/default)
# -------------------------------------------------
BASE_ALPHA = 0.50   # embedding
BASE_BETA  = 0.30  # consistency
BASE_GAMMA = 0.20   # negation


def _compute_adaptive_weights(
    p_model: float,
    consistency_score: float,
    negation_score: float,
) -> tuple[float, float, float]:
    """
    Dynamically adjust weights based on signal confidence.

    Logic:
    ─────
    1. Embedding boost:  When model is very confident (far from 0.5),
       its signal is more trustworthy → increase its weight.

    2. Consistency boost: When consistency is very low (< 0.3), the LLM
       contradicted itself → strong hallucination indicator → boost weight.

    3. Negation boost: When negation score is high (> 0.6), a clear
       contradiction was detected → boost weight.

    All weights are normalized to sum to 1.0.
    """
    alpha = BASE_ALPHA
    beta = BASE_BETA
    gamma = BASE_GAMMA

    # ── Embedding confidence boost ──
    # Distance from 0.5 = how confident the model is
    model_confidence = abs(p_model - 0.5) * 2  # 0.0 = uncertain, 1.0 = certain
    alpha += 0.15 * model_confidence

    # ── Consistency boost ──
    # Low consistency = LLM gave different answers to same question
    if consistency_score < 0.3:
        beta += 0.20  # strong signal — heavily boost
    elif consistency_score < 0.5:
        beta += 0.10  # moderate signal

    # ── Negation boost ──
    # High negation = LLM contradicted itself on negated question
    if negation_score > 0.6:
        gamma += 0.20  # strong contradiction detected
    elif negation_score > 0.4:
        gamma += 0.10

    # ── Normalize so weights sum to 1.0 ──
    total = alpha + beta + gamma
    alpha /= total
    beta /= total
    gamma /= total

    return alpha, beta, gamma


def fuse_prediction(
    p_model: float,
    consistency_score: float,
    negation_score: float,
) -> float:
    """
    Weighted fusion of three signals with adaptive weights.

    Returns a float in [0, 1] representing hallucination risk.
    Higher = more likely hallucination.
    """

    # Safety for None values
    if consistency_score is None:
        consistency_score = 0.0
    if negation_score is None:
        negation_score = 0.0

    # Compute adaptive weights
    alpha, beta, gamma = _compute_adaptive_weights(
        p_model, consistency_score, negation_score
    )

    final_score = (
        alpha * p_model
        + beta * (1 - consistency_score)    # invert: low consistency → high risk
        + gamma * negation_score
    )

    return float(final_score)


def decompose_prediction(
    p_model: float,
    consistency_score: float,
    negation_score: float,
) -> dict:
    """
    Direct Decomposition — exact contribution of each signal.

    Returns a breakdown showing:
      • Each signal's weight, raw contribution, and percentage of risk
      • The dominant signal
      • The adaptive weights used
    """

    if consistency_score is None:
        consistency_score = 0.0
    if negation_score is None:
        negation_score = 0.0

    alpha, beta, gamma = _compute_adaptive_weights(
        p_model, consistency_score, negation_score
    )

    emb_contrib = alpha * p_model
    con_contrib = beta * (1 - consistency_score)
    neg_contrib = gamma * negation_score
    total = emb_contrib + con_contrib + neg_contrib

    # Percentage of total risk attributed to each signal
    if total > 0:
        emb_pct = emb_contrib / total * 100
        con_pct = con_contrib / total * 100
        neg_pct = neg_contrib / total * 100
    else:
        emb_pct = con_pct = neg_pct = 33.3

    contributions = {
        "Embedding": abs(emb_contrib),
        "Consistency": abs(con_contrib),
        "Negation": abs(neg_contrib),
    }
    dominant = max(contributions, key=contributions.get)

    return {
        "final_risk": float(total),
        "embedding": {
            "weight": round(alpha, 3),
            "contribution": round(emb_contrib, 4),
            "percentage": round(emb_pct, 1),
        },
        "consistency": {
            "weight": round(beta, 3),
            "contribution": round(con_contrib, 4),
            "percentage": round(con_pct, 1),
        },
        "negation": {
            "weight": round(gamma, 3),
            "contribution": round(neg_contrib, 4),
            "percentage": round(neg_pct, 1),
        },
        "dominant_signal": dominant,
        "weights_used": {
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "gamma": round(gamma, 3),
        },
    }