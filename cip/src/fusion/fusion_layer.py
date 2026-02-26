# cip/src/fusion/fusion_layer.py

def fuse_prediction(p_model, c_score, n_score):
    """
    Combines:
        - SVM hallucination probability
        - Consistency score
        - Negation contradiction score

    Returns:
        final hallucination probability
    """

    # Confidence guard
    if abs(p_model - 0.5) > 0.25:
        return p_model

    # Convert to aligned risk signals
    risk_model = p_model
    risk_consistency = 1 - c_score
    risk_negation = n_score

    # Weighted fusion (LOCKED)
    final_risk = (
        0.5 * risk_model
        + 0.25 * risk_consistency
        + 0.25 * risk_negation
    )

    # Clamp to valid range
    final_risk = max(0.0, min(1.0, final_risk))

    return final_risk