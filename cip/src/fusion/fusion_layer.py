# cip/src/fusion/fusion_layer.py

ALPHA = 0.6
BETA = 0.25
GAMMA = 0.15


def fuse_prediction(p_model: float,
                    consistency_score: float,
                    negation_score: float) -> float:
    """
    Weighted fusion of:
    - Embedding model probability
    - Rephrase consistency score
    - Negation contradiction score
    """

    # Safety for None values
    if consistency_score is None:
        consistency_score = 0.0

    if negation_score is None:
        negation_score = 0.0

    final_score = (
        ALPHA * p_model +
        BETA * (1 - consistency_score) +   # invert consistency
        GAMMA * negation_score
    )

    return float(final_score)