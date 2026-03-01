"""
Intent Gate for the Negation Probe.

Returns a confidence multiplier [0.0 – 1.0] that indicates how much
the negation signal should be trusted for a given question type.

  1.0 = full trust   (boolean/conditional questions)
  0.3 = dampened     (identity questions — who/what/when/where)
  0.0 = skip         (explanatory/opinion questions — why/how/best)
"""

# Full dampening — negation is not meaningful
_BLOCK_PREFIXES = ("why", "how")

# Partial dampening — negation naturally produces different answers
# but can still provide weak signal for hallucination detection
_DAMPEN_PREFIXES = ("who", "what", "when", "where", "which", "name")
_DAMPEN_FACTOR = 0.3

# Opinion markers — negation not meaningful
_OPINION_MARKERS = [
    "best", "worst", "greatest", "most",
    "favorite", "better", "opinion",
]


def negation_confidence(question: str) -> float:
    """
    Returns a confidence multiplier for the negation signal.

      1.0  →  Full weight  (boolean, conditional, "is X true?")
      0.3  →  Dampened     (who/what/when/where — identity queries)
      0.0  →  Skip         (why/how, opinion queries)
    """
    q = question.lower().strip()

    # Block: explanatory questions
    if q.startswith(_BLOCK_PREFIXES):
        return 0.0

    # Block: opinion questions
    if any(marker in q for marker in _OPINION_MARKERS):
        return 0.0

    # Dampen: identity questions
    if q.startswith(_DAMPEN_PREFIXES):
        return _DAMPEN_FACTOR

    # Full trust: boolean, conditional, etc.
    return 1.0


# Backward compatible wrapper
def is_factual_question(question: str) -> bool:
    """Returns True if negation should run at all (confidence > 0)."""
    return negation_confidence(question) > 0.0