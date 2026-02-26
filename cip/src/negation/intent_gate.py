def is_factual_question(question: str) -> bool:
    """
    Determines whether negation is logically meaningful.

    Strategy:
    - Block clearly explanatory (why/how)
    - Block clearly subjective/opinion queries
    - Allow everything else (including conditionals)
    """

    q = question.lower().strip()

    # Block explanatory reasoning questions
    if q.startswith(("why", "how")):
        return False

    # Block explicit opinion-based questions
    opinion_markers = [
        "best", "worst", "greatest", "most",
        "favorite", "better", "opinion"
    ]

    if any(marker in q for marker in opinion_markers):
        return False

    # Otherwise allow
    return True