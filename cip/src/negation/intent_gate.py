def is_factual_question(question: str) -> bool:
    """
    Determines whether negation is logically meaningful.
    Less strict: allows entity hallucinations while blocking opinions.
    """

    q = question.lower().strip()

    # Exclude explanatory questions
    if q.startswith(("why", "how")):
        return False

    # Opinion markers (only block if explicitly subjective)
    opinion_markers = [
        "best", "worst", "greatest", "most", "favorite",
        "better", "opinion"
    ]

    if any(marker in q for marker in opinion_markers) and q.startswith(("who", "what")):
        return False

    # Allow common factual patterns
    factual_starts = ("who", "what", "when", "where", "which", "is", "are")
    return q.startswith(factual_starts)
