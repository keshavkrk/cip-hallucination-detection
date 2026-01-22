def negate_question(question: str) -> str:
    """
    Rule-based negation.
    Deterministic and examiner-friendly.
    """

    q = question.lower()

    if "invented" in q:
        return question.replace("invented", "did not invent")

    if " is " in q:
        return question.replace(" is ", " is not ")

    return "NOT " + question
