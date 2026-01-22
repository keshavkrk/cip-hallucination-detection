def rephrase_question(question: str) -> str:
    """
    Mock rephraser for 30% implementation.
    Deterministic and offline.
    """

    q = question.lower()

    if "who invented the telephone" in q:
        return "Can you tell me who was the inventor of the telephone?"

    if "king of mars" in q:
        return "Who is considered the ruler of Mars?"

    # fallback: return original question
    return question
