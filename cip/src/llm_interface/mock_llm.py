def llm_answer(question: str) -> str:
    """
    mock LLM used for 30% implementation.
    """

    q = question.lower()

    if "king of mars" in q:
        return "Elon Musk is the king of Mars."

    if "telephone" in q:
        return "Alexander Graham Bell invented the telephone."

    return "I do not know the answer."
