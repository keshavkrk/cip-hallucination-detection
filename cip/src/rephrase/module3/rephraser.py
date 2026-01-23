from llm_interface.real_llm import llm_answer


def rephrase_question(question: str, k: int = 3) -> list[str]:
    """
    Generate k paraphrases of a question using real LLM.
    """

    prompt = (
        f"Generate {k} different paraphrases of the following question.\n"
        f"Return only the paraphrased questions, one per line.\n\n"
        f"Question: {question}"
    )

    response = llm_answer(prompt)

    paraphrases = [
        line.strip("- ").strip()
        for line in response.split("\n")
        if line.strip()
    ]

    return paraphrases[:k]
