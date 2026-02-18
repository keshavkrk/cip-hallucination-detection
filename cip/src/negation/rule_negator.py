# cip/src/negation/rule_negator.py

import spacy

nlp = spacy.load("en_core_web_sm")


def negate_question(question: str) -> str:
    doc = nlp(question)

    tokens = [token.text for token in doc]
    lower_tokens = [token.text.lower() for token in doc]

    # -------------------------
    # Case 1: Auxiliary present
    # -------------------------
    for token in doc:
        if token.dep_ == "aux":
            idx = token.i
            tokens.insert(idx + 1, "not")
            return " ".join(tokens)

    # -------------------------
    # Case 2: BE verb question
    # -------------------------
    for token in doc:
        if token.lemma_ == "be" and token.pos_ == "AUX":
            idx = token.i
            tokens.insert(idx + 1, "not")
            return " ".join(tokens)

    # -------------------------
    # Case 3: DO-support insertion
    # -------------------------
    root = [t for t in doc if t.dep_ == "ROOT"]
    if root:
        root = root[0]

        # Past tense
        if root.tag_ == "VBD":
            return question.replace(
                root.text,
                f"did not {root.lemma_}"
            )

        # Present 3rd singular
        if root.tag_ == "VBZ":
            return question.replace(
                root.text,
                f"does not {root.lemma_}"
            )

        # Present plural
        if root.tag_ == "VBP":
            return question.replace(
                root.text,
                f"do not {root.lemma_}"
            )

    # -------------------------
    # Fallback
    # -------------------------
    return question
