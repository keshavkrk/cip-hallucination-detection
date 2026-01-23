import spacy

# Load once
_nlp = spacy.load("en_core_web_sm")


def negate_question(question: str) -> str:
    """
    spaCy-based logical negation.
    Handles auxiliary verbs + main verbs.
    Deterministic and explainable.
    """

    doc = _nlp(question)

    tokens = []
    negated = False

    for token in doc:
        # Case 1: auxiliary verb → insert NOT
        if token.dep_ == "aux" and not negated:
            tokens.append(token.text)
            tokens.append("not")
            negated = True

        # Case 2: main verb without aux → do-support
        elif token.pos_ == "VERB" and not negated:
            tokens.append("did")
            tokens.append("not")
            tokens.append(token.lemma_)
            negated = True

        else:
            tokens.append(token.text)

    return " ".join(tokens)
