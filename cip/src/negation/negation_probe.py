import logging
from llm_interface.mock_llm import llm_answer
from negation.rule_negator import negate_question
from negation.nli_scorer import NLIScorer
from negation.intent_gate import is_factual_question


class NegationProbe:
    """
    Module 4: Negation Probe
    NLI model is loaded ONCE and reused.
    """

    def __init__(self):
        self.nli = NLIScorer()  # lightweight handle, model already loaded
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Module4")

    def run(self, question: str, original_answer: str) -> dict:

        if not is_factual_question(question):
            return {
                "antonym_contradiction_flag": None,
                "reason": "non_factual_question"
            }

        # Rule-based negation
        neg_question = negate_question(question)

        # Same LLM answers negated question
        neg_answer = llm_answer(neg_question)

        # Contradiction score (MNLI)
        score = self.nli.contradiction_score(
            premise=original_answer,
            hypothesis=neg_answer
        )

        flag = 1 if score > 0.5 else 0

        self.logger.info(f"Q     : {question}")
        self.logger.info(f"Q_neg : {neg_question}")
        self.logger.info(f"A     : {original_answer}")
        self.logger.info(f"A_neg : {neg_answer}")
        self.logger.info(f"Score : {score:.4f}")

        return {
            "antonym_contradiction_flag": flag,
            "contradiction_score": score,
            "negated_answer": neg_answer
        }
