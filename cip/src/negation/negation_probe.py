import logging
from llm_interface.real_llm import llm_answer   # ✅ real LLM
from negation.rule_negator import negate_question
from negation.nli_scorer import NLIScorer
from negation.intent_gate import is_factual_question


class NegationProbe:
    """
    Module 4: Negation Probe (100%)

    - spaCy-based negation
    - SAME real LLM answers negated question
    - MNLI contradiction detection (offline)
    """

    def __init__(self):
        self.nli = NLIScorer()   # model already loaded once
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Module4")

    def run(self, question: str, original_answer: str) -> dict:

        if not is_factual_question(question):
            return {
                "antonym_contradiction_flag": None,
                "reason": "non_factual_question"
            }

        # Step 1: Negate question (spaCy)
        neg_question = negate_question(question)

        # Step 2: Same LLM answers negated question
        neg_answer = llm_answer(neg_question)

        # Step 3: MNLI contradiction score
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
