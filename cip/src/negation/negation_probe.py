import logging
from llm_interface.real_llm import llm_answer
from negation.rule_negator import negate_question
from negation.nli_scorer import NLIScorer
from negation.intent_gate import is_factual_question


class NegationProbe:
    """
    Module 4: Negation Probe

    - spaCy-based negation
    - Same LLM answers negated question
    - MNLI contradiction detection
    - Guaranteed safe return structure
    """

    def __init__(self):
        self.nli = NLIScorer()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Module4")

    def run(self, question: str, original_answer: str) -> dict:

        # Default safe return
        safe_output: dict = {
            "antonym_contradiction_flag": 0,
            "contradiction_score": 0.0,
            "negated_question": None,
            "negated_answer": None,
            "reason": None
        }

        try:

            # Intent gate check
            if not is_factual_question(question):
                safe_output["reason"] = "non_factual_question"
                return safe_output

            # Step 1: Negate question
            neg_question = negate_question(question)

            # Step 2: LLM answer
            neg_answer = llm_answer(neg_question)

            # Step 3: MNLI contradiction
            score = self.nli.contradiction_score(
                premise=original_answer,
                hypothesis=neg_answer
            )

            flag = 1 if score > 0.5 else 0

            self.logger.info(f"Q     : {question}")
            self.logger.info(f"Q_neg : {neg_question}")
            self.logger.info(f"Score : {score:.4f}")

            return {
                "antonym_contradiction_flag": flag,
                "contradiction_score": score,
                "negated_question": neg_question,
                "negated_answer": neg_answer,
                "reason": None
            }

        except Exception as e:
            self.logger.warning(f"Negation probe failed: {e}")
            safe_output["reason"] = "exception"
            return safe_output