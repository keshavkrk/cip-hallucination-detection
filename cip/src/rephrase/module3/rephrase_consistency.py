from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from rephrase.module3.rephraser import rephrase_question
from llm_interface.mock_llm import llm_answer


# ---------------------------
# Simple intent gate (stub)
# ---------------------------
def is_factual(question: str) -> bool:
    """
    Intent gate.
    For 30%: simple heuristic.
    For 100%: replace with intent classifier.
    """
    question = question.lower()
    factual_starts = ("who", "what", "when", "where", "which")
    return question.startswith(factual_starts)


class RephraseConsistencyAnalyzer:
    """
    Module 3: Rephrase Consistency Analyzer

    Produces a soft semantic consistency score ∈ [0,1].
    Does NOT make a hallucination decision.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_logging: bool = True
    ):
        self.embedder = SentenceTransformer(embedding_model)

        if enable_logging:
            logging.basicConfig(level=logging.INFO)

        self.logger = logging.getLogger("Module3-RephraseConsistency")

    # ---------------------------
    # Embedding similarity
    # ---------------------------
    def _embed_similarity(self, a: str, b: str) -> float:
        e1 = self.embedder.encode([a])
        e2 = self.embedder.encode([b])
        return float(cosine_similarity(e1, e2)[0][0])

    # ---------------------------
    # Main entry point
    # ---------------------------
    def run(self, question: str, original_answer: str) -> dict:
        """
        Inputs come directly from Module 2:
        - question (cleaned)
        - original_answer (from mock LLM)

        Output:
        - consistency_score ∈ [0,1] OR None
        """

        # ---------------------------
        # Intent gating
        # ---------------------------
        if not is_factual(question):
            self.logger.info("Non-factual question detected. Skipping Module 3.")
            return {
                "consistency_score": None,
                "reason": "non_factual_question"
            }

        # ---------------------------
        # Rephrase question
        # ---------------------------
        q_re = rephrase_question(question)

        if not q_re or q_re.strip() == "":
            self.logger.warning("Empty rephrased question.")
            return {
                "consistency_score": 0.0,
                "reason": "empty_rephrase"
            }

        # ---------------------------
        # Re-query same LLM
        # ---------------------------
        a_re = llm_answer(q_re)

        if not a_re or a_re.strip() == "":
            self.logger.warning("Empty rephrased answer from LLM.")
            return {
                "consistency_score": 0.0,
                "reason": "empty_llm_answer"
            }

        # ---------------------------
        # Similarity computation
        # ---------------------------
        score = self._embed_similarity(original_answer, a_re)

        # ---------------------------
        # Logging (audit-friendly)
        # ---------------------------
        self.logger.info("Module 3 executed successfully")
        self.logger.info(f"Q_original  : {question}")
        self.logger.info(f"Q_rephrased : {q_re}")
        self.logger.info(f"A_original  : {original_answer}")
        self.logger.info(f"A_rephrased : {a_re}")
        self.logger.info(f"Score       : {score:.4f}")

        return {
            "consistency_score": score,
            "original_answer": original_answer,
            "rephrased_answer": a_re
        }
