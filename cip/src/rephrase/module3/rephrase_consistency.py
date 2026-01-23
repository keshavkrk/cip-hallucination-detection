from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from rephrase.module3.rephraser import rephrase_question
from llm_interface.real_llm import llm_answer


# ---------------------------
# Intent gate
# ---------------------------
def is_factual(question: str) -> bool:
    q = question.lower().strip()
    return q.startswith(("who", "what", "when", "where", "which"))


class RephraseConsistencyAnalyzer:
    """
    Module 3: Rephrase Consistency Analyzer (100%)

    - Real LLM paraphrasing
    - k paraphrases
    - Semantic consistency via SBERT
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        num_paraphrases: int = 3,
        enable_logging: bool = True
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.k = num_paraphrases

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

        if not is_factual(question):
            return {
                "consistency_score": None,
                "reason": "non_factual_question"
            }

        # Step 1: Generate paraphrases
        paraphrases = rephrase_question(question, k=self.k)

        if len(paraphrases) == 0:
            return {
                "consistency_score": 0.0,
                "reason": "no_paraphrases_generated"
            }

        scores = []
        answers = []

        # Step 2: Re-query LLM for each paraphrase
        for q_re in paraphrases:
            a_re = llm_answer(q_re)
            sim = self._embed_similarity(original_answer, a_re)

            scores.append(sim)
            answers.append(a_re)

        # Step 3: Aggregate similarity (mean)
        final_score = sum(scores) / len(scores)

        # Logging
        self.logger.info("Module 3 executed successfully")
        self.logger.info(f"Q_original : {question}")
        self.logger.info(f"Paraphrases: {paraphrases}")
        self.logger.info(f"Scores     : {[round(s,4) for s in scores]}")
        self.logger.info(f"FinalScore : {final_score:.4f}")

        return {
            "consistency_score": final_score,
            "paraphrases": paraphrases,
            "rephrased_answers": answers
        }
