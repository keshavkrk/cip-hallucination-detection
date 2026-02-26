from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from rephrase.module3.rephraser import rephrase_question
from llm_interface.real_llm import llm_answer


class RephraseConsistencyAnalyzer:
    """
    Module 3: Rephrase Consistency Analyzer

    - No intent gate
    - Always attempts paraphrasing
    - Always returns float consistency_score
    - Exception safe
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

        try:
            # Step 1: Generate paraphrases
            paraphrases = rephrase_question(question, k=self.k)

            if not paraphrases:
                return {
                    "consistency_score": 0.0,
                    "paraphrases": [],
                    "rephrased_answers": [],
                    "reason": "no_paraphrases_generated"
                }

            scores = []
            answers = []

            # Step 2: Re-query LLM for each paraphrase
            for q_re in paraphrases:
                try:
                    a_re = llm_answer(q_re)
                    sim = self._embed_similarity(original_answer, a_re)

                    scores.append(sim)
                    answers.append(a_re)

                except Exception:
                    continue

            # If similarity failed completely
            if not scores:
                return {
                    "consistency_score": 0.0,
                    "paraphrases": paraphrases,
                    "rephrased_answers": answers,
                    "reason": "similarity_failed"
                }

            # Step 3: Aggregate similarity
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
                "rephrased_answers": answers,
                "reason": None
            }

        except Exception as e:
            self.logger.warning(f"Consistency module failed: {e}")
            return {
                "consistency_score": 0.0,
                "paraphrases": [],
                "rephrased_answers": [],
                "reason": "exception"
            }