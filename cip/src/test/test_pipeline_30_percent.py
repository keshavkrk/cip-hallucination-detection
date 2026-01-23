from preprocessing.module2_preprocess import module2_process
from rephrase.module3.rephrase_consistency import RephraseConsistencyAnalyzer
from negation.negation_probe import NegationProbe


def test_full_pipeline(question: str):
    print("\n========== FULL PIPELINE TEST ==========\n")

    # ---------------------------
    # Module 2: Preprocessing + LLM answer
    # ---------------------------
    print("▶ Module 2: Preprocessing + LLM Answer")
    m2_out = module2_process(question)

    print("Question :", m2_out["question"])
    print("Answer   :", m2_out["answer"])
    print("Input IDs shape :", m2_out["input_ids"].shape)

    # ---------------------------
    # Module 3: Rephrase Consistency
    # ---------------------------
    print("\n▶ Module 3: Rephrase Consistency")
    m3 = RephraseConsistencyAnalyzer()

    m3_out = m3.run(
        question=m2_out["question"],
        original_answer=m2_out["answer"]
    )

    print("Consistency score :", m3_out["consistency_score"])
    print("Paraphrases       :", m3_out.get("paraphrases"))

    # ---------------------------
    # Module 4: Negation Probe
    # ---------------------------
    print("\n▶ Module 4: Negation Probe")
    m4 = NegationProbe()

    m4_out = m4.run(
        question=m2_out["question"],
        original_answer=m2_out["answer"]
    )

    print("Contradiction flag :", m4_out["antonym_contradiction_flag"])
    print("Contradiction score:", m4_out["contradiction_score"])
    print("Negated answer     :", m4_out["negated_answer"])

    print("\n========== END ==========\n")


if __name__ == "__main__":
    test_full_pipeline("Who invented the telephone?")
