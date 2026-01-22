from preprocessing.module2_preprocess import module2_process
from rephrase.module3.rephrase_consistency import RephraseConsistencyAnalyzer
from negation.negation_probe import NegationProbe


def test_full_pipeline(question: str):
    print("\n========== PIPELINE TEST ==========\n")

    # ---------------------------
    # Module 2
    # ---------------------------
    print("▶ Running Module 2 (Preprocessing)")
    m2_out = module2_process(question)

    print("Question :", m2_out["question"])
    print("Answer   :", m2_out["answer"])
    print("Input IDs shape :", m2_out["input_ids"].shape)

    # ---------------------------
    # Module 3
    # ---------------------------
    print("\n▶ Running Module 3 (Rephrase Consistency)")
    module3 = RephraseConsistencyAnalyzer()
    m3_out = module3.run(
        question=m2_out["question"],
        original_answer=m2_out["answer"]
    )

    print("Consistency score :", m3_out.get("consistency_score"))
    print("Rephrased answer  :", m3_out.get("rephrased_answer"))

    # ---------------------------
    # Module 4
    # ---------------------------
    print("\n▶ Running Module 4 (Negation Probe)")
    module4 = NegationProbe()
    m4_out = module4.run(
        question=m2_out["question"],
        original_answer=m2_out["answer"]
    )

    print("Contradiction flag :", m4_out.get("antonym_contradiction_flag"))
    print("Negated answer     :", m4_out.get("negated_answer"))

    print("\n========== END TEST ==========\n")


if __name__ == "__main__":
    test_full_pipeline("Who invented the telephone?")
