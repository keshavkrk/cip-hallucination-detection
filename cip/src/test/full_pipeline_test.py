# cip/src/test/full_pipeline_test.py

import numpy as np
import joblib

from llm_interface.real_llm import llm_answer
from preprocessing.module2_preprocess import module2_process
from classifier.feature_extractor import DistilBERTFeatureExtractor
from rephrase.module3.rephrase_consistency import RephraseConsistencyAnalyzer
from negation.negation_probe import NegationProbe
from fusion.fusion_layer import fuse_prediction
from explainability.lime_explainer import CIPExplainer


MODEL_PATH = "../data/processed/hallucination_model.pkl"
BACKGROUND_PATH = "../data/processed/X.npy"


def run_pipeline(question: str):

    print("\n==============================")
    print("Running Full Pipeline Test")
    print("==============================\n")

    # -------------------------------------------------
    # 1️⃣ Get LLM Answer
    # -------------------------------------------------
    answer = llm_answer(question)

    print("Question:", question)
    print("Answer:", answer)

    # -------------------------------------------------
    # 2️⃣ Build Features
    # -------------------------------------------------
    extractor = DistilBERTFeatureExtractor()
    m3 = RephraseConsistencyAnalyzer(num_paraphrases=1)
    m4 = NegationProbe()

    m2 = module2_process(question, answer)

    embedding = extractor.extract(
        m2["input_ids"],
        m2["attention_mask"]
    ).cpu().numpy().flatten()

    consistency = m3.run(question, answer)["consistency_score"]
    contradiction = m4.run(question, answer)["contradiction_score"]

    vector = np.concatenate([
        embedding,
        np.array([consistency]),
        np.array([contradiction])
    ])

    vector = vector.reshape(1, -1)

    print("\nFeature vector shape:", vector.shape)
    print("Consistency score:", consistency)
    print("Negation score:", contradiction)

    # -------------------------------------------------
    # 3️⃣ Load Model
    # -------------------------------------------------
    model = joblib.load(MODEL_PATH)

    p_model = model.predict_proba(vector)[0, 1]

    print("\nCalibrated Model Probability:", round(p_model, 4))

    # -------------------------------------------------
    # 4️⃣ Fusion
    # -------------------------------------------------
    final_risk = fuse_prediction(
        p_model,
        consistency,
        contradiction
    )

    label = "Hallucination" if final_risk > 0.5 else "Factual"

    print("Final Fused Risk:", round(final_risk, 4))
    print("Final Prediction:", label)

    # -------------------------------------------------
    # 5️⃣ LIME Explanation
    # -------------------------------------------------
    X_background = np.load(BACKGROUND_PATH)
    explainer = CIPExplainer(MODEL_PATH, X_background)

    explanation = explainer.explain_instance(vector.flatten())

    print("\nLIME Explanation (Grouped)")
    print("---------------------------------")
    print("Embedding Signal  :", round(explanation["embedding_signal"], 4))
    print("Consistency Signal:", round(explanation["consistency_signal"], 4))
    print("Negation Signal   :", round(explanation["negation_signal"], 4))


if __name__ == "__main__":

    # Try safe test first
    test_question = "if elon musk is the king of mars then who is the king of venus?"

    run_pipeline(test_question)