import re
from transformers import DistilBertTokenizerFast
from llm_interface.mock_llm import llm_answer


# -------------------------------------------------
# Step 2.3 – Load tokenizer (ONCE per program)
# -------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)


# -------------------------------------------------
# Step 2.1 – Basic text cleaning
# -------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans input text by:
    - stripping leading/trailing spaces
    - normalizing multiple spaces
    - removing non-printable characters
    - keeping punctuation (important for meaning)
    """
    # Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())

    # Strip leading/trailing spaces
    text = text.strip()

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text


# -------------------------------------------------
# Module 2 main entry point
# -------------------------------------------------
def module2_process(question: str, answer: str | None = None) -> dict:
    """
    Module 2: Text Preprocessing

    If answer is None → calls LLM (inference mode)
    If answer is provided → uses given answer (training mode)
    """

    # ---------------------------
    # Step 1 – Get answer if needed
    # ---------------------------
    if answer is None:
        raw_answer = llm_answer(question)
    else:
        raw_answer = answer

    # ---------------------------
    # Step 2 – Clean text
    # ---------------------------
    question_clean = clean_text(question)
    answer_clean = clean_text(raw_answer)

    # ---------------------------
    # Step 3 – QA pair formation
    # ---------------------------
    qa_text = question_clean + " [SEP] " + answer_clean

    # ---------------------------
    # Step 4 – Tokenization
    # ---------------------------
    encoded = tokenizer(
        qa_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    return {
        "question": question_clean,
        "answer": answer_clean,
        "qa_text": qa_text,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }
