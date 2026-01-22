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
def module2_process(question: str) -> dict:
    """
    Module 2: Text Preprocessing

    Input:
        - question (string) from user / GUI

    Output:
        {
          question: cleaned question,
          answer: mock LLM answer,
          qa_text: combined QA text,
          input_ids: tensor [1, 512],
          attention_mask: tensor [1, 512]
        }
    """

    # ---------------------------
    # Step 2.0 – Get LLM answer
    # ---------------------------
    raw_answer = llm_answer(question)

    # ---------------------------
    # Step 2.1 – Clean text
    # ---------------------------
    question_clean = clean_text(question)
    answer_clean = clean_text(raw_answer)

    # ---------------------------
    # Step 2.2 – QA pair formation (CRITICAL)
    # DO NOT add [CLS] manually
    # ---------------------------
    qa_text = question_clean + " [SEP] " + answer_clean

    # ---------------------------
    # Step 2.4 – Tokenization
    # ---------------------------
    encoded = tokenizer(
        qa_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    # ---------------------------
    # Step 2.7 – Return tensors
    # ---------------------------
    return {
        "question": question_clean,
        "answer": answer_clean,
        "qa_text": qa_text,
        "input_ids": encoded["input_ids"],          # [1, 512]
        "attention_mask": encoded["attention_mask"] # [1, 512]
    }
