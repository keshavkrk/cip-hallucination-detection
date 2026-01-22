import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------
# Load model ONCE at module import
# -------------------------------------------------
_TOKENIZER = AutoTokenizer.from_pretrained("roberta-large-mnli")
_MODEL = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
_MODEL.eval()


class NLIScorer:
    """
    Independent MNLI-based contradiction scorer
    (lightweight wrapper, no reloading)
    """

    contradiction_idx = 0  # MNLI: 0 = contradiction

    @torch.no_grad()
    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        inputs = _TOKENIZER(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True
        )

        logits = _MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, self.contradiction_idx])
