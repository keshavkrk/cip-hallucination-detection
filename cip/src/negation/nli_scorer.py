import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------
# Device Setup (auto GPU if available)
# -------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load model ONCE at module import
# -------------------------------------------------
_MODEL_NAME = "roberta-large-mnli"

_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
_MODEL = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)

_MODEL.to(_DEVICE)
_MODEL.eval()


class NLIScorer:
    """
    MNLI-based NLI scorer.

    Backward compatible:
    - contradiction_score() remains unchanged.

    Added:
    - entailment_score()
    - neutral_score()
    - full_scores()  (single forward pass for efficiency)
    """

    # MNLI label mapping
    # 0 = contradiction
    # 1 = neutral
    # 2 = entailment
    contradiction_idx = 0
    neutral_idx = 1
    entailment_idx = 2

    @torch.no_grad()
    def _predict(self, premise: str, hypothesis: str):
        """
        Internal forward pass.
        Returns softmax probabilities tensor of shape (3,)
        """

        inputs = _TOKENIZER(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

        logits = _MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        return probs[0]  # shape: (3,)

    # -------------------------------------------------
    # Backward Compatible Method
    # -------------------------------------------------
    @torch.no_grad()
    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        probs = self._predict(premise, hypothesis)
        return float(probs[self.contradiction_idx].cpu())

    # -------------------------------------------------
    # New Methods
    # -------------------------------------------------
    @torch.no_grad()
    def entailment_score(self, premise: str, hypothesis: str) -> float:
        probs = self._predict(premise, hypothesis)
        return float(probs[self.entailment_idx].cpu())

    @torch.no_grad()
    def neutral_score(self, premise: str, hypothesis: str) -> float:
        probs = self._predict(premise, hypothesis)
        return float(probs[self.neutral_idx].cpu())

    @torch.no_grad()
    def full_scores(self, premise: str, hypothesis: str) -> dict:
        """
        Efficient method — single forward pass.
        Recommended for training pipeline.
        """
        probs = self._predict(premise, hypothesis)

        return {
            "contradiction": float(probs[self.contradiction_idx].cpu()),
            "neutral": float(probs[self.neutral_idx].cpu()),
            "entailment": float(probs[self.entailment_idx].cpu())
        }