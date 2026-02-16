# cip/src/classifier/feature_extractor.py

import torch
import torch.nn.functional as F
from transformers import DistilBertModel


class DistilBERTFeatureExtractor:
    """
    Module 5: Transformer Feature Extractor

    - Uses distilbert-base-uncased
    - Frozen model (no training)
    - Returns L2-normalized CLS embedding (768-d)
    """

    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device("cpu")

        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, input_ids, attention_mask):
        """
        Inputs:
            input_ids: [1, seq_len]
            attention_mask: [1, seq_len]

        Returns:
            embedding: tensor shape [1, 768]
        """

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token is first token
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # L2 normalize (important for stability)
        normalized = F.normalize(cls_embedding, p=2, dim=1)

        return normalized
