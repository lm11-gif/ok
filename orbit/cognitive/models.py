"""Small randomly initialised specialist networks for ORBIT Cognitive v1."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Fast masked mean/max word encoder; intentionally compact for a T4."""
    def __init__(self, vocab_size: int, dim: int = 96, out_dim: int = 96, dropout: float = 0.10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.035)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = token_ids.ne(0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        mean = (embedded * mask).sum(dim=1) / denom
        maximum = embedded.masked_fill(~mask, torch.finfo(embedded.dtype).min).amax(dim=1)
        maximum = torch.where(mask.any(dim=1), maximum, torch.zeros_like(maximum))
        return self.projection(torch.cat((mean, maximum), dim=-1))


class PerceptionNet(nn.Module):
    """Predicts the subject concept and requested relation from a question."""
    def __init__(self, vocab_size: int, num_entities: int, num_relations: int, dim: int = 96):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, dim=dim, out_dim=dim)
        self.subject_head = nn.Linear(dim, num_entities)
        self.relation_head = nn.Linear(dim, num_relations)

    def forward(self, token_ids: torch.Tensor):
        hidden = self.encoder(token_ids)
        return self.subject_head(hidden), self.relation_head(hidden)


class AssociationNet(nn.Module):
    """DistMult-style graph link predictor over concept and relation IDs."""
    def __init__(self, num_entities: int, num_relations: int, dim: int = 96):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        self.object_bias = nn.Embedding(num_entities, 1)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.zeros_(self.object_bias.weight)

    def forward(self, subject_ids: torch.Tensor, relation_ids: torch.Tensor):
        query = self.entity(subject_ids) * self.relation(relation_ids)
        return query @ self.entity.weight.T + self.object_bias.weight.squeeze(-1)


class MemoryNet(nn.Module):
    """Learns to rank a fact store for a natural-language question."""
    def __init__(self, vocab_size: int, dim: int = 96):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, dim=dim, out_dim=dim)
        self.temperature = 0.07

    def encode_query(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(token_ids), dim=-1)

    def encode_facts(self, fact_token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(fact_token_ids), dim=-1)

    def forward(self, query_ids: torch.Tensor, fact_token_ids: torch.Tensor):
        query = self.encode_query(query_ids)
        facts = self.encode_facts(fact_token_ids)
        return query @ facts.T / self.temperature
