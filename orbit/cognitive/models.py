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
    """Asymmetric bilinear graph scorer with separate subject/object tables."""
    def __init__(self, num_entities: int, num_relations: int, dim: int = 96):
        super().__init__()
        self.subject = nn.Embedding(num_entities, dim)
        self.object = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        self.object_bias = nn.Embedding(num_entities, 1)
        nn.init.xavier_uniform_(self.subject.weight)
        nn.init.xavier_uniform_(self.object.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.zeros_(self.object_bias.weight)

    def forward(self, subject_ids: torch.Tensor, relation_ids: torch.Tensor):
        query = self.subject(subject_ids) * self.relation(relation_ids)
        return query @ self.object.weight.T + self.object_bias.weight.squeeze(-1)


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


class GeneratorCausalSelfAttention(nn.Module):
    """Causal self-attention for the byte-level autoregressive language model."""
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("generator dim must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(attended)


class GeneratorBlock(nn.Module):
    """Pre-norm Transformer decoder block."""
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = GeneratorCausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class NeuralLanguageGenerator(nn.Module):
    """Randomly initialised byte-level GPT: 0..255 are UTF-8 bytes; 256 is EOS."""
    vocab_size = 257
    eos_id = 256

    def __init__(self, config: dict):
        super().__init__()
        self.block_size = int(config["block_size"])
        dim = int(config["dim"])
        heads = int(config["heads"])
        layers = int(config["layers"])
        dropout = float(config.get("dropout", 0.1))
        self.token_embedding = nn.Embedding(self.vocab_size, dim)
        self.position_embedding = nn.Embedding(self.block_size, dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            GeneratorBlock(dim, heads, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.language_head = nn.Linear(dim, self.vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.block_size:
            token_ids = token_ids[:, -self.block_size:]
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)
        hidden = self.final_norm(self.blocks(hidden))
        return self.language_head(hidden)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 240,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        """Autoregressively sample new UTF-8 bytes."""
        self.eval()
        prompt_bytes = list(prompt.encode("utf-8"))[-self.block_size:]
        if not prompt_bytes:
            prompt_bytes = [32]
        tokens = torch.tensor([prompt_bytes], dtype=torch.long,
                              device=self.token_embedding.weight.device)
        generated: list[int] = []
        for step in range(max(1, int(max_new_tokens))):
            logits = self(tokens[:, -self.block_size:])[:, -1, :].float()[0]
            if step < min(8, max_new_tokens):
                logits[self.eos_id] = float("-inf")
            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / max(float(temperature), 1.0e-4)
                if top_k > 0:
                    k = min(int(top_k), logits.numel())
                    threshold = torch.topk(logits, k).values[-1]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            if next_id == self.eos_id:
                break
            generated.append(next_id)
            tokens = torch.cat((tokens, torch.tensor([[next_id]], device=tokens.device)), dim=1)
        answer = bytes(generated).decode("utf-8", errors="replace")
        for marker in ("\n<|end|>", "\n<|user|>", "\n<|assistant|>"):
            answer = answer.split(marker, 1)[0]
        return answer.strip()
