"""Word-piece language model used by ORBIT Cognitive v2.3 Conversational.

The tokenizer learns frequent whole words and punctuation from the local training
corpus, while retaining UTF-8 byte fallback for words it has never seen.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

TOKEN_PATTERN = re.compile(
    r"<\|[^>]+\|>|\w+(?:['’]\w+)*|[^\w\s]|\s+",
    flags=re.UNICODE,
)


class VocabularyTokenizer:
    """Corpus-trained word/punctuation vocabulary with lossless UTF-8 fallback."""

    PAD_ID = 256
    EOS_ID = 257
    UNK_ID = 258
    TOKEN_OFFSET = 259
    FORMAT = "orbit-wordpiece-byte-fallback-v1"

    def __init__(self, tokens: list[str]):
        if len(tokens) != len(set(tokens)):
            raise ValueError("Tokenizer vocabulary contains duplicate pieces.")
        self.tokens = list(tokens)
        self.token_to_id = {
            token: self.TOKEN_OFFSET + index
            for index, token in enumerate(self.tokens)
        }
        self.vocab_size = self.TOKEN_OFFSET + len(self.tokens)

    @classmethod
    def build(cls, texts: list[str], max_vocab: int = 8192,
              min_frequency: int = 2) -> "VocabularyTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(TOKEN_PATTERN.findall(str(text)))
        limit = max(256, int(max_vocab) - cls.TOKEN_OFFSET)
        pieces = [
            token for token, count in counts.most_common()
            if count >= max(1, int(min_frequency))
        ][:limit]
        return cls(pieces)

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        for piece in TOKEN_PATTERN.findall(str(text)):
            token_id = self.token_to_id.get(piece)
            if token_id is not None:
                result.append(token_id)
            else:
                result.extend(piece.encode("utf-8"))
        return result

    def decode(self, token_ids: list[int] | tuple[int, ...],
               stop_at_chat_marker: bool = False) -> str:
        pieces: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes():
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        stop_tokens = {"<|end|>", "<|context|>", "<|system|>"}
        if stop_at_chat_marker:
            stop_tokens.update({"<|user|>", "<|assistant|>", "<|context|>", "<|system|>"})
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id == self.EOS_ID or token_id == self.PAD_ID:
                break
            if 0 <= token_id < 256:
                byte_buffer.append(token_id)
                continue
            flush_bytes()
            if token_id == self.UNK_ID:
                pieces.append("�")
                continue
            index = token_id - self.TOKEN_OFFSET
            if 0 <= index < len(self.tokens):
                piece = self.tokens[index]
                if piece in stop_tokens:
                    break
                pieces.append(piece)
        flush_bytes()
        return "".join(pieces).strip()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "format": self.FORMAT,
            "tokens": self.tokens,
            "pad_id": self.PAD_ID,
            "eos_id": self.EOS_ID,
            "unk_id": self.UNK_ID,
            "token_offset": self.TOKEN_OFFSET,
            "vocab_size": self.vocab_size,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.FORMAT:
            raise ValueError("Unsupported ORBIT tokenizer format.")
        tokenizer = cls(payload["tokens"])
        if int(payload.get("vocab_size", tokenizer.vocab_size)) != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match its token list.")
        return tokenizer

    def token_id(self, token: str) -> int | None:
        return self.token_to_id.get(token)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("Model width must be divisible by attention heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        query, key, value = self.qkv(hidden).split(width, dim=-1)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        result = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        result = result.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(result)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.norm1(hidden))
        return hidden + self.feed_forward(self.norm2(hidden))


class NeuralLanguageGenerator(nn.Module):
    """Randomly initialised causal Transformer over learned word and punctuation units."""

    def __init__(self, config: dict):
        super().__init__()
        self.block_size = int(config["block_size"])
        self.vocab_size = int(config["vocab_size"])
        self.pad_id = int(config.get("pad_id", VocabularyTokenizer.PAD_ID))
        self.eos_id = int(config.get("eos_id", VocabularyTokenizer.EOS_ID))
        dim = int(config["dim"])
        heads = int(config["heads"])
        layers = int(config["layers"])
        dropout = float(config.get("dropout", 0.1))
        self.token_embedding = nn.Embedding(
            self.vocab_size, dim, padding_idx=self.pad_id
        )
        self.position_embedding = nn.Embedding(self.block_size, dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            DecoderBlock(dim, heads, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.language_head = nn.Linear(dim, self.vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_id].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence].")
        if token_ids.shape[1] > self.block_size:
            token_ids = token_ids[:, -self.block_size:]
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        hidden = self.embedding_dropout(hidden)
        hidden = self.final_norm(self.blocks(hidden))
        return self.language_head(hidden)

    @torch.inference_mode()
    def generate(self, prompt: str, tokenizer: VocabularyTokenizer,
                 max_new_tokens: int = 160, temperature: float = 0.72,
                 top_k: int = 40, top_p: float = 0.92) -> str:
        """Generate subword pieces; recent user context is retained on truncation."""
        self.eval()
        prompt_ids = tokenizer.encode(prompt)[-self.block_size:]
        if not prompt_ids:
            prompt_ids = [32]
        tokens = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.token_embedding.weight.device
        )
        generated: list[int] = []
        stop_ids = {
            tokenizer.token_id("<|end|>"),
            tokenizer.token_id("<|user|>"),
            tokenizer.token_id("<|assistant|>"),
            tokenizer.token_id("<|context|>"),
            tokenizer.token_id("<|system|>"),
        }
        stop_ids.discard(None)
        for step in range(max(1, int(max_new_tokens))):
            logits = self(tokens[:, -self.block_size:])[:, -1, :].float()[0]
            logits[self.pad_id] = float("-inf")
            if step < 3:
                logits[self.eos_id] = float("-inf")
                for stop_id in stop_ids:
                    logits[stop_id] = float("-inf")
            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / max(float(temperature), 1.0e-4)
                if top_k > 0:
                    count = min(int(top_k), logits.numel())
                    threshold = torch.topk(logits, count).values[-1]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float("-inf")
                    filtered = torch.full_like(logits, float("-inf"))
                    filtered.scatter_(0, sorted_indices, sorted_logits)
                    logits = filtered
                probabilities = torch.softmax(logits, dim=-1)
                if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
                    next_id = int(torch.argmax(logits).item())
                else:
                    next_id = int(torch.multinomial(probabilities, 1).item())
            if next_id == self.eos_id or next_id in stop_ids:
                break
            generated.append(next_id)
            tokens = torch.cat((
                tokens,
                torch.tensor([[next_id]], dtype=torch.long, device=tokens.device),
            ), dim=1)
        return tokenizer.decode(generated, stop_at_chat_marker=True)
