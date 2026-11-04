#!/usr/bin/env python3
"""Local chat runner for ORBIT Seed v0.3 Scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class OrbitScratchGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config["block_size"]
        dim = config["dim"]
        heads = config["heads"]
        dropout = config["dropout"]
        vocab = config["vocab_size"]
        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(self.block_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(dim, heads, dropout) for _ in range(config["layers"])
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        t = idx.shape[1]
        if t > self.block_size:
            idx = idx[:, -self.block_size:]
            t = self.block_size
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos)[None, :, :])
        x = self.ln_f(self.blocks(x))
        return self.lm_head(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model_dir = args.model.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = OrbitScratchGPT(config).to(device)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=device))
    model.eval()

    system = "You are ORBIT, a helpful and honest assistant. Answer clearly and stay on topic."
    system_ids = tokenizer.encode(
        "<|system|>" + system + "<|end|>",
        out_type=int,
    )
    user_id = tokenizer.piece_to_id("<|user|>")
    assistant_id = tokenizer.piece_to_id("<|assistant|>")
    end_id = tokenizer.piece_to_id("<|end|>")
    print(
        f"ORBIT Seed v0.3 Scratch | {config['parameters']:,} parameters | "
        f"{device} | type /exit to quit"
    )

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/exit", "exit", "quit"}:
            break
        if not prompt:
            continue
        ids = system_ids + [user_id] + tokenizer.encode(prompt, out_type=int) + [end_id, assistant_id]
        ids = ids[-config["block_size"]:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        generated = []
        with torch.inference_mode():
            for _ in range(max(1, args.max_new_tokens)):
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(x)
                else:
                    logits = model(x)
                next_logits = logits[0, -1].float()
                if args.temperature <= 0:
                    next_id = int(torch.argmax(next_logits).item())
                else:
                    next_logits = next_logits / max(args.temperature, 1.0e-5)
                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                if next_id in {end_id, tokenizer.eos_id()}:
                    break
                generated.append(next_id)
                next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
                x = torch.cat([x, next_tensor], dim=1)
                if x.shape[1] > config["block_size"]:
                    x = x[:, -config["block_size"]:]
        print("\nORBIT: " + tokenizer.decode(generated).strip())

if __name__ == "__main__":
    main()
