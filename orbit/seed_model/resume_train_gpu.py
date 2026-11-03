"""Continue ORBIT Seed from its existing checkpoint, using CUDA when available.

This script never initializes a new model. It loads model.safetensors and, when
present, optimizer_state.pt. It is suitable for Kaggle/Colab GPU notebooks.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=100, help="Additional optimizer updates")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seq-len", type=int, default=160)
parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
args = parser.parse_args()

if args.steps < 1 or args.batch_size < 1 or args.seq_len < 1:
    parser.error("steps, batch-size and seq-len must be positive")

SEED = 73421
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but torch.cuda.is_available() is False. Enable a GPU accelerator.")
DEVICE = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

config_path = ROOT / "config.json"
weights_path = ROOT / "model.safetensors"
tokenizer_path = ROOT / "tokenizer.model"
train_path = ROOT / "train_corpus.txt"
valid_path = ROOT / "validation_corpus.txt"
for required in (config_path, weights_path, tokenizer_path, train_path, valid_path):
    if not required.exists():
        raise FileNotFoundError(f"Required ORBIT file is missing: {required.name}")

config = json.loads(config_path.read_text(encoding="utf-8"))
sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).split(dim, dim=2)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.dropout(self.proj(out))

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class OrbitGPT(nn.Module):
    def __init__(self):
        super().__init__()
        dim = int(config["dim"])
        self.block_size = int(config["block_size"])
        self.token_emb = nn.Embedding(int(config["vocab_size"]), dim)
        self.pos_emb = nn.Embedding(self.block_size, dim)
        self.drop = nn.Dropout(float(config["dropout"]))
        self.blocks = nn.Sequential(*[
            Block(dim, int(config["heads"]), float(config["dropout"]))
            for _ in range(int(config["layers"]))
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, int(config["vocab_size"]), bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.size(1) > self.block_size:
            tokens = tokens[:, -self.block_size:]
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.drop(self.token_emb(tokens) + self.pos_emb(positions)[None, :, :])
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            targets = targets[:, -tokens.size(1):]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

def encode_file(path: Path) -> torch.Tensor:
    documents = [part.strip() for part in path.read_text(encoding="utf-8").split("\n\n") if part.strip()]
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(sp.encode(document, out_type=int))
        token_ids.append(sp.eos_id())
    if len(token_ids) <= args.seq_len + 1:
        raise ValueError(f"Not enough tokens in {path.name} for sequence length {args.seq_len}")
    return torch.tensor(token_ids, dtype=torch.long, device=DEVICE)

train_data = encode_file(train_path)
valid_data = encode_file(valid_path)

model = OrbitGPT()
model.load_state_dict(load_file(str(weights_path), device="cpu"))
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

state_path = ROOT / "optimizer_state.pt"
log_path = ROOT / "training_log.json"
current_step = 0
if state_path.exists():
    saved = torch.load(state_path, map_location=DEVICE, weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    current_step = int(saved.get("step", 0))
else:
    if log_path.exists():
        history = json.loads(log_path.read_text(encoding="utf-8"))
        current_step = max((int(row.get("step", 0)) for row in history), default=0)
    print(f"No optimizer state found; continuing weights from step {current_step} with a fresh optimizer.", flush=True)

batch_size = args.batch_size
seq_len = min(args.seq_len, int(config["block_size"]))

def get_batch(data: torch.Tensor):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), device=DEVICE)
    offsets = torch.arange(seq_len, device=DEVICE)
    x = data[starts[:, None] + offsets[None, :]]
    y = data[starts[:, None] + offsets[None, :] + 1]
    return x, y

@torch.no_grad()
def evaluate() -> float:
    model.eval()
    values = []
    for _ in range(8):
        x, y = get_batch(valid_data)
        _, loss = model(x, y)
        values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)

print({
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "existing_step": current_step,
    "additional_steps": args.steps,
    "batch_size": batch_size,
    "sequence_length": seq_len,
    "train_tokens": len(train_data),
    "validation_tokens": len(valid_data),
}, flush=True)

model.train()
start = time.time()
loss_total = 0.0
for index in range(args.steps):
    step = current_step + index + 1
    progress = (index + 1) / args.steps
    lr = args.learning_rate * (0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * progress)))
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_total += float(loss.item())

validation_loss = evaluate()
train_average = loss_total / args.steps
save_file({key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}, str(weights_path))
optimizer_state = optimizer.state_dict()
for state in optimizer_state["state"].values():
    for key, value in list(state.items()):
        if torch.is_tensor(value):
            state[key] = value.detach().cpu()
torch.save({"step": current_step + args.steps, "optimizer": optimizer_state}, state_path)

history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
record = {
    "step": current_step + args.steps,
    "chunk_train_loss": round(train_average, 4),
    "val_loss": round(validation_loss, 4),
    "chunk_seconds": round(time.time() - start, 1),
    "device": str(DEVICE),
    "batch_size": batch_size,
    "sequence_length": seq_len,
}
history.append(record)
log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
print(record, flush=True)
print("Updated model.safetensors and optimizer_state.pt", flush=True)
