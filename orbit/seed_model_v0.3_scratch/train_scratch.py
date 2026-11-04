#!/usr/bin/env python3
"""Train ORBIT Seed v0.3 Scratch from random initialization.

No pretrained student checkpoint is loaded. FineWeb-Edu trains the base
language model; Smol-SmolTalk plus teacher-written answers teach chat behavior.
The teacher contributes text only, never model weights.
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet",
        "datasets==3.3.2",
        "sentencepiece>=0.2.0",
        "safetensors>=0.4.5",
        "transformers==4.53.3",
        "accelerate>=1.3.0",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20261104
PRETRAIN_TOKENS = 100_000_000
MAX_RAW_CHARS = 420_000_000
MAX_DOC_CHARS = 16_000
VOCAB_SIZE = 12_000
BLOCK_SIZE = 512
DIM = 448
LAYERS = 8
HEADS = 7
DROPOUT = 0.08
BATCH_SIZE = 8
GRAD_ACCUM = 4
PRETRAIN_LR = 3.0e-4
PRETRAIN_WARMUP = 150
SFT_EXAMPLES = 4_000
SFT_BATCH_SIZE = 4
SFT_GRAD_ACCUM = 2
SFT_LR = 2.0e-5
TEACHER_MODEL = "HuggingFaceTB/SmolLM3-3B"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
CHAT_DATASET = "HuggingFaceTB/smol-smoltalk"

WORK = Path("/kaggle/working")
OUT = WORK / "orbit-seed-v0.3-scratch"
OUT.mkdir(parents=True, exist_ok=True)
RAW_PATH = WORK / "fineweb-edu-sample.txt"
TRAIN_BIN = WORK / "scratch-train.bin"
VAL_BIN = WORK / "scratch-val.bin"

TEACHER_PROMPTS = [
    "Explain the difference between RAM and storage to a beginner.",
    "Why does a metal spoon feel colder than a wooden spoon in the same room?",
    "Explain how a rainbow forms using simple language.",
    "What is the difference between weather and climate?",
    "Explain recursion with a simple Python example.",
    "Give a realistic two-week plan for learning Python from zero.",
    "Explain what an API is using a restaurant analogy and then a real example.",
    "What is the difference between a Python list, tuple, and dictionary?",
    "Write a Python function that checks whether a string is a palindrome and include tests.",
    "Write a Python function that removes duplicates from a list while preserving order.",
    "Explain why a loop using while x != 10 and x += 2 might never stop.",
    "Show how to read JSON safely in Python, handling missing or invalid files.",
    "Explain the difference between a class and an object.",
    "A program works on one computer but fails on another. Give a debugging checklist.",
    "Explain what a Git commit is and how it differs from a Git push.",
    "Describe a safe way to implement autosaving in a small game.",
    "Design a simple save-file format for a single-player game.",
    "A Minecraft mod crashes because a dependency is missing. What should I check?",
    "Explain client-side versus server-side logic in a multiplayer game.",
    "Suggest a lightweight architecture for villagers that gather resources and build homes.",
    "How can a game simulate distant NPCs without keeping each one fully active?",
    "Explain A-star pathfinding in simple terms.",
    "What is a behavior tree and when is it useful?",
    "Explain the difference between an LLM, a tokenizer, and an embedding.",
    "What does quantization do to a language model?",
    "Why is a model with more parameters not automatically better at every task?",
    "Explain pretraining, supervised fine-tuning, and preference tuning.",
    "Explain Direct Preference Optimization using chosen and rejected answers.",
    "Why should an assistant admit uncertainty instead of inventing an answer?",
    "Give a good clarifying question for an ambiguous programming request.",
    "Summarize what a technical README should contain.",
    "Write a polite message explaining that a software release is delayed by a bug.",
    "Write a short story about a robot repairing an old lighthouse.",
    "Describe a quiet village during the first snowfall.",
    "Suggest five original mechanics for a cooperative survival game.",
    "Explain how to balance a powerful game ability with a long cooldown.",
    "Explain binary search and when it is useful.",
    "What is a race condition? Give a small example.",
    "Why is input validation important in a local application?",
    "Explain the principle of least privilege with a practical example.",
    "A user says an application is slow. Give a methodical profiling plan.",
    "Explain what a database index does and why too many can hurt.",
    "Explain authentication versus authorization.",
    "Give an example of an answer that honestly says code was not executed.",
    "Explain why test coverage does not prove a program is correct.",
    "What should a useful bug report contain?",
    "How can NPCs avoid all selecting the same resource node?",
    "Explain how a priority queue helps pathfinding.",
    "Explain the difference between a hard-coded rule and learned behavior.",
    "How should I test whether a new model is better than an old one?",
    "Explain data leakage in model evaluation.",
    "Why can synthetic training data amplify a teacher's mistakes?",
    "What is catastrophic forgetting and how can it be reduced?",
    "Why does low training loss not guarantee coherent answers?",
    "Explain gradient accumulation and why it helps on limited GPUs.",
    "What does learning-rate warmup do?",
    "What is the difference between validation data and test data?",
    "A model's replies became repetitive after fine-tuning. What should I inspect?",
    "Explain a minimal reproducible example for a software bug.",
    "Explain compiler errors versus runtime errors.",
    "What is a memory leak in a game loop?",
    "Why should an AI coding agent inspect project files before editing them?",
    "What should an agent do when it cannot access a required tool?",
    "Explain how to distinguish a confirmed fact from an assumption.",
    "How should an assistant respond when a user corrects a mistaken assumption?",
    "Explain backward compatibility when changing a configuration file.",
    "Why are deterministic tests useful for game AI?",
    "How can an NPC decision loop avoid running every frame?",
    "Explain local simulation versus abstract simulation for distant agents.",
    "Why are collision detection and collision response separate?",
    "How can an agent plan to gather wood, build a tool, and make a shelter?",
    "Explain breadth-first search versus depth-first search.",
    "What information should an agent log when it repeatedly fails a task?",
    "Why should a model never claim a test passed if it did not run?",
    "Explain correlation versus causation using an everyday example.",
    "Why can a laptop slow down during a long compute task?",
    "Explain thermal throttling and its effect on performance.",
    "What is a checksum and how can it detect file corruption?",
    "Why must a tokenizer match the language-model weights?",
    "What should a reproducible model training report include?",
    "Explain how to keep an AI answer concise without omitting caveats.",
    "Write a Python example that safely handles division by zero.",
    "Explain type hints in Python.",
    "What is the difference between a shallow copy and a deep copy?",
    "Explain why retries should have limits and backoff.",
    "How should an AI agent report partial completion after a failure?",
    "Explain what a model checkpoint is.",
    "Give a sensible first benchmark for a small conversational assistant.",
    "Explain precision, recall, and F1 score in simple terms.",
    "How can a game prevent multiple agents reserving the same resource?",
    "Describe a simple approach to generating a connected maze.",
    "Explain why a model can sound confident while being wrong.",
    "Write a clear guide to creating and switching to a Git branch.",
    "Explain merging versus rebasing in Git for a beginner.",
    "Describe how continuous integration catches syntax errors.",
    "Why should training scripts never print API tokens?",
    "Explain why pretraining, data generation, and evaluation should be separate stages.",
    "Suggest a folder structure for a small Python game.",
    "Write a basic Python dataclass for a 2D position.",
    "Explain why an AI coding agent should test changes rather than merely describe them.",
    "How would you debug an NPC that stops moving after reaching a doorway?",
    "Suggest a safe backup-and-restore design for user game worlds.",
    "Explain how to compare two GPUs for a specific workload.",
    "Give an example of a useful answer when there is not enough information to be certain.",
]

def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available. Enable a Kaggle GPU accelerator.")
    return torch.cuda.get_device_name(0)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collect_fineweb() -> list[str]:
    """Stream a bounded subset of FineWeb-Edu. No model weights are downloaded."""
    print(f"Streaming {DATASET_ID}/{DATASET_CONFIG}...", flush=True)
    stream = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
    ).shuffle(buffer_size=10_000, seed=SEED)
    docs: list[str] = []
    total_chars = 0
    for row in stream:
        text = row.get("text")
        if not isinstance(text, str):
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 300:
            continue
        text = text[:MAX_DOC_CHARS]
        if len(text) < 200:
            continue
        docs.append(text)
        total_chars += len(text)
        if len(docs) % 5_000 == 0:
            print(f"Downloaded {len(docs):,} docs / {total_chars:,} chars", flush=True)
        if total_chars >= MAX_RAW_CHARS:
            break
    if len(docs) < 10_000:
        raise RuntimeError(f"Too few FineWeb-Edu documents collected: {len(docs)}")
    random.Random(SEED).shuffle(docs)
    print(f"Collected {len(docs):,} docs and {total_chars:,} characters.", flush=True)
    return docs

def train_tokenizer(train_docs: list[str]) -> spm.SentencePieceProcessor:
    raw_train = WORK / "tokenizer-training-corpus.txt"
    with raw_train.open("w", encoding="utf-8") as handle:
        for doc in train_docs:
            handle.write(doc.replace("\n", " ") + "\n")
    print("Training ORBIT's own SentencePiece BPE tokenizer...", flush=True)
    spm.SentencePieceTrainer.train(
        input=str(raw_train),
        model_prefix=str(OUT / "tokenizer"),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
        max_sentence_length=16_000,
        num_threads=4,
        input_sentence_size=0,
        shuffle_input_sentence=False,
    )
    return spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer.model"))

def write_token_bin(docs: list[str], path: Path, sp: spm.SentencePieceProcessor) -> int:
    count = 0
    with path.open("wb") as handle:
        for doc in docs:
            ids = sp.encode(doc, out_type=int)
            if not ids:
                continue
            arr = np.asarray(ids + [sp.eos_id()], dtype=np.uint16)
            arr.tofile(handle)
            count += arr.size
    return count

def prepare_data():
    docs = collect_fineweb()
    val_count = max(500, int(len(docs) * 0.02))
    train_docs, val_docs = docs[:-val_count], docs[-val_count:]
    sp = train_tokenizer(train_docs)
    print(f"ORBIT tokenizer has {sp.get_piece_size():,} pieces.", flush=True)
    train_count = write_token_bin(train_docs, TRAIN_BIN, sp)
    val_count_tokens = write_token_bin(val_docs, VAL_BIN, sp)
    train_ids = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_ids = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    if len(train_ids) < 2_000_000 or len(val_ids) < 100_000:
        raise RuntimeError(
            f"Not enough tokens: train={len(train_ids):,}, validation={len(val_ids):,}"
        )
    print(
        f"Fresh tokenizer/data ready: train={len(train_ids):,} tokens; "
        f"validation={len(val_ids):,} tokens.",
        flush=True,
    )
    return sp, train_ids, val_ids

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
    def __init__(self, vocab: int):
        super().__init__()
        self.block_size = BLOCK_SIZE
        self.token_emb = nn.Embedding(vocab, DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[
            Block(DIM, HEADS, DROPOUT) for _ in range(LAYERS)
        ])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, vocab, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * LAYERS))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        if t > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            t = self.block_size
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos)[None, :, :])
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

def get_pretrain_batch(data: np.memmap):
    max_start = len(data) - BLOCK_SIZE - 2
    starts = torch.randint(0, max_start, (BATCH_SIZE,)).tolist()
    x = np.stack([np.asarray(data[s:s + BLOCK_SIZE], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(data[s + 1:s + BLOCK_SIZE + 1], dtype=np.int64) for s in starts])
    return torch.from_numpy(x).cuda(non_blocking=True), torch.from_numpy(y).cuda(non_blocking=True)

@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, batches=12):
    model.eval()
    result = {}
    for name, data in (("train", train_ids), ("validation", val_ids)):
        values = []
        for _ in range(batches):
            x, y = get_pretrain_batch(data)
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
            values.append(float(loss.item()))
        result[name] = float(sum(values) / len(values))
    model.train()
    return result

def make_teacher_answers():
    """Create assistant training targets using teacher-generated text only."""
    print(f"Loading teacher {TEACHER_MODEL} to author synthetic examples...", flush=True)
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    examples = []
    for index, prompt in enumerate(TEACHER_PROMPTS):
        messages = [
            {
                "role": "system",
                "content": "/no_think\nYou are an accurate, clear teacher. Answer the user's request directly. Do not invent facts; state uncertainty where necessary.",
            },
            {"role": "user", "content": prompt},
        ]
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tok(formatted, return_tensors="pt", truncation=True, max_length=768)
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        with torch.inference_mode():
            out = model.generate(
                **encoded,
                max_new_tokens=180,
                do_sample=True,
                temperature=0.35,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        answer = tok.decode(
            out[0, encoded["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.I | re.S).strip()
        if len(answer) >= 30:
            examples.append({"user": prompt, "assistant": answer, "source": "SmolLM3-3B synthetic teacher"})
        if (index + 1) % 16 == 0:
            print(f"Teacher examples generated: {index + 1}/{len(TEACHER_PROMPTS)}", flush=True)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    if len(examples) < 70:
        raise RuntimeError(f"Teacher generated only {len(examples)} usable examples.")
    return examples

def encode_conversation(messages, sp):
    user_id = sp.piece_to_id("<|user|>")
    system_id = sp.piece_to_id("<|system|>")
    assistant_id = sp.piece_to_id("<|assistant|>")
    end_id = sp.piece_to_id("<|end|>")
    ids, labels = [], []
    for message in messages:
        role = message.get("role")
        text = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(text, str) or not text.strip():
            continue
        if role == "system":
            ids.append(system_id); labels.append(-100)
            content = sp.encode(text.strip(), out_type=int)
            ids.extend(content); labels.extend([-100] * len(content))
            ids.append(end_id); labels.append(-100)
        elif role == "user":
            ids.append(user_id); labels.append(-100)
            content = sp.encode(text.strip(), out_type=int)
            ids.extend(content); labels.extend([-100] * len(content))
            ids.append(end_id); labels.append(-100)
        else:
            ids.append(assistant_id); labels.append(-100)
            content = sp.encode(text.strip(), out_type=int)
            ids.extend(content); labels.extend(content)
            ids.append(end_id); labels.append(end_id)
    if len(ids) < 8 or not any(x != -100 for x in labels):
        return None
    if len(ids) > BLOCK_SIZE:
        ids = ids[-BLOCK_SIZE:]
        labels = labels[-BLOCK_SIZE:]
    return {"input_ids": ids, "labels": labels}

def collate_sft(rows, pad_id: int):
    max_len = max(len(row["input_ids"]) for row in rows)
    x = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(rows), max_len), -100, dtype=torch.long)
    for i, row in enumerate(rows):
        length = len(row["input_ids"])
        x[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        y[i, :length] = torch.tensor(row["labels"], dtype=torch.long)
    return x.cuda(), y.cuda()

def teacher_sft(model, sp):
    print("Loading conversation dataset for assistant-only SFT...", flush=True)
    raw = load_dataset(CHAT_DATASET, split="train")
    raw = raw.shuffle(seed=SEED).select(range(min(SFT_EXAMPLES, len(raw))))
    examples = []
    for item in raw:
        messages = item.get("messages")
        if not isinstance(messages, list):
            continue
        cleaned = [
            m for m in messages
            if isinstance(m, dict)
            and m.get("role") in {"system", "user", "assistant"}
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        ]
        feature = encode_conversation(cleaned, sp)
        if feature:
            examples.append(feature)

    teacher_examples = make_teacher_answers()
    for row in teacher_examples:
        feature = encode_conversation(
            [
                {"role": "system", "content": "You are ORBIT, a helpful and honest assistant."},
                {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": row["assistant"]},
            ],
            sp,
        )
        if feature:
            examples.append(feature)

    if len(examples) < 1_000:
        raise RuntimeError(f"Too few assistant-only SFT examples: {len(examples)}")
    print(
        f"Assistant-only SFT examples: {len(examples):,} "
        f"(including {len(teacher_examples)} teacher-written examples).",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SFT_LR, betas=(0.9, 0.95), weight_decay=0.05
    )
    model.train()
    rng = random.Random(SEED + 1)
    order = list(range(len(examples)))
    rng.shuffle(order)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    for batch_start in range(0, len(order), SFT_BATCH_SIZE * SFT_GRAD_ACCUM):
        accum_losses = []
        for accum in range(SFT_GRAD_ACCUM):
            start = batch_start + accum * SFT_BATCH_SIZE
            selected = order[start:start + SFT_BATCH_SIZE]
            if not selected:
                continue
            x, labels = collate_sft([examples[i] for i in selected], sp.pad_id())
            inputs = x[:, :-1]
            targets = labels[:, 1:]
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = model(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SFT loss at update {updates}.")
            (loss / SFT_GRAD_ACCUM).backward()
            accum_losses.append(float(loss.item()))
        if not accum_losses:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        losses.extend(accum_losses)
        if updates % 50 == 0:
            print(
                f"SFT update {updates}; loss={sum(losses[-50:]) / len(losses[-50:]):.4f}",
                flush=True,
            )
    return {
        "examples": len(examples),
        "teacher_examples": len(teacher_examples),
        "updates": updates,
        "mean_batch_loss": round(sum(losses) / max(1, len(losses)), 4),
    }

def sample_answer(model, sp, prompt: str, max_new_tokens=120):
    prefix = (
        "<|system|>You are ORBIT, a helpful and honest assistant.<|end|>"
        f"<|user|>{prompt}<|end|><|assistant|>"
    )
    ids = sp.encode(prefix, out_type=int)
    x = torch.tensor([ids[-BLOCK_SIZE:]], dtype=torch.long, device="cuda")
    model.eval()
    generated = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(x)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if next_id in {sp.piece_to_id("<|end|>"), sp.eos_id()}:
                break
            generated.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device="cuda")], dim=1)
            if x.shape[1] > BLOCK_SIZE:
                x = x[:, -BLOCK_SIZE:]
    return sp.decode(generated).strip()

def main():
    started = time.time()
    gpu_name = require_cuda()
    set_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True

    docs = collect_fineweb()
    val_docs_count = max(500, int(len(docs) * 0.02))
    train_docs, val_docs = docs[:-val_docs_count], docs[-val_docs_count:]
    sp = train_tokenizer(train_docs)
    print(f"ORBIT tokenizer size: {sp.get_piece_size():,}", flush=True)

    train_count = write_token_bin(train_docs, TRAIN_BIN, sp)
    val_count = write_token_bin(val_docs, VAL_BIN, sp)
    train_ids = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_ids = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    if len(train_ids) < PRETRAIN_TOKENS:
        token_budget = len(train_ids) - BLOCK_SIZE * BATCH_SIZE * 10
        print(
            f"Only {len(train_ids):,} train tokens available; using {token_budget:,} for this run.",
            flush=True,
        )
    else:
        token_budget = PRETRAIN_TOKENS

    model = OrbitScratchGPT(sp.get_piece_size()).cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"Initialized ORBIT from random weights: {parameter_count:,} parameters. "
        "No pretrained student checkpoint was loaded.",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PRETRAIN_LR, betas=(0.9, 0.95), weight_decay=0.1
    )
    tokens_per_update = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM
    steps = max(1, token_budget // tokens_per_update)
    print(
        f"Pretraining {token_budget:,} tokens over {steps:,} updates; "
        f"batch={BATCH_SIZE}, sequence={BLOCK_SIZE}, accumulation={GRAD_ACCUM}.",
        flush=True,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_val = float("inf")
    train_log = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_started = time.time()
    for step in range(1, steps + 1):
        if step <= PRETRAIN_WARMUP:
            lr = PRETRAIN_LR * step / PRETRAIN_WARMUP
        else:
            progress = (step - PRETRAIN_WARMUP) / max(1, steps - PRETRAIN_WARMUP)
            lr = PRETRAIN_LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        losses = []
        for _ in range(GRAD_ACCUM):
            x, y = get_pretrain_batch(train_ids)
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite pretraining loss at step {step}.")
            scaler.scale(loss).backward()
            losses.append(float(loss.item()) * GRAD_ACCUM)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 250 == 0 or step == steps:
            metrics = estimate_loss(model, train_ids, val_ids, batches=8)
            elapsed = time.time() - run_started
            row = {
                "step": step,
                "tokens_seen": step * tokens_per_update,
                "train_loss": round(metrics["train"], 4),
                "validation_loss": round(metrics["validation"], 4),
                "learning_rate": lr,
                "elapsed_seconds": round(elapsed, 1),
            }
            train_log.append(row)
            print(json.dumps(row), flush=True)
            if metrics["validation"] < best_val:
                best_val = metrics["validation"]
                save_file(
                    {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
                    str(OUT / "model.safetensors"),
                )
                (OUT / "config.json").write_text(
                    json.dumps({
                        "model_name": "ORBIT Seed v0.3 Scratch",
                        "version": "0.3-scratch",
                        "architecture": "decoder-only causal Transformer",
                        "from_scratch": True,
                        "pretrained_student_weights": False,
                        "vocab_size": sp.get_piece_size(),
                        "block_size": BLOCK_SIZE,
                        "dim": DIM,
                        "layers": LAYERS,
                        "heads": HEADS,
                        "dropout": DROPOUT,
                        "parameters": parameter_count,
                        "initialization": "random normal; tied token/output embeddings",
                        "seed": SEED,
                    }, indent=2),
                    encoding="utf-8",
                )
                (OUT / "pretraining_log.json").write_text(
                    json.dumps(train_log, indent=2), encoding="utf-8"
                )

    # Restore the best validation checkpoint before instruction tuning.
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(OUT / "model.safetensors"), device="cuda"))
    sft_report = teacher_sft(model, sp)

    prompts = [
        "Hi! What can you help me with?",
        "Explain why the sky looks blue in simple terms.",
        "Write a Python function that returns the factorial of a non-negative integer.",
        "A game has too many active NPCs and performance is dropping. Suggest a practical design.",
        "What should you do when you are not sure an answer is correct?",
    ]
    generations = []
    for prompt in prompts:
        answer = sample_answer(model, sp, prompt)
        generations.append({"prompt": prompt, "answer": answer})
        print(f"SMOKE PROMPT: {prompt}\nORBIT: {answer}\n", flush=True)

    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
        str(OUT / "model.safetensors"),
    )
    report = {
        "name": "ORBIT Seed v0.3 Scratch",
        "from_scratch": True,
        "pretrained_student_weights": False,
        "student_initialization": "random",
        "teacher_model": TEACHER_MODEL,
        "teacher_role": "synthetic text generation only; no teacher weights copied into ORBIT",
        "pretraining_dataset": DATASET_ID,
        "pretraining_config": DATASET_CONFIG,
        "pretraining_license": "ODC-By 1.0",
        "instruction_dataset": CHAT_DATASET,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameters": parameter_count,
        "vocabulary_size": sp.get_piece_size(),
        "context_length": BLOCK_SIZE,
        "pretraining_tokens_processed": min(token_budget, steps * tokens_per_update),
        "pretraining_updates": steps,
        "best_validation_loss": best_val,
        "pretraining_log": train_log,
        "instruction_tuning": sft_report,
        "elapsed_seconds": round(time.time() - started, 1),
        "smoke_test_generations": generations,
        "limitations": [
            "This is a small experimental model, not comparable to multi-billion-parameter pretrained assistants.",
            "A single 100M-token run is an initial training phase; further data and compute would improve it.",
            "The teacher contributes generated text only; ORBIT's weights are initialized randomly.",
        ],
    }
    (OUT / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "NOTICE.md").write_text(
        "# ORBIT Seed v0.3 Scratch notices\n\n"
        "ORBIT student weights were initialized randomly and trained from scratch.\n\n"
        "Pretraining text: HuggingFaceFW/fineweb-edu, sample-10BT, ODC-By 1.0: "
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu\n\n"
        "Conversation data: HuggingFaceTB/smol-smoltalk: "
        "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk\n\n"
        "Teacher used only to generate additional text targets: "
        "HuggingFaceTB/SmolLM3-3B, Apache-2.0: "
        "https://huggingface.co/HuggingFaceTB/SmolLM3-3B\n\n"
        "No teacher weights, embeddings, or pretrained student checkpoint are used.\n",
        encoding="utf-8",
    )
    archive = WORK / "orbit-seed-v0.3-scratch.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=OUT)
    print("ORBIT_SCRATCH_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    print(f"ORBIT_SCRATCH_ARCHIVE={archive} bytes={archive.stat().st_size}", flush=True)

if __name__ == "__main__":
    main()
