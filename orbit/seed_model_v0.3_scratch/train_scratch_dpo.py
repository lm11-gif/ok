#!/usr/bin/env python3
"""Teacher preference generation and DPO for ORBIT Seed v0.3 Scratch.

The policy and reference are both loaded from ORBIT's own randomly initialized
and scratch-trained checkpoint. SmolLM3 is only an answer-writing teacher.
"""

from __future__ import annotations

import copy
import gc
import json
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet",
        "transformers==4.53.3",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
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
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat import OrbitScratchGPT

# This stage only operates on a checkpoint created by the random-init scratch trainer.
SCRATCH_RELEASE_URL = (
    "https://github.com/lm11-gif/ok/releases/download/"
    "orbit-seed-v0.3-scratch/orbit-seed-v0.3-scratch.zip"
)
TEACHER_MODEL = "HuggingFaceTB/SmolLM3-3B"
SEED = 20261104
# The reference is a frozen copy of the scratch-trained policy before DPO.
# Reference policy is a frozen copy of this scratch-trained ORBIT checkpoint.
BETA = 0.1
DPO_EPOCHS = 2
LEARNING_RATE = 1.0e-5
MAX_NEW_TOKENS = 160
MAX_PAIRS = 96
MIN_PAIRS = 70
WORK = Path("/kaggle/working")
BASE_DIR = WORK / "orbit-seed-v0.3-scratch"
OUT = WORK / "orbit-seed-v0.3-scratch-dpo"
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
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
]

def download_scratch_checkpoint():
    import requests

    archive = WORK / "orbit-seed-v0.3-scratch.zip"
    with requests.get(SCRATCH_RELEASE_URL, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with archive.open("wb") as target:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    target.write(chunk)
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(BASE_DIR)
    required = ["config.json", "tokenizer.model", "model.safetensors"]
    for name in required:
        if not (BASE_DIR / name).is_file():
            raise RuntimeError(f"Scratch checkpoint is missing {name}")

def prompt_token_ids(sp, prompt):
    text = (
        "<|system|>You are ORBIT, a helpful and honest assistant.<|end|>"
        f"<|user|>{prompt}<|end|><|assistant|>"
    )
    return sp.encode(text, out_type=int)

def sample_student(model, sp, prompt, temperature=0.8):
    config_block = model.block_size
    ids = prompt_token_ids(sp, prompt)
    x = torch.tensor([ids[-config_block:]], dtype=torch.long, device="cuda")
    generated = []
    model.eval()
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            logits = logits[0, -1].float() / max(temperature, 1.0e-5)
            probs = torch.softmax(logits, dim=-1)
            # Nucleus sampling without an external generation dependency.
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs > 0.9
            sorted_probs[remove] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
            pick = int(torch.multinomial(sorted_probs, 1).item())
            next_id = int(sorted_indices[pick].item())
            if next_id in {sp.piece_to_id("<|end|>"), sp.eos_id()}:
                break
            generated.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device="cuda")], dim=1)
            if x.shape[1] > config_block:
                x = x[:, -config_block:]
    return sp.decode(generated).strip()

def teacher_answer(teacher, teacher_tokenizer, request, draft):
    review = (
        "/no_think\nYou are a careful senior assistant supervising ORBIT. "
        "Write the answer ORBIT should have given. Correct factual or coding mistakes, "
        "answer the actual request, be clear and proportionate, and do not invent facts. "
        "State uncertainty when needed. Output only the improved final answer.\n\n"
        f"USER REQUEST:\n{request}\n\nORBIT DRAFT:\n{draft}\n\nIMPROVED ANSWER:"
    )
    messages = [
        {
            "role": "system",
            "content": "/no_think\nYou are an accurate, careful assistant.",
        },
        {"role": "user", "content": review},
    ]
    formatted = teacher_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = teacher_tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=1024
    )
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.inference_mode():
        out = teacher.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.25,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=teacher_tokenizer.pad_token_id,
            eos_token_id=teacher_tokenizer.eos_token_id,
        )
    answer = teacher_tokenizer.decode(
        out[0, encoded["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.I | re.S)
    answer = re.sub(r"<\|[^|]+\|>", "", answer)
    return re.sub(r"\n{3,}", "\n\n", answer).strip()

def sequence_logprob(model, sp, prompt_ids, answer_ids, block_size):
    # Preserve the answer and the latest part of the prompt if context is full.
    answer_ids = answer_ids[: block_size - 2]
    keep_prompt = block_size - len(answer_ids)
    prompt_ids = prompt_ids[-keep_prompt:]
    full = prompt_ids + answer_ids
    if len(full) < 2:
        raise RuntimeError("Preference sequence too short.")
    x = torch.tensor([full[:-1]], dtype=torch.long, device="cuda")
    targets = torch.tensor([full[1:]], dtype=torch.long, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    first_answer_target = len(prompt_ids) - 1
    return token_log_probs[:, first_answer_target:].sum()

def main():
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available; enable a Kaggle GPU.")
    gpu_name = torch.cuda.get_device_name(0)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Downloading the ORBIT Seed v0.3 Scratch checkpoint...", flush=True)
    download_scratch_checkpoint()
    config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
    if not config.get("from_scratch") or config.get("pretrained_student_weights") is not False:
        raise RuntimeError("The base checkpoint is not marked as a genuine scratch model.")
    sp = spm.SentencePieceProcessor(model_file=str(BASE_DIR / "tokenizer.model"))
    block_size = int(config["block_size"])
    end_id = sp.piece_to_id("<|end|>")

    print("Loading scratch student to make rejected drafts...", flush=True)
    student = OrbitScratchGPT(config).cuda()
    student.load_state_dict(load_file(str(BASE_DIR / "model.safetensors"), device="cuda"))
    drafts = []
    prompts = PROMPTS[:MAX_PAIRS]
    for index, prompt in enumerate(prompts):
        draft = sample_student(student, sp, prompt, temperature=0.8)
        drafts.append(draft)
        if (index + 1) % 16 == 0:
            print(f"Scratch student drafts: {index + 1}/{len(prompts)}", flush=True)
    del student
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading teacher {TEACHER_MODEL}...", flush=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=True)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    teacher.eval()

    rows = []
    for index, (prompt, draft) in enumerate(zip(prompts, drafts)):
        if len(draft) < 12:
            continue
        chosen = teacher_answer(teacher, teacher_tokenizer, prompt, draft)
        rejected = draft.strip()
        if len(chosen) < 24 or len(rejected) < 12:
            continue
        if chosen.casefold() == rejected.casefold():
            continue
        if len(chosen) > 5000 or len(rejected) > 5000:
            continue
        rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
        if (index + 1) % 16 == 0:
            print(f"Teacher-reviewed pairs: {index + 1}/{len(prompts)}; kept={len(rows)}", flush=True)

    del teacher
    del teacher_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if len(rows) < MIN_PAIRS:
        raise RuntimeError(
            f"Only {len(rows)} usable preference pairs were generated; need {MIN_PAIRS}."
        )
    with (OUT / "preferences.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    policy = OrbitScratchGPT(config).cuda()
    policy.load_state_dict(load_file(str(BASE_DIR / "model.safetensors"), device="cuda"))
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    end_id = sp.piece_to_id("<|end|>")
    preference_data = []
    for row in rows:
        p_ids = prompt_token_ids(sp, row["prompt"])
        chosen_ids = sp.encode(row["chosen"], out_type=int) + [end_id]
        rejected_ids = sp.encode(row["rejected"], out_type=int) + [end_id]
        if not chosen_ids or not rejected_ids:
            continue
        preference_data.append((p_ids, chosen_ids, rejected_ids))
    if len(preference_data) < MIN_PAIRS:
        raise RuntimeError("Too few tokenized preference pairs for DPO.")

    rng = random.Random(SEED + 4)
    rng.shuffle(preference_data)
    split = max(1, int(len(preference_data) * 0.9))
    train_pairs = preference_data[:split]
    eval_pairs = preference_data[split:]

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    eval_losses = []

    for epoch in range(DPO_EPOCHS):
        rng.shuffle(train_pairs)
        accumulated = 0
        for index, (p_ids, chosen_ids, rejected_ids) in enumerate(train_pairs):
            policy_chosen = sequence_logprob(policy, sp, p_ids, chosen_ids, block_size)
            policy_rejected = sequence_logprob(policy, sp, p_ids, rejected_ids, block_size)
            with torch.no_grad():
                reference_chosen = sequence_logprob(reference, sp, p_ids, chosen_ids, block_size)
                reference_rejected = sequence_logprob(reference, sp, p_ids, rejected_ids, block_size)
            margin = (
                (policy_chosen - reference_chosen)
                - (policy_rejected - reference_rejected)
            )
            loss = -F.logsigmoid(BETA * margin).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite DPO loss at epoch {epoch + 1}, pair {index}.")
            (loss / 4.0).backward()
            losses.append(float(loss.detach().item()))
            accumulated += 1
            if accumulated == 4 or index == len(train_pairs) - 1:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            if (index + 1) % 16 == 0:
                print(
                    f"DPO epoch {epoch + 1}/{DPO_EPOCHS}: "
                    f"{index + 1}/{len(train_pairs)} pairs; "
                    f"loss={sum(losses[-16:]) / len(losses[-16:]):.4f}",
                    flush=True,
                )

    policy.eval()
    with torch.no_grad():
        for p_ids, chosen_ids, rejected_ids in eval_pairs:
            pc = sequence_logprob(policy, sp, p_ids, chosen_ids, block_size)
            pr = sequence_logprob(policy, sp, p_ids, rejected_ids, block_size)
            rc = sequence_logprob(reference, sp, p_ids, chosen_ids, block_size)
            rr = sequence_logprob(reference, sp, p_ids, rejected_ids, block_size)
            eval_losses.append(float((-F.logsigmoid(BETA * ((pc - rc) - (pr - rr)))).item()))

    smoke_prompts = prompts[:5]
    comparisons = []
    for prompt in smoke_prompts:
        sft_answer = sample_student(reference, sp, prompt, temperature=0.0)
        dpo_answer = sample_student(policy, sp, prompt, temperature=0.0)
        comparisons.append({
            "prompt": prompt,
            "sft_answer": sft_answer,
            "dpo_answer": dpo_answer,
        })

    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in policy.state_dict().items()},
        str(OUT / "model.safetensors"),
    )
    shutil.copy2(BASE_DIR / "tokenizer.model", OUT / "tokenizer.model")
    shutil.copy2(BASE_DIR / "config.json", OUT / "config.json")
    for notice in ("NOTICE.md",):
        source = BASE_DIR / notice
        if source.exists():
            shutil.copy2(source, OUT / notice)

    report = {
        "name": "ORBIT Seed v0.3 Scratch DPO",
        "from_scratch": True,
        "pretrained_student_weights": False,
        "base_checkpoint": "ORBIT Seed v0.3 Scratch",
        "teacher_model": TEACHER_MODEL,
        "teacher_role": "preference-data author only; no teacher weights copied into the student",
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "preference_pairs": len(rows),
        "train_pairs": len(train_pairs),
        "heldout_pairs": len(eval_pairs),
        "beta": BETA,
        "epochs": DPO_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "mean_training_loss": round(sum(losses) / max(1, len(losses)), 5),
        "heldout_preference_loss": round(sum(eval_losses) / max(1, len(eval_losses)), 5),
        "elapsed_seconds": round(time.time() - started, 1),
        "smoke_tests": comparisons,
        "limitations": [
            "Teacher-generated preference pairs can contain mistakes.",
            "DPO loss is not a factuality or general capability score.",
            "Inspect preferences.jsonl and compare against the SFT checkpoint.",
        ],
    }
    (OUT / "dpo_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    archive = WORK / "orbit-seed-v0.3-scratch-dpo.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=OUT)
    print("ORBIT_SCRATCH_DPO_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    print(f"ORBIT_SCRATCH_DPO_ARCHIVE={archive} bytes={archive.stat().st_size}", flush=True)

if __name__ == "__main__":
    main()
