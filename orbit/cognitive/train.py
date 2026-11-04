#!/usr/bin/env python3
"""Train ORBIT Cognitive v2 networks, including a scratch-initialised language generator.

GPU is mandatory by default. Use --device cpu only for CI/smoke tests.
All generated training examples are based on the versioned seed knowledge graph.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import urllib.request
from contextlib import nullcontext
import random
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .core import build_seed_data, encode_examples, encode_facts, encode_text, QUESTION_TEMPLATES
    from .models import PerceptionNet, AssociationNet, MemoryNet
    from .language import VocabularyTokenizer, NeuralLanguageGenerator
except ImportError:  # Kaggle script mode: modules are side-by-side.
    from core import build_seed_data, encode_examples, encode_facts, encode_text, QUESTION_TEMPLATES
    from models import PerceptionNet, AssociationNet, MemoryNet
    from language import VocabularyTokenizer, NeuralLanguageGenerator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output", default="artifacts/orbit-cognitive-v2.2")
    parser.add_argument("--epochs-perception", type=int, default=35)
    parser.add_argument("--epochs-association", type=int, default=250)
    parser.add_argument("--epochs-memory", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--generator-epochs", type=int, default=30)
    parser.add_argument("--generator-batch-size", type=int, default=16)
    parser.add_argument("--generator-vocab-size", type=int, default=8192)
    parser.add_argument("--generator-dim", type=int, default=256)
    parser.add_argument("--generator-layers", type=int, default=8)
    parser.add_argument("--generator-heads", type=int, default=8)
    parser.add_argument("--generator-block-size", type=int, default=512)
    parser.add_argument("--generator-learning-rate", type=float, default=0.00035)
    parser.add_argument("--smoke", action="store_true", help="One short epoch per model for CI.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.long, device=device)


def make_optimizer(parameters, lr: float, device: torch.device):
    kwargs = {"lr": lr, "weight_decay": 1.0e-4}
    fused = False
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(parameters, fused=True, **kwargs)
            fused = True
            return optimizer, fused
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(parameters, **kwargs), fused


def autocast_context(device: torch.device, amp_enabled: bool):
    if device.type != "cuda" or not amp_enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def new_scaler(device: torch.device, amp_enabled: bool):
    enabled = device.type == "cuda" and amp_enabled
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def sync_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().item())


def save_checkpoint(path: Path, model, metadata: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "metadata": metadata}, path)


def run_perception(data, device, out_dir, args, amp_enabled):
    rows = data["examples"]
    train_x, train_s, train_r, _ = encode_examples(rows["train"], data["vocab"])
    val_x, val_s, val_r, _ = encode_examples(rows["validation"], data["vocab"])
    test_x, test_s, test_r, _ = encode_examples(rows["test"], data["vocab"])
    tx, ts, tr = (to_device(train_x, device), to_device(train_s, device), to_device(train_r, device))
    vx, vs, vr = (to_device(val_x, device), to_device(val_s, device), to_device(val_r, device))
    qx, qs, qr = (to_device(test_x, device), to_device(test_s, device), to_device(test_r, device))

    model = PerceptionNet(len(data["vocab"]), len(data["entities"]), len(data["relations"]), args.dim).to(device)
    optimizer, fused = make_optimizer(model.parameters(), args.learning_rate, device)
    scaler = new_scaler(device, amp_enabled)
    best_score, best_state, best_epoch = -1.0, None, 0
    epochs = 1 if args.smoke else args.epochs_perception
    sync_cuda(device)
    sync_cuda(device)
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(tx.shape[0], device=device)
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                subject_logits, relation_logits = model(tx.index_select(0, ids))
                loss = F.cross_entropy(subject_logits, ts.index_select(0, ids))
                loss = loss + F.cross_entropy(relation_logits, tr.index_select(0, ids))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epochs_ran = epoch + 1
        model.eval()
        with torch.no_grad(), autocast_context(device, amp_enabled):
            sv, rv = model(vx)
            score = (accuracy(sv, vs) + accuracy(rv, vr)) / 2.0
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not args.smoke and epoch - best_epoch >= 9:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast_context(device, amp_enabled):
        st, rt = model(qx)
        test_metrics = {"subject_accuracy": accuracy(st, qs), "relation_accuracy": accuracy(rt, qr)}
    sync_cuda(device)
    sync_cuda(device)
    elapsed = time.perf_counter() - start
    save_checkpoint(out_dir / "perception.pt", model, {
        "name": "PerceptionNet", "random_initialisation": True,
        "validation_score": best_score, "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    })
    return {
        "validation_score": best_score, "best_epoch": best_epoch,
        "test": test_metrics, "seconds": elapsed, "fused_adamw": fused,
        "train_examples": int(tx.shape[0]),
        "epochs_run": epochs_ran,
        "training_examples_per_second": round(int(tx.shape[0]) * epochs_ran / max(elapsed, 1e-9), 2),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_association(data, device, out_dir, args, amp_enabled):
    rows = data["association_splits"]["train"]
    num_entities = len(data["entities"])

    subjects = torch.tensor([row["subject_id"] for row in rows], dtype=torch.long, device=device)
    relations = torch.tensor([row["relation_id"] for row in rows], dtype=torch.long, device=device)
    targets = torch.zeros((len(rows), num_entities), dtype=torch.float32, device=device)
    for row_index, row in enumerate(rows):
        targets[row_index, row["object_ids"]] = 1.0

    model = AssociationNet(num_entities, len(data["relations"]), args.dim).to(device)
    optimizer, fused = make_optimizer(model.parameters(), args.learning_rate * 4.0, device)
    scaler = new_scaler(device, amp_enabled)
    epochs = 1 if args.smoke else args.epochs_association
    best_score, best_state, best_epoch = -1.0, None, 0
    start = time.perf_counter()

    def group_metrics(logits, target_matrix):
        top1 = logits.argmax(dim=-1, keepdim=True)
        hit1 = target_matrix.gather(1, top1).squeeze(1).gt(0).float().mean().item()
        k = min(5, logits.shape[-1])
        topk = logits.topk(k, dim=-1).indices
        hitk = target_matrix.gather(1, topk).gt(0).any(dim=1).float().mean().item()
        return {"top1_hit_rate": float(hit1), "top5_recall": float(hitk)}

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(subjects.shape[0], device=device)
        for batch_ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = model(subjects[batch_ids], relations[batch_ids])
                batch_targets = targets[batch_ids]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                loss = -(
                    (log_probs * batch_targets).sum(dim=-1)
                    / batch_targets.sum(dim=-1).clamp_min(1.0)
                ).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epochs_ran = epoch + 1
        model.eval()
        with torch.no_grad(), autocast_context(device, amp_enabled):
            logits = model(subjects, relations)
            metrics = group_metrics(logits, targets)
            score = metrics["top1_hit_rate"]
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not args.smoke and score >= 1.0 and epoch - best_epoch >= 20:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast_context(device, amp_enabled):
        known_metrics = group_metrics(model(subjects, relations), targets)
    sync_cuda(device)
    elapsed = time.perf_counter() - start
    save_checkpoint(out_dir / "association.pt", model, {
        "name": "AssociationNet",
        "random_initialisation": True,
        "known_graph_reconstruction_top1": best_score,
        "best_epoch": best_epoch,
        "known_graph_metrics": known_metrics,
        "evaluation_scope": "known graph reconstruction only; not held-out link prediction",
        "multi_positive_objective": "mean negative log probability across all valid objects per subject/relation",
        "asymmetric_embeddings": True,
    })
    return {
        "known_graph_reconstruction_top1": best_score,
        "best_epoch": best_epoch,
        "known_graph_metrics": known_metrics,
        "seconds": elapsed,
        "fused_adamw": fused,
        "train_subject_relation_groups": int(subjects.shape[0]),
        "epochs_run": epochs_ran,
        "training_groups_per_second": round(int(subjects.shape[0]) * epochs_ran / max(elapsed, 1e-9), 2),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "evaluation_scope": "known graph reconstruction only; not held-out link prediction",
    }


def run_memory(data, device, out_dir, args, amp_enabled):
    facts_np = encode_facts(data["facts"], data["vocab"])
    facts_x = to_device(facts_np, device)
    num_facts = len(data["facts"])
    encoded = {}
    for split in ("train", "validation", "test"):
        rows = data["examples"][split]
        x = np.stack([encode_text(row["text"], data["vocab"]) for row in rows])
        targets = torch.zeros((len(rows), num_facts), dtype=torch.float32, device=device)
        for row_index, row in enumerate(rows):
            targets[row_index, row["fact_ids"]] = 1.0
        encoded[split] = (to_device(x, device), targets)

    model = MemoryNet(len(data["vocab"]), args.dim).to(device)
    optimizer, fused = make_optimizer(model.parameters(), args.learning_rate, device)
    scaler = new_scaler(device, amp_enabled)
    best_score, best_state, best_epoch = -1.0, None, 0
    epochs = 1 if args.smoke else args.epochs_memory
    start = time.perf_counter()

    def retrieval_metrics(logits, target_matrix):
        top1 = logits.argmax(dim=-1, keepdim=True)
        recall1 = target_matrix.gather(1, top1).squeeze(1).gt(0).float().mean().item()
        k = min(5, logits.shape[-1])
        topk = logits.topk(k, dim=-1).indices
        recallk = target_matrix.gather(1, topk).gt(0).any(dim=1).float().mean().item()
        return {"recall_at_1": float(recall1), "recall_at_5": float(recallk)}

    for epoch in range(epochs):
        model.train()
        train_x, train_targets = encoded["train"]
        order = torch.randperm(train_x.shape[0], device=device)
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = model(train_x.index_select(0, ids), facts_x)
                batch_targets = train_targets[ids]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                loss = -(
                    (log_probs * batch_targets).sum(dim=-1)
                    / batch_targets.sum(dim=-1).clamp_min(1.0)
                ).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epochs_ran = epoch + 1
        model.eval()
        val_x, val_targets = encoded["validation"]
        with torch.no_grad(), autocast_context(device, amp_enabled):
            val_logits = model(val_x, facts_x)
            metrics = retrieval_metrics(val_logits, val_targets)
            score = metrics["recall_at_1"]
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not args.smoke and epoch - best_epoch >= 10:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_x, test_targets = encoded["test"]
    with torch.no_grad(), autocast_context(device, amp_enabled):
        test_logits = model(test_x, facts_x)
        test_metrics = retrieval_metrics(test_logits, test_targets)
    sync_cuda(device)
    elapsed = time.perf_counter() - start
    save_checkpoint(out_dir / "memory.pt", model, {
        "name": "MemoryNet",
        "random_initialisation": True,
        "validation_recall_at_1": best_score,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "multi_positive_fact_objective": "mean negative log probability across all valid fact records for each question",
    })
    return {
        "validation_retrieval": {"recall_at_1": best_score},
        "best_epoch": best_epoch,
        "test": test_metrics,
        "seconds": elapsed,
        "fused_adamw": fused,
        "train_queries": int(train_x.shape[0]),
        "memory_records": int(facts_x.shape[0]),
        "epochs_run": epochs_ran,
        "training_queries_per_second": round(int(train_x.shape[0]) * epochs_ran / max(elapsed, 1e-9), 2),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


LANGUAGE_SYSTEM_PROMPT = (
    "You are ORBIT, a helpful conversational assistant. Speak naturally and clearly. "
    "Answer the latest message directly, use earlier conversation turns to maintain context, "
    "and avoid repeating yourself. Keep a friendly, concise tone. "
    "Use retrieved memory facts as evidence, but do not claim memory is complete. "
    "When you are unsure, say so instead of inventing details."
)


def load_base_language_corpus() -> str:
    """Load the locally authored multi-turn dialogue corpus."""
    bundled = globals().get("LANGUAGE_BASE_CORPUS")
    if bundled is not None:
        return str(bundled)
    from runpy import run_path
    corpus_script = Path(__file__).resolve().parents[1] / "seed_model" / "make_corpus.py"
    if not corpus_script.is_file():
        raise FileNotFoundError(f"Missing dialogue corpus builder: {corpus_script}")
    run_path(str(corpus_script))
    corpus_path = corpus_script.with_name("corpus.txt")
    if not corpus_path.is_file():
        raise RuntimeError("The dialogue corpus builder did not create corpus.txt.")
    return corpus_path.read_text(encoding="utf-8")



PUBLIC_DATASET_SOURCES = [
    {
        "name": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "description": "English-only, review-approved human-written assistant conversation trees.",
    },
    {
        "name": "3nesdeniz/english-daily-dialogues-10k",
        "revision": "97e96d244d20c8970ac5be707cb89e5f94253357",
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/3nesdeniz/english-daily-dialogues-10k",
        "description": "Synthetic multi-turn English daily-life dialogues, split by scenario.",
    },
]


def _download_parquet_rows(url: str, label: str) -> tuple[list[dict[str, Any]], str, int]:
    """Download a pinned public Parquet split and return its rows and checksum."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Public dialogue training requires pandas and pyarrow in the Kaggle image."
        ) from exc

    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "ORBIT-Cognitive/2.2 (training data loader)"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            frame = pd.read_parquet(io.BytesIO(payload))
            return frame.to_dict(orient="records"), hashlib.sha256(payload).hexdigest(), len(payload)
        except Exception as exc:
            last_error = exc
            print(
                f"Dataset download/read attempt {attempt + 1}/3 failed for {label}: {exc}",
                flush=True,
            )
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Could not load public dialogue split {label}: {last_error}") from last_error


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except Exception:
        try:
            return bool(value != value)
        except Exception:
            return False


def _clean_external_text(value: Any) -> str:
    text = str(value or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<\|[^>]*\|>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\b([A-Za-z]+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _format_chat_messages(messages: list[dict[str, str]], include_end: bool = True) -> str:
    parts = [f"<|system|>{LANGUAGE_SYSTEM_PROMPT}"]
    for message in messages:
        role = message["role"]
        text = _clean_external_text(message["text"])
        if not text:
            continue
        if role in {"user", "prompter"}:
            parts.append(
                f"<|user|> {text}\n<|context|> No verified memory facts were retrieved."
            )
        elif role == "assistant":
            parts.append(f"<|assistant|> {text}")
    if include_end:
        parts.append("<|end|>")
    return "\n".join(parts)


def _format_daily_dialogue(row: dict[str, Any]) -> tuple[str, list[dict[str, str]]] | None:
    turns = row.get("turns")
    if not isinstance(turns, (list, tuple)) or len(turns) < 4:
        return None
    first_speaker = str(turns[0].get("speaker", "A"))
    messages: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        text = _clean_external_text(turn.get("text", ""))
        if not text:
            continue
        role = "user" if str(turn.get("speaker", first_speaker)) == first_speaker else "assistant"
        messages.append({"role": role, "text": text})
    if len(messages) < 4 or messages[0]["role"] != "user":
        return None
    if any(messages[i]["role"] == messages[i - 1]["role"] for i in range(1, len(messages))):
        return None
    return _format_chat_messages(messages), messages


def _load_daily_dialogues(seed: int) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    base = (
        "https://huggingface.co/datasets/3nesdeniz/english-daily-dialogues-10k/"
        "resolve/97e96d244d20c8970ac5be707cb89e5f94253357/data/"
    )
    loaded: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    byte_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows, digest, byte_count = _download_parquet_rows(
            base + f"{split}.parquet", f"English Daily Dialogues 10K/{split}"
        )
        loaded[split], hashes[split], byte_counts[split] = rows, digest, byte_count

    excluded_domains = {"health_and_wellness", "shopping_and_errands"}
    split_docs: dict[str, list[str]] = {name: [] for name in loaded}
    eval_examples: list[dict[str, Any]] = []
    for split, rows in loaded.items():
        for row in rows:
            if str(row.get("domain", "")) in excluded_domains:
                continue
            formatted = _format_daily_dialogue(row)
            if formatted is None:
                continue
            document, messages = formatted
            split_docs[split].append(document)
            if split == "test":
                assistant_indices = [
                    i for i, message in enumerate(messages) if message["role"] == "assistant"
                ]
                if not assistant_indices:
                    continue
                target_index = assistant_indices[-1]
                prefix = messages[:target_index]
                if not prefix or prefix[-1]["role"] != "user":
                    continue
                eval_examples.append({
                    "dataset": "English Daily Dialogues 10K test",
                    "prompt": prefix[-1]["text"],
                    "reference": messages[target_index]["text"],
                    "history": prefix[:-1],
                    "formatted_prompt": _format_chat_messages(prefix, include_end=False)
                    + "\n<|assistant|>",
                })

    rng = random.Random(seed)
    rng.shuffle(split_docs["train"])
    rng.shuffle(split_docs["validation"])
    rng.shuffle(eval_examples)
    metadata = {
        "name": "English Daily Dialogues 10K",
        "url": "https://huggingface.co/datasets/3nesdeniz/english-daily-dialogues-10k",
        "revision": "97e96d244d20c8970ac5be707cb89e5f94253357",
        "license": "CC-BY-4.0",
        "raw_rows": {key: len(value) for key, value in loaded.items()},
        "usable_documents": {key: len(value) for key, value in split_docs.items()},
        "sha256": hashes,
        "download_bytes": byte_counts,
        "excluded_domains": sorted(excluded_domains),
    }
    return split_docs["train"], split_docs["validation"], eval_examples, metadata


def _load_oasst_dialogues(seed: int) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    base = (
        "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
        "fdf72ae0827c1cda404aff25b6603abec9e3399b/data/"
    )
    train_rows, train_hash, train_bytes = _download_parquet_rows(
        base + "train-00000-of-00001-b42a775f407cee45.parquet", "OpenAssistant OASST1/train"
    )
    validation_rows, validation_hash, validation_bytes = _download_parquet_rows(
        base + "validation-00000-of-00001-134b8fd0c89408b6.parquet",
        "OpenAssistant OASST1/validation",
    )
    rows = train_rows + validation_rows
    by_id: dict[str, dict[str, Any]] = {}
    accepted = 0
    for raw in rows:
        if str(raw.get("lang", "")).lower() != "en":
            continue
        if (not _missing(raw.get("deleted")) and bool(raw.get("deleted"))) or (
            not _missing(raw.get("synthetic")) and bool(raw.get("synthetic"))
        ):
            continue
        if raw.get("review_result") is not None and not _missing(raw.get("review_result")):
            if not bool(raw.get("review_result")):
                continue
        state = raw.get("tree_state")
        if not _missing(state) and str(state) != "ready_for_export":
            continue
        role = str(raw.get("role", ""))
        if role not in {"prompter", "assistant"}:
            continue
        text = _clean_external_text(raw.get("text", ""))
        if len(text) < 2 or len(text) > 4000:
            continue
        tree_id = raw.get("message_tree_id")
        message_id = raw.get("message_id")
        if _missing(tree_id) or _missing(message_id):
            continue
        item = dict(raw)
        item["text"] = text
        item["message_id"] = str(message_id)
        item["parent_id"] = None if _missing(raw.get("parent_id")) else str(raw.get("parent_id"))
        item["message_tree_id"] = str(tree_id)
        by_id[item["message_id"]] = item
        accepted += 1

    # Reconstruct one supervised example for every accepted assistant message.
    # Each example contains its real ancestor turns, not an isolated prompt/answer pair.
    tree_examples: dict[str, list[dict[str, Any]]] = {}
    for row in by_id.values():
        if row["role"] != "assistant":
            continue
        chain: list[dict[str, Any]] = []
        current: dict[str, Any] | None = row
        visited: set[str] = set()
        while current is not None and current["message_id"] not in visited and len(chain) < 12:
            visited.add(current["message_id"])
            chain.append(current)
            parent_id = current.get("parent_id")
            current = by_id.get(parent_id) if parent_id else None
        chain.reverse()
        while chain and chain[0]["role"] != "prompter":
            chain.pop(0)
        if len(chain) < 2 or chain[-1]["role"] != "assistant":
            continue
        chain = chain[-12:]
        if chain[0]["role"] != "prompter":
            continue
        if any(chain[i]["role"] == chain[i - 1]["role"] for i in range(1, len(chain))):
            continue
        messages = [
            {"role": "user" if item["role"] == "prompter" else "assistant", "text": item["text"]}
            for item in chain
        ]
        document = _format_chat_messages(messages)
        prefix = messages[:-1]
        if not prefix or prefix[-1]["role"] != "user":
            continue
        tree_id = row["message_tree_id"]
        tree_examples.setdefault(tree_id, []).append({
            "document": document,
            "dataset": "OpenAssistant OASST1",
            "prompt": prefix[-1]["text"],
            "reference": messages[-1]["text"],
            "history": prefix[:-1],
            "formatted_prompt": _format_chat_messages(prefix, include_end=False) + "\n<|assistant|>",
        })

    tree_ids = list(tree_examples)
    random.Random(seed).shuffle(tree_ids)
    validation_tree_count = max(1, int(len(tree_ids) * 0.05))
    validation_tree_ids = set(tree_ids[:validation_tree_count])
    train_documents: list[str] = []
    validation_documents: list[str] = []
    validation_examples: list[dict[str, Any]] = []
    for tree_id, examples in tree_examples.items():
        if tree_id in validation_tree_ids:
            validation_documents.extend(example["document"] for example in examples)
            validation_examples.extend(examples)
        else:
            train_documents.extend(example["document"] for example in examples)
    random.Random(seed + 31).shuffle(train_documents)
    random.Random(seed + 37).shuffle(validation_documents)
    random.Random(seed + 41).shuffle(validation_examples)
    metadata = {
        "name": "OpenAssistant OASST1",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "Apache-2.0",
        "raw_rows": {"train": len(train_rows), "validation": len(validation_rows)},
        "accepted_english_messages": accepted,
        "conversation_trees": len(tree_ids),
        "assistant_training_documents": {
            "train": len(train_documents), "validation": len(validation_documents)
        },
        "sha256": {"train": train_hash, "validation": validation_hash},
        "download_bytes": {"train": train_bytes, "validation": validation_bytes},
        "split_method": "conversation-tree-held-out; all assistant turns use reconstructed ancestor history",
    }
    return train_documents, validation_documents, validation_examples, metadata


def _build_language_quality_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep real held-out reference replies beside generated replies for manual review."""
    reports = []
    for sample in samples:
        answer = sample["answer"]
        words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", answer.lower())
        ngrams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
        row = {
            "dataset": sample.get("dataset", "authored prompt"),
            "prompt": sample["prompt"],
            "answer": answer,
            "word_count": len(words),
            "unique_word_ratio": round(len(set(words)) / max(1, len(words)), 3),
            "repeated_fourgram_count": len(ngrams) - len(set(ngrams)),
            "replacement_character_count": answer.count("�"),
        }
        for key in ("reference", "history"):
            if sample.get(key):
                row[key] = sample[key]
        reports.append(row)
    return {
        "samples": reports,
        "mean_unique_word_ratio": round(
            sum(row["unique_word_ratio"] for row in reports) / max(1, len(reports)), 3
        ),
        "total_repeated_fourgram_count": sum(row["repeated_fourgram_count"] for row in reports),
        "total_replacement_character_count": sum(row["replacement_character_count"] for row in reports),
        "note": "Includes held-out multi-turn conversations with reference replies; diagnostics are not a substitute for human review.",
    }


def _natural_fact_answer(group: list[dict[str, Any]]) -> str:
    subject = group[0]["subject"].replace("_", " ")
    relation = group[0]["relation"]
    objects = [fact["object"].replace("_", " ") for fact in group]
    obj = objects[0] if len(objects) == 1 else ", ".join(objects[:-1]) + " and " + objects[-1]
    if relation == "is_a":
        if obj in {"precipitation", "furniture", "water", "oxygen", "computation", "transportation"}:
            return f"{subject} is {obj}."
        article = "an" if obj[:1].lower() in "aeiou" else "a"
        return f"{subject} is {article} {obj}."
    forms = {
        "made_of": f"{subject} is made of {obj}.",
        "falls_from": f"{subject} falls from {obj}.",
        "located_in": f"{subject} is found in {obj}.",
        "produces": f"{subject} produces {obj}.",
        "used_for": f"{subject} is used for {obj}.",
        "powered_by": f"{subject} is powered by {obj}.",
        "contains": f"{subject} contains {obj}.",
        "performs": f"{subject} performs {obj}.",
        "part_of": f"{subject} is part of {obj}.",
        "stores": f"{subject} stores {obj}.",
        "needs": f"{subject} needs {obj}.",
        "absorbs": f"{subject} absorbs {obj}.",
        "orbits": f"{subject} orbits {obj}.",
        "causes": f"{subject} can cause {obj}.",
        "flows_into": f"{subject} flows into {obj}.",
        "has_property": f"One property of {subject} is {obj}.",
        "connects": f"{subject} connects {obj}.",
    }
    return forms.get(relation, f"{subject} is related to {obj}.")


def build_language_documents(
    data: dict[str, Any], seed: int, include_public: bool = True
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build assistant-supervised chat documents with held-out conversation trees."""
    authored: list[str] = []
    for chunk in load_base_language_corpus().split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            chunk = chunk.replace(
                "\n<|assistant|>",
                "\n<|context|>No verified memory facts were retrieved.\n<|assistant|>",
            )
            authored.append(f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n{chunk}\n<|end|>")

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for fact in data["facts"]:
        grouped.setdefault((fact["subject_id"], fact["relation_id"]), []).append(fact)
    for group in grouped.values():
        subject = group[0]["subject"].replace("_", " ")
        relation = group[0]["relation"]
        context = "; ".join(
            f"{fact['subject'].replace('_', ' ')} {fact['relation'].replace('_', ' ')} "
            f"{fact['object'].replace('_', ' ')}" for fact in group
        )
        answer = _natural_fact_answer(group)
        for template in QUESTION_TEMPLATES[relation]:
            authored.append(
                f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n"
                f"<|user|> {template.format(s=subject)}\n"
                f"<|context|> {context}\n"
                f"<|assistant|> {answer}\n<|end|>"
            )
    authored.extend([
        f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n"
        "<|user|> What is a flibbertigibbet?\n"
        "<|context|> No verified memory facts were retrieved.\n"
        "<|assistant|> I do not have enough information about that term to give a reliable definition.\n<|end|>",
        f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n"
        "<|user|> Can you answer every possible question?\n"
        "<|context|> No verified memory facts were retrieved.\n"
        "<|assistant|> No. I have limited knowledge and can make mistakes. I should say when I am unsure instead of presenting a guess as a fact.\n<|end|>",
    ])

    rng = random.Random(seed)
    rng.shuffle(authored)
    internal_split = max(1, int(len(authored) * 0.95))
    train_documents = authored[:internal_split]
    validation_documents = authored[internal_split:]
    eval_examples: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []

    if include_public:
        oasst_train, oasst_validation, oasst_eval, oasst_info = _load_oasst_dialogues(seed)
        daily_train, daily_validation, daily_eval, daily_info = _load_daily_dialogues(seed)
        train_documents.extend(oasst_train)
        validation_documents.extend(oasst_validation)
        train_documents.extend(daily_train)
        validation_documents.extend(daily_validation)
        eval_examples.extend(oasst_eval)
        eval_examples.extend(daily_eval)
        provenance.extend([oasst_info, daily_info])

    rng.shuffle(train_documents)
    rng.shuffle(validation_documents)
    rng.shuffle(eval_examples)
    return train_documents, validation_documents, eval_examples, provenance


def assistant_target_mask(
    token_ids: list[int] | torch.Tensor, tokenizer: VocabularyTokenizer
) -> torch.Tensor:
    """Mark assistant reply text and response-ending control tokens as targets."""
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
    assistant_id = tokenizer.token_id("<|assistant|>")
    user_id = tokenizer.token_id("<|user|>")
    context_id = tokenizer.token_id("<|context|>")
    system_id = tokenizer.token_id("<|system|>")
    end_id = tokenizer.token_id("<|end|>")
    controls = {value for value in (user_id, context_id, system_id) if value is not None}
    mask = [False] * len(ids)
    active = False
    for index, token_id in enumerate(ids):
        if token_id == assistant_id:
            active = True
            continue
        if token_id in controls or token_id in {end_id, tokenizer.EOS_ID}:
            if active and token_id in {user_id, end_id}:
                mask[index] = True
            active = False
            continue
        if active:
            mask[index] = True
    return torch.tensor(mask, dtype=torch.bool)


def _language_batch(
    sequences: list[torch.Tensor],
    target_masks: list[torch.Tensor],
    block_size: int,
    batch_size: int,
    pad_id: int,
    device: torch.device,
):
    """Sample windows containing at least one assistant target."""
    inputs = torch.full((batch_size, block_size), pad_id, dtype=torch.long, device=device)
    targets = torch.full_like(inputs, pad_id)
    loss_masks = torch.zeros((batch_size, block_size), dtype=torch.bool, device=device)
    eligible = []
    for sequence, target_mask in zip(sequences, target_masks):
        max_start = max(0, int(sequence.numel()) - block_size - 1)
        target_positions = torch.nonzero(target_mask, as_tuple=False).flatten()
        valid_positions = target_positions[
            (target_positions >= 1) & (target_positions <= max_start + block_size)
        ]
        if valid_positions.numel():
            eligible.append((sequence, target_mask, max_start, valid_positions))
    if not eligible:
        raise RuntimeError("No sequence contains an assistant reply target within the context window.")

    for row in range(batch_size):
        sequence, target_mask, max_start, positions = eligible[random.randrange(len(eligible))]
        target_position = int(positions[random.randrange(int(positions.numel()))].item())
        low = max(0, target_position - block_size)
        high = min(max_start, target_position - 1)
        start = random.randint(low, high)
        sample = sequence[start:start + block_size + 1]
        usable = max(0, min(block_size, int(sample.numel()) - 1))
        if usable:
            inputs[row, :usable] = sample[:usable]
            targets[row, :usable] = sample[1:usable + 1]
            loss_masks[row, :usable] = target_mask[start + 1:start + 1 + usable].to(device)
    return inputs, targets, loss_masks


def _masked_language_loss(
    logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor, pad_id: int
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).view_as(targets)
    active = loss_mask & targets.ne(pad_id)
    if not bool(active.any()):
        raise RuntimeError("Sampled batch contains no assistant reply targets.")
    return per_token[active].mean()


def run_language_generator(data, device, out_dir, args, amp_enabled):
    train_documents, validation_documents, dialogue_eval, dataset_sources = build_language_documents(
        data, args.seed, include_public=not args.smoke
    )
    if len(train_documents) < 20 or not validation_documents:
        raise RuntimeError("Language train/validation documents are insufficient.")

    tokenizer = VocabularyTokenizer.build(
        train_documents, max_vocab=args.generator_vocab_size, min_frequency=2
    )

    def encode_documents(documents):
        sequences = []
        target_masks = []
        for document in documents:
            ids = tokenizer.encode(document) + [tokenizer.EOS_ID]
            sequences.append(torch.tensor(ids, dtype=torch.long, device=device))
            target_masks.append(assistant_target_mask(ids, tokenizer))
        return sequences, target_masks

    train_sequences, train_masks = encode_documents(train_documents)
    validation_sequences, validation_masks = encode_documents(validation_documents)
    config = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.PAD_ID,
        "eos_id": tokenizer.EOS_ID,
        "block_size": args.generator_block_size,
        "dim": args.generator_dim,
        "layers": args.generator_layers,
        "heads": args.generator_heads,
        "dropout": 0.10,
        "tokenizer": tokenizer.FORMAT,
        "from_scratch": True,
        "objective": "assistant-only causal next-token prediction over multi-turn context",
    }
    model = NeuralLanguageGenerator(config).to(device)
    optimizer, fused = make_optimizer(model.parameters(), args.generator_learning_rate, device)
    scaler = new_scaler(device, amp_enabled)
    epochs = 1 if args.smoke else args.generator_epochs
    batch_size = min(args.generator_batch_size, 4) if args.smoke else args.generator_batch_size
    train_token_count = sum(int(sequence.numel()) for sequence in train_sequences)
    assistant_target_count = sum(int(mask.sum().item()) for mask in train_masks)
    steps_per_epoch = max(
        120, min(320, assistant_target_count // max(1, config["block_size"] * batch_size))
    )
    if args.smoke:
        steps_per_epoch = 3
    validation_batches = 2 if args.smoke else 30
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs),
        eta_min=args.generator_learning_rate * 0.10,
    )
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            inputs, targets, loss_mask = _language_batch(
                train_sequences, train_masks, config["block_size"], batch_size,
                tokenizer.PAD_ID, device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = _masked_language_loss(logits, targets, loss_mask, tokenizer.PAD_ID)
            if not torch.isfinite(loss):
                raise RuntimeError("Language-generator training produced a non-finite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item())

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for _ in range(validation_batches):
                inputs, targets, loss_mask = _language_batch(
                    validation_sequences, validation_masks, config["block_size"], batch_size,
                    tokenizer.PAD_ID, device,
                )
                with autocast_context(device, amp_enabled):
                    logits = model(inputs)
                    loss = _masked_language_loss(logits, targets, loss_mask, tokenizer.PAD_ID)
                validation_loss += float(loss.item())
        validation_loss /= validation_batches
        train_loss = running_loss / steps_per_epoch
        scheduler.step()
        print(
            f"ConversationalGenerator epoch {epoch + 1}/{epochs}: "
            f"train_assistant_loss={train_loss:.4f} validation_assistant_loss={validation_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if not args.smoke and stale_epochs >= 7:
                break

    if best_state is None:
        raise RuntimeError("Language-generator training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    tokenizer.save(out_dir / "tokenizer.json")

    evaluation_samples = []
    torch.manual_seed(args.seed + 991)
    random.Random(args.seed + 991).shuffle(dialogue_eval)
    for sample in dialogue_eval[:24]:
        answer = model.generate(
            sample["formatted_prompt"], tokenizer, max_new_tokens=96,
            temperature=0.42, top_k=32, top_p=0.90,
        )
        evaluation_samples.append({
            "dataset": sample["dataset"],
            "prompt": sample["prompt"],
            "history": sample.get("history", []),
            "reference": sample["reference"],
            "answer": answer,
        })

    basic_prompts = [
        "Hi! How are you doing today?",
        "I've been learning programming, but loops are confusing. Could you explain them simply?",
        "I want to make a small game. What is a sensible first step?",
        "I do not know the answer to something. How should I handle that?",
    ]
    for prompt in basic_prompts:
        formatted = (
            f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n"
            f"<|user|> {prompt}\n"
            "<|context|> No verified memory facts were retrieved.\n"
            "<|assistant|>"
        )
        answer = model.generate(
            formatted, tokenizer, max_new_tokens=80,
            temperature=0.42, top_k=32, top_p=0.90,
        )
        evaluation_samples.append({
            "dataset": "authored basic conversation prompt",
            "prompt": prompt,
            "answer": answer,
        })
    generation_evaluation = _build_language_quality_stats(evaluation_samples)

    sync_cuda(device)
    elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    config.update({
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_validation_assistant_loss": best_validation_loss,
        "training_documents": len(train_sequences),
        "validation_documents": len(validation_sequences),
        "training_tokens": train_token_count,
        "assistant_target_tokens": assistant_target_count,
        "validation_tokens": sum(int(x.numel()) for x in validation_sequences),
        "validation_assistant_target_tokens": sum(int(mask.sum().item()) for mask in validation_masks),
        "vocab_size": tokenizer.vocab_size,
        "random_initialisation": True,
        "dataset_sources": dataset_sources,
    })
    save_checkpoint(out_dir / "generator.pt", model, {
        "name": "NeuralLanguageGenerator",
        "random_initialisation": True,
        "objective": config["objective"],
        "best_epoch": best_epoch,
        "best_validation_assistant_loss": best_validation_loss,
        "training_documents": len(train_sequences),
        "validation_documents": len(validation_sequences),
        "training_tokens": train_token_count,
        "assistant_target_tokens": assistant_target_count,
        "optimizer_fused": fused,
        "context_length_tokens": config["block_size"],
        "tokenizer_format": tokenizer.FORMAT,
        "dataset_sources": dataset_sources,
    })
    return {
        "parameters": parameters,
        "epochs_run": epoch + 1,
        "best_epoch": best_epoch,
        "best_validation_assistant_loss": best_validation_loss,
        "training_assistant_loss_last_epoch": train_loss,
        "training_documents": len(train_sequences),
        "validation_documents": len(validation_sequences),
        "training_tokens": train_token_count,
        "assistant_target_tokens": assistant_target_count,
        "vocab_size": tokenizer.vocab_size,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "seconds": elapsed,
        "fused_adamw": fused,
        "dataset_sources": dataset_sources,
        "generation_evaluation": generation_evaluation,
    }, config


def write_runtime_package(output: Path) -> list[str]:
    """Add the complete inference runner to the model archive."""
    names = ["chat.py", "core.py", "models.py", "language.py", "requirements.txt"]
    embedded = globals().get("RUNTIME_SOURCE_FILES")
    source_root = Path(__file__).resolve().parent
    written = []
    for name in names:
        if isinstance(embedded, dict) and name in embedded:
            content = str(embedded[name])
        else:
            source_path = source_root / name
            if not source_path.is_file():
                raise FileNotFoundError(f"Cannot package inference source file: {source_path}")
            content = source_path.read_text(encoding="utf-8")
        (output / name).write_text(content, encoding="utf-8")
        written.append(name)

    (output / "README.txt").write_text(
        "ORBIT Cognitive v2.2 Conversational - complete inference package\n"
        "=================================================\n\n"
        "This archive contains trained checkpoints, the neural language generator, "
        "configuration, fact memory, and the Python files needed to run the model.\n\n"
        "Training data attribution and licences are listed in DATA_SOURCES.txt.\n\n"
        "Requirements: Python 3.10 or newer, PyTorch, and NumPy. Dependencies are "
        "not installed automatically. Use the Python environment where you already "
        "installed PyTorch and NumPy.\n\n"
        "Windows: double-click run_orbit.bat.\n"
        "PowerShell: python chat.py --model . --device cpu\n"
        "NVIDIA CUDA: python chat.py --model . --device cuda\n\n"
        "Type /reset to clear conversation history and /exit to quit. The model is "
        "experimental and may produce incoherent or incorrect answers.\n",
        encoding="utf-8",
    )
    (output / "DATA_SOURCES.txt").write_text(
        "ORBIT Cognitive v2.2 training data sources\n"
        "==========================================\n\n"
        "OpenAssistant OASST1\n"
        "Licence: Apache-2.0\n"
        "Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1\n"
        "Pinned revision: fdf72ae0827c1cda404aff25b6603abec9e3399b\n"
        "Only English, non-deleted, non-synthetic, review-approved messages from ready-for-export trees are used.\n\n"
        "English Daily Dialogues 10K\n"
        "Author: Enes Deniz\n"
        "Licence: CC-BY-4.0\n"
        "Dataset: https://huggingface.co/datasets/3nesdeniz/english-daily-dialogues-10k\n"
        "Pinned revision: 97e96d244d20c8970ac5be707cb89e5f94253357\n"
        "Synthetic multi-turn dialogues; health/wellness and shopping/errands domains are excluded.\n\n"
        "The exact downloaded split checksums and row counts are recorded in training_report.json.\n"
        "ORBIT student model weights are randomly initialised; no pretrained model weights are loaded.\n",
        encoding="utf-8",
    )
    (output / "run_orbit.bat").write_text(
        "@echo off\n"
        "setlocal\n"
        "cd /d \"%~dp0\"\n"
        "if exist \"%~dp0.venv\\Scripts\\python.exe\" (\n"
        "  \"%~dp0.venv\\Scripts\\python.exe\" chat.py --model \"%~dp0\" %*\n"
        ") else (\n"
        "  python chat.py --model \"%~dp0\" %*\n"
        ")\n"
        "echo.\n"
        "pause\n",
        encoding="utf-8",
    )
    return written


def main():
    args = parse_args()
    if args.smoke:
        args.epochs_perception = args.epochs_association = args.epochs_memory = 1
        args.generator_epochs = 1
        args.generator_dim = 32
        args.generator_vocab_size = 1024
        args.generator_layers = 1
        args.generator_heads = 2
        args.generator_block_size = 64
        args.generator_batch_size = 4
        args.batch_size = min(args.batch_size, 32)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    if device.type == "cpu" and not args.smoke:
        raise SystemExit("CPU training is disabled for normal runs. Use --smoke for CI only.")

    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    amp_enabled = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.cuda.get_device_capability(0)[0] >= 8
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(0)
        device_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    else:
        device_name, device_memory_gb = "CPU smoke test", None

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = build_seed_data()
    # Store the exact data and vocabulary used by the checkpoint.
    (output / "vocab.json").write_text(json.dumps(data["vocab"], indent=2), encoding="utf-8")
    (output / "entities.json").write_text(json.dumps(data["entities"], indent=2), encoding="utf-8")
    (output / "relations.json").write_text(json.dumps(data["relations"], indent=2), encoding="utf-8")
    serial_facts = [{k: v for k, v in fact.items()} for fact in data["facts"]]
    (output / "facts.json").write_text(json.dumps(serial_facts, indent=2), encoding="utf-8")

    data_fingerprint = hashlib.sha256(
        json.dumps(serial_facts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = {
        "project": "ORBIT Cognitive v2.2 Conversational",
        "dataset_fingerprint_sha256": data_fingerprint,
        "from_scratch": True,
        "pretrained_student_weights": False,
        "specialists": ["PerceptionNet", "AssociationNet", "MemoryNet", "NeuralLanguageGenerator"],
        "dim": args.dim,
        "max_length": data["max_len"],
        "vocab_size": len(data["vocab"]),
        "entity_count": len(data["entities"]),
        "relation_count": len(data["relations"]),
        "fact_count": len(data["facts"]),
        "language_generator": {"block_size": args.generator_block_size, "dim": args.generator_dim, "layers": args.generator_layers, "heads": args.generator_heads, "dropout": 0.10, "tokenizer": VocabularyTokenizer.FORMAT, "from_scratch": True},
        "seed": args.seed,
        "device": device_name,
        "device_memory_gb": device_memory_gb,
        "device_capability": (
            list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": amp_enabled,
        "optimizer": "AdamW (fused when supported)",
        "training_examples": {k: len(v) for k, v in data["examples"].items()},
        "dataset": "Original authored dialogue plus English-only reviewed OpenAssistant OASST1 and English Daily Dialogues 10K; conversation-level validation split",
        "dataset_scope_warning": "Student weights are randomly initialised. Public dialogue datasets improve conversational coverage but do not guarantee general reasoning or factual accuracy.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("ORBIT Cognitive v2.2 Conversational training configuration:")
    print(json.dumps(config, indent=2), flush=True)

    start = time.perf_counter()
    metrics = {
        "perception": run_perception(data, device, output, args, amp_enabled),
        "association": run_association(data, device, output, args, amp_enabled),
        "memory": run_memory(data, device, output, args, amp_enabled),
    }
    generator_metrics, generator_config = run_language_generator(data, device, output, args, amp_enabled)
    metrics["language_generator"] = generator_metrics
    config["language_generator"] = generator_config
    config["language_dataset_sources"] = generator_config.get("dataset_sources", [])
    total_seconds = time.perf_counter() - start
    report = {
        "project": "ORBIT Cognitive v2.2 Conversational",
        "status": "smoke_test_only" if args.smoke else "trained",
        "all_students_randomly_initialised": True,
        "teacher_weights_loaded": False,
        "config": config,
        "metrics": metrics,
        "total_training_seconds": round(total_seconds, 3),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if device.type == "cuda" else None
        ),
        "speed_features": {
            "cuda_training": device.type == "cuda",
            "mixed_precision_fp16": amp_enabled,
            "fused_adamw_requested_when_supported": device.type == "cuda",
            "gpu_resident_tokenised_datasets": device.type == "cuda",
            "vectorised_all_fact_retrieval": True,
            "tf32_enabled_on_ampere_or_newer": bool(
                device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32
            ),
            "torch_compile": False,
        },
        "limitations": [
            "The seed fact graph is small and manually curated.",
            "Public conversation datasets are finite and domain-limited; generated answers still require human evaluation.",
            "High benchmark scores do not establish general reasoning or broad language understanding.",
            "The generator is trained from scratch on reviewed public dialogue datasets plus authored examples; this does not guarantee human-level fluency or factual accuracy.",
            "The 67-fact graph provides narrow factual grounding; the language model can still make unsupported statements.",
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    packaged_sources = write_runtime_package(output)
    report["runtime_files"] = packaged_sources + ["README.txt", "DATA_SOURCES.txt", "run_orbit.bat"]
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    archive = output.parent / f"{output.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(output))
    print(f"Training complete in {total_seconds:.1f}s. Archive: {archive}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
