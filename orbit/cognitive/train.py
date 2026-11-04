#!/usr/bin/env python3
"""Train ORBIT Cognitive v2 networks, including a scratch-initialised language generator.

GPU is mandatory by default. Use --device cpu only for CI/smoke tests.
All generated training examples are based on the versioned seed knowledge graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
import random
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
    from .models import PerceptionNet, AssociationNet, MemoryNet, NeuralLanguageGenerator
except ImportError:  # Kaggle script mode: modules are side-by-side.
    from core import build_seed_data, encode_examples, encode_facts, encode_text, QUESTION_TEMPLATES
    from models import PerceptionNet, AssociationNet, MemoryNet, NeuralLanguageGenerator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output", default="artifacts/orbit-cognitive-v2")
    parser.add_argument("--epochs-perception", type=int, default=35)
    parser.add_argument("--epochs-association", type=int, default=250)
    parser.add_argument("--epochs-memory", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--generator-epochs", type=int, default=12)
    parser.add_argument("--generator-batch-size", type=int, default=32)
    parser.add_argument("--generator-dim", type=int, default=128)
    parser.add_argument("--generator-layers", type=int, default=4)
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--generator-block-size", type=int, default=384)
    parser.add_argument("--generator-learning-rate", type=float, default=0.0006)
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


LANGUAGE_EOS_ID = 256
LANGUAGE_SYSTEM_PROMPT = (
    "You are ORBIT, an experimental AI assistant. Respond naturally and directly. "
    "Use retrieved memory facts as evidence, but do not claim that memory is complete. "
    "When evidence is missing, be honest rather than inventing a factual answer."
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


def build_language_documents(data: dict[str, Any], seed: int) -> list[str]:
    """Join original conversations with fact-grounded prompts for retrieval conditioning."""
    base_text = load_base_language_corpus()
    documents: list[str] = []
    for chunk in base_text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = chunk.replace(
            "\n<|assistant|>",
            "\n<|context|>No verified memory facts were retrieved.\n<|assistant|>",
        )
        documents.append(f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n{chunk}\n<|end|>")

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
            documents.append(
                f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n"
                f"<|user|> {template.format(s=subject)}\n"
                f"<|context|> {context}\n"
                f"<|assistant|> {answer}\n<|end|>"
            )
    documents.extend([
        f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n"
        "<|user|> What is a flibbertigibbet?\n"
        "<|context|> No verified memory facts were retrieved.\n"
        "<|assistant|> I do not have enough information about that term to give a reliable definition.\n<|end|>",
        f"<|system|>{LANGUAGE_SYSTEM_PROMPT}\n"
        "<|user|> Can you answer every possible question?\n"
        "<|context|> No verified memory facts were retrieved.\n"
        "<|assistant|> No. I have limited knowledge and can make mistakes. I should say when I am unsure instead of presenting a guess as a fact.\n<|end|>",
    ])
    random.Random(seed).shuffle(documents)
    return documents


def encode_language_documents(documents: list[str]) -> tuple[list[int], list[int]]:
    if len(documents) < 2:
        raise ValueError("The language corpus needs at least two documents.")
    split = max(1, int(len(documents) * 0.97))
    split = min(split, len(documents) - 1)

    def flatten(rows: list[str]) -> list[int]:
        values: list[int] = []
        for row in rows:
            values.extend(row.encode("utf-8"))
            values.append(LANGUAGE_EOS_ID)
        return values

    return flatten(documents[:split]), flatten(documents[split:])


def _language_batch(tokens: torch.Tensor, block_size: int, batch_size: int):
    max_start = tokens.numel() - block_size - 1
    if max_start < 1:
        raise ValueError("Language corpus is shorter than the generator context window.")
    starts = torch.randint(max_start, (batch_size,), device=tokens.device)
    offsets = torch.arange(block_size + 1, device=tokens.device)
    batch = tokens[starts[:, None] + offsets[None, :]]
    return batch[:, :-1], batch[:, 1:]


def run_language_generator(data, device, out_dir, args, amp_enabled):
    documents = build_language_documents(data, args.seed)
    train_ids, validation_ids = encode_language_documents(documents)
    train_tokens = torch.tensor(train_ids, dtype=torch.long, device=device)
    validation_tokens = torch.tensor(validation_ids, dtype=torch.long, device=device)
    config = {
        "vocab_size": NeuralLanguageGenerator.vocab_size,
        "block_size": args.generator_block_size,
        "dim": args.generator_dim,
        "layers": args.generator_layers,
        "heads": args.generator_heads,
        "dropout": 0.10,
        "tokenizer": "UTF-8 byte-level; IDs 0..255 are bytes, 256 is EOS",
        "from_scratch": True,
    }
    model = NeuralLanguageGenerator(config).to(device)
    optimizer, fused = make_optimizer(model.parameters(), args.generator_learning_rate, device)
    scaler = new_scaler(device, amp_enabled)
    epochs = 1 if args.smoke else args.generator_epochs
    batch_size = min(args.generator_batch_size, 4) if args.smoke else args.generator_batch_size
    steps_per_epoch = max(1, train_tokens.numel() // (config["block_size"] * batch_size))
    if args.smoke:
        steps_per_epoch = min(2, steps_per_epoch)
    validation_batches = 2 if args.smoke else 12
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            inputs, targets = _language_batch(train_tokens, config["block_size"], batch_size)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, NeuralLanguageGenerator.vocab_size),
                    targets.reshape(-1),
                )
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
                inputs, targets = _language_batch(
                    validation_tokens, config["block_size"], batch_size
                )
                with autocast_context(device, amp_enabled):
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, NeuralLanguageGenerator.vocab_size),
                        targets.reshape(-1),
                    )
                validation_loss += float(loss.item())
        validation_loss /= validation_batches
        train_loss = running_loss / steps_per_epoch
        print(
            f"LanguageGenerator epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}",
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
            if not args.smoke and stale_epochs >= 4:
                break

    if best_state is None:
        raise RuntimeError("Language-generator training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    sync_cuda(device)
    elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    config.update({
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_documents": len(documents),
        "training_bytes": len(train_ids),
        "validation_bytes": len(validation_ids),
        "random_initialisation": True,
    })
    save_checkpoint(out_dir / "generator.pt", model, {
        "name": "NeuralLanguageGenerator",
        "random_initialisation": True,
        "objective": "causal next-byte prediction",
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_documents": len(documents),
        "training_bytes": len(train_ids),
        "validation_bytes": len(validation_ids),
        "optimizer_fused": fused,
        "context_length_bytes": config["block_size"],
    })
    return {
        "parameters": parameters,
        "epochs_run": epoch + 1,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_loss_last_epoch": train_loss,
        "training_documents": len(documents),
        "training_bytes": len(train_ids),
        "validation_bytes": len(validation_ids),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "seconds": elapsed,
        "fused_adamw": fused,
    }, config


def write_runtime_package(output: Path) -> list[str]:
    """Add the complete inference runner to the model archive."""
    names = ["chat.py", "core.py", "models.py", "requirements.txt"]
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
        "ORBIT Cognitive v2 - complete inference package\n"
        "=================================================\n\n"
        "This archive contains trained checkpoints, the neural language generator, "
        "configuration, fact memory, and the Python files needed to run the model.\n\n"
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
        "project": "ORBIT Cognitive v2",
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
        "language_generator": {"vocab_size": 257, "block_size": args.generator_block_size, "dim": args.generator_dim, "layers": args.generator_layers, "heads": args.generator_heads, "dropout": 0.10, "tokenizer": "UTF-8 byte-level; 256 is EOS", "from_scratch": True},
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
        "dataset": "Seed fact graph plus original synthetic multi-turn dialogue corpus and fact-grounded prompts",
        "dataset_scope_warning": "Small bootstrap benchmark only; not broad language understanding or comprehensive world knowledge.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("ORBIT Cognitive v2 training configuration:")
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
    total_seconds = time.perf_counter() - start
    report = {
        "project": "ORBIT Cognitive v2",
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
            "Template-generated language coverage is narrow.",
            "High benchmark scores do not establish general reasoning or broad language understanding.",
            "The generator is a small byte-level model trained on an original synthetic dialogue corpus; fluency and factual accuracy remain limited.",
            "The 67-fact graph provides narrow factual grounding; the language model can still make unsupported statements.",
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    packaged_sources = write_runtime_package(output)
    report["runtime_files"] = packaged_sources + ["README.txt", "run_orbit.bat"]
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
