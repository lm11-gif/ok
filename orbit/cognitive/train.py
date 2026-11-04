#!/usr/bin/env python3
"""Train ORBIT Cognitive v1 specialist networks from random initialisation.

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
    from .core import build_seed_data, encode_examples, encode_facts
    from .models import PerceptionNet, AssociationNet, MemoryNet
except ImportError:  # Kaggle script mode: modules are side-by-side.
    from core import build_seed_data, encode_examples, encode_facts
    from models import PerceptionNet, AssociationNet, MemoryNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output", default="artifacts/orbit-cognitive-v1")
    parser.add_argument("--epochs-perception", type=int, default=35)
    parser.add_argument("--epochs-association", type=int, default=250)
    parser.add_argument("--epochs-memory", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20261104)
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
        x = np.stack([__import__("core").encode_text(row["text"], data["vocab"]) for row in rows]) \
            if rows and "core" in __import__("sys").modules else None
        if x is None:
            from core import encode_text as _encode_text
            x = np.stack([_encode_text(row["text"], data["vocab"]) for row in rows])
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
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main():
    args = parse_args()
    if args.smoke:
        args.epochs_perception = args.epochs_association = args.epochs_memory = 1
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
        "project": "ORBIT Cognitive v1",
        "dataset_fingerprint_sha256": data_fingerprint,
        "from_scratch": True,
        "pretrained_student_weights": False,
        "specialists": ["PerceptionNet", "AssociationNet", "MemoryNet"],
        "dim": args.dim,
        "max_length": data["max_len"],
        "vocab_size": len(data["vocab"]),
        "entity_count": len(data["entities"]),
        "relation_count": len(data["relations"]),
        "fact_count": len(data["facts"]),
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
        "dataset": "Versioned built-in seed knowledge graph and deterministic template-generated questions",
        "dataset_scope_warning": "Small bootstrap benchmark only; not broad language understanding or comprehensive world knowledge.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("ORBIT Cognitive v1 training configuration:")
    print(json.dumps(config, indent=2), flush=True)

    start = time.perf_counter()
    metrics = {
        "perception": run_perception(data, device, output, args, amp_enabled),
        "association": run_association(data, device, output, args, amp_enabled),
        "memory": run_memory(data, device, output, args, amp_enabled),
    }
    total_seconds = time.perf_counter() - start
    report = {
        "project": "ORBIT Cognitive v1",
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
            "The first release does not include a free-form language generation network.",
        ],
    }
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
