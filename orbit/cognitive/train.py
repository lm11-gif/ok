#!/usr/bin/env python3
"""Train ORBIT Cognitive v1 specialist networks from random initialisation.

GPU is mandatory by default. Use --device cpu only for CI/smoke tests.
All generated training examples are based on the versioned seed knowledge graph.
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--epochs-association", type=int, default=70)
    parser.add_argument("--epochs-memory", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
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
    return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_enabled))


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
    }


def run_association(data, device, out_dir, args, amp_enabled):
    splits = data["association_splits"]
    def ids(rows):
        s = torch.tensor([x["subject_id"] for x in rows], dtype=torch.long, device=device)
        r = torch.tensor([x["relation_id"] for x in rows], dtype=torch.long, device=device)
        o = torch.tensor([x["object_id"] for x in rows], dtype=torch.long, device=device)
        return s, r, o
    train_s, train_r, train_o = ids(splits["train"])
    val_s, val_r, val_o = ids(splits["validation"])
    test_s, test_r, test_o = ids(splits["test"])
    model = AssociationNet(len(data["entities"]), len(data["relations"]), args.dim).to(device)
    optimizer, fused = make_optimizer(model.parameters(), args.learning_rate * 0.8, device)
    scaler = new_scaler(device, amp_enabled)
    best_score, best_state, best_epoch = -1.0, None, 0
    epochs = 1 if args.smoke else args.epochs_association
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(train_s.shape[0], device=device)
        for batch_ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = model(train_s[batch_ids], train_r[batch_ids])
                loss = F.cross_entropy(logits, train_o[batch_ids])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        with torch.no_grad(), autocast_context(device, amp_enabled):
            val_logits = model(val_s, val_r)
            score = accuracy(val_logits, val_o)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not args.smoke and epoch - best_epoch >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast_context(device, amp_enabled):
        test_logits = model(test_s, test_r)
        test_metrics = {"object_top1_accuracy": accuracy(test_logits, test_o)}
        k = min(5, test_logits.shape[-1])
        test_metrics["object_top5_recall"] = float(
            (test_logits.topk(k, dim=-1).indices == test_o.unsqueeze(1)).any(dim=1).float().mean().item()
        )
    elapsed = time.perf_counter() - start
    save_checkpoint(out_dir / "association.pt", model, {
        "name": "AssociationNet", "random_initialisation": True,
        "validation_object_accuracy": best_score, "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    })
    return {
        "validation_object_accuracy": best_score, "best_epoch": best_epoch,
        "test": test_metrics, "seconds": elapsed, "fused_adamw": fused,
        "train_edges": int(train_s.shape[0]),
    }


def run_memory(data, device, out_dir, args, amp_enabled):
    facts_np = encode_facts(data["facts"], data["vocab"])
    facts_x = to_device(facts_np, device)
    encoded = {}
    for split in ("train", "validation", "test"):
        x, _, _, fact_ids = encode_examples(data["examples"][split], data["vocab"])
        encoded[split] = (to_device(x, device), to_device(fact_ids, device))

    model = MemoryNet(len(data["vocab"]), args.dim).to(device)
    optimizer, fused = make_optimizer(model.parameters(), args.learning_rate, device)
    scaler = new_scaler(device, amp_enabled)
    best_score, best_state, best_epoch = -1.0, None, 0
    epochs = 1 if args.smoke else args.epochs_memory
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_x, train_y = encoded["train"]
        order = torch.randperm(train_x.shape[0], device=device)
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = model(train_x.index_select(0, ids), facts_x)
                loss = F.cross_entropy(logits, train_y.index_select(0, ids))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        val_x, val_y = encoded["validation"]
        with torch.no_grad(), autocast_context(device, amp_enabled):
            val_logits = model(val_x, facts_x)
            score = accuracy(val_logits, val_y)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not args.smoke and epoch - best_epoch >= 10:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_x, test_y = encoded["test"]
    with torch.no_grad(), autocast_context(device, amp_enabled):
        test_logits = model(test_x, facts_x)
        top1 = accuracy(test_logits, test_y)
        topk = min(5, test_logits.shape[-1])
        top5 = float(
            (test_logits.topk(topk, dim=-1).indices == test_y.unsqueeze(1)).any(dim=1).float().mean().item()
        )
    elapsed = time.perf_counter() - start
    save_checkpoint(out_dir / "memory.pt", model, {
        "name": "MemoryNet", "random_initialisation": True,
        "validation_recall_at_1": best_score, "best_epoch": best_epoch,
        "test_metrics": {"recall_at_1": top1, "recall_at_5": top5},
    })
    return {
        "validation_recall_at_1": best_score, "best_epoch": best_epoch,
        "test": {"recall_at_1": top1, "recall_at_5": top5},
        "seconds": elapsed, "fused_adamw": fused,
        "train_queries": int(train_x.shape[0]), "memory_records": int(facts_x.shape[0]),
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

    config = {
        "project": "ORBIT Cognitive v1",
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

    archive = output.parent / "orbit-cognitive-v1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(output))
    print(f"Training complete in {total_seconds:.1f}s. Archive: {archive}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
