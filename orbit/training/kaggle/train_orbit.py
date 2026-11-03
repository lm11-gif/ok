#!/usr/bin/env python3
"""Kaggle entrypoint for continuing the committed ORBIT Seed checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "ok"
MODEL_DIR = REPO / "orbit" / "seed_model"
OUTPUT_DIR = WORK / "orbit-seed-trained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = int(os.environ.get("ORBIT_STEPS", "500"))
BATCH_SIZE = int(os.environ.get("ORBIT_BATCH_SIZE", "32"))
SEQ_LEN = int(os.environ.get("ORBIT_SEQ_LEN", "160"))
LEARNING_RATE = float(os.environ.get("ORBIT_LEARNING_RATE", "3e-5"))

def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if STEPS < 1 or STEPS > 50000:
    raise ValueError("ORBIT_STEPS must be between 1 and 50000.")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle did not provide CUDA. Enable GPU acceleration and retry.")

gpu_name = torch.cuda.get_device_name(0)
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": gpu_name,
    "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
    "steps": STEPS,
    "batch_size": BATCH_SIZE,
    "sequence_length": SEQ_LEN,
}, indent=2), flush=True)

if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", "main",
     "https://github.com/lm11-gif/ok.git", str(REPO)])

required = [
    MODEL_DIR / "config.json",
    MODEL_DIR / "model.safetensors",
    MODEL_DIR / "tokenizer.model",
    MODEL_DIR / "make_corpus.py",
    MODEL_DIR / "prepare_training_data.py",
    MODEL_DIR / "resume_train_gpu.py",
]
for path in required:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required ORBIT file: {path}")

initial_weights_sha = sha256(MODEL_DIR / "model.safetensors")

run([sys.executable, "-m", "pip", "install", "--quiet", "sentencepiece>=0.2.0", "safetensors>=0.4.0"])
run([sys.executable, "make_corpus.py"], cwd=MODEL_DIR)
run([sys.executable, "prepare_training_data.py"], cwd=MODEL_DIR)

started = time.time()
run([
    sys.executable, "resume_train_gpu.py",
    "--device", "cuda",
    "--steps", str(STEPS),
    "--batch-size", str(BATCH_SIZE),
    "--seq-len", str(SEQ_LEN),
    "--learning-rate", str(LEARNING_RATE),
], cwd=MODEL_DIR)

log_path = MODEL_DIR / "training_log.json"
history = json.loads(log_path.read_text(encoding="utf-8"))
last_record = history[-1] if history else {}
report = {
    "project": "ORBIT Seed",
    "training_mode": "continued training from existing checkpoint",
    "device": "cuda",
    "gpu": gpu_name,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "steps_added": STEPS,
    "starting_step": int(last_record.get("step", 0)) - STEPS,
    "ending_step": int(last_record.get("step", 0)),
    "batch_size": BATCH_SIZE,
    "sequence_length": SEQ_LEN,
    "learning_rate": LEARNING_RATE,
    "initial_weights_sha256": initial_weights_sha,
    "final_weights_sha256": sha256(MODEL_DIR / "model.safetensors"),
    "training_record": last_record,
    "elapsed_seconds": round(time.time() - started, 1),
    "note": "Validation loss measures next-token prediction on ORBIT's existing synthetic split; it is not a conversational quality benchmark.",
}
(MODEL_DIR / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

for name in [
    "model.safetensors",
    "tokenizer.model",
    "config.json",
    "training_log.json",
    "optimizer_state.pt",
    "training_report.json",
]:
    source = MODEL_DIR / name
    if not source.is_file():
        raise FileNotFoundError(f"Expected output not created: {name}")
    shutil.copy2(source, OUTPUT_DIR / name)

archive = WORK / "orbit-seed-trained.zip"
if archive.exists():
    archive.unlink()
shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=OUTPUT_DIR)

print("ORBIT_TRAINING_REPORT=" + json.dumps(report), flush=True)
print(f"OUTPUT_ARCHIVE={archive} ({archive.stat().st_size:,} bytes)", flush=True)
