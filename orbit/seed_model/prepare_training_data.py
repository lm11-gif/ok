"""Recreate the fixed ORBIT Seed train/validation text split without starting training."""
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
source = ROOT / "corpus.txt"
if not source.exists():
    raise FileNotFoundError("corpus.txt is missing; run python make_corpus.py first.")

random.seed(73421)
documents = [item.strip() for item in source.read_text(encoding="utf-8").split("\n\n") if item.strip()]
random.shuffle(documents)
cut = max(1, int(len(documents) * 0.94))
train_docs, validation_docs = documents[:cut], documents[cut:]

(ROOT / "train_corpus.txt").write_text("\n\n".join(train_docs), encoding="utf-8")
(ROOT / "validation_corpus.txt").write_text("\n\n".join(validation_docs), encoding="utf-8")

print(f"Prepared {len(train_docs):,} training examples and {len(validation_docs):,} validation examples.")
