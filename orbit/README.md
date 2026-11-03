# ORBIT

ORBIT is split into two separate parts:

1. **AI Playground** — the browser interface for connecting external/local models.
2. **ORBIT Seed** — an experimental language model trained from random initialization.

The playground does not embed model weights. It connects to a model through Ollama or an OpenAI-compatible endpoint.

## Contents

- `ai_playground/` — HTML app and its setup notes.
- `seed_model/` — ORBIT Seed architecture, tokenizer vocabulary, training scripts, evaluation notes, and training data.
- `training/` — Kaggle GPU notebook.

## Checkpoint files that must be added

The binary assets `seed_model/model.safetensors` and `seed_model/tokenizer.model` are still missing from the GitHub commit. Upload them from the original `orbit-seed-gpu-ready.zip` archive using GitHub's **Add file → Upload files** page. The rest of the source and instructions are tracked here.
