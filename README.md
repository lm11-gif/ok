# ORBIT workspace

This repository now houses the ORBIT AI Playground and ORBIT Seed research model.

## Projects

- [AI Playground](orbit/ai_playground/README.md) — browser-based front end for Ollama and OpenAI-compatible models.
- [ORBIT Seed 0.1](orbit/seed_model/README.md) — the small decoder-only Transformer trained from random initialization.
- [Kaggle GPU continuation guide](orbit/seed_model/KAGGLE_GPU_GUIDE.md) — instructions for continuing the existing Seed checkpoint on a hosted GPU.
- [Kaggle notebook](orbit/training/ORBIT_Kaggle_GPU_Training.ipynb) — GPU continuation notebook.
- [GitHub binary uploader](orbit/tools/github-binary-uploader.html) — browser utility for uploading ORBIT's weights and tokenizer.

## Important model status

ORBIT Seed 0.1 is an educational prototype with 871,552 parameters, a 160-token context, and 2,400 recorded CPU training updates. Its evaluation found that generated answers were not coherent enough for dependable conversation. Lower training loss on its small synthetic corpus does not establish general language ability.

The app source, model architecture, inference and training scripts, corpus generator, documentation, and Kaggle notebook are tracked here. The multi-megabyte synthetic corpus files can be regenerated locally with `orbit/seed_model/make_corpus.py`. The original binary checkpoint and SentencePiece model are not included in this commit because the connected GitHub actions available in this session cannot upload arbitrary binary files. Add these two files from the original model archive to `orbit/seed_model/`:
- `model.safetensors`
- `tokenizer.model`

Do not replace them with placeholders. The GPU continuation script requires both files. Follow [the remaining binary asset upload steps](orbit/UPLOAD_REMAINING.md) to add them through GitHub's web uploader.

## Repository access

The connected GitHub integration has verified read and push permissions for this repository. Changes made through it are committed directly to `main`.
