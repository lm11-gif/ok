# ORBIT workspace

This repository houses the ORBIT AI Playground and several deliberately
separate ORBIT Seed model experiments.

## Projects

- [AI Playground](orbit/ai_playground/README.md) — browser-based front end for Ollama and OpenAI-compatible models.
- [ORBIT Seed v0.1](orbit/seed_model/README.md) — the original 871k-parameter model trained from random initialization. It is an educational prototype and is not yet a coherent general-purpose assistant.
- [ORBIT Seed v0.2](orbit/seed_model_v0.2/README.md) — a separate conversational model fine-tuned from pretrained SmolLM2-360M. This version is **not** scratch-trained.
- [ORBIT Seed v0.3 Scratch](orbit/seed_model_v0.3_scratch/README.md) — a separate model with random student-weight initialization, its own tokenizer, FineWeb-Edu pretraining, assistant tuning, and a scratch-specific teacher DPO stage.
- [ORBIT Cognitive v1](orbit/cognitive/README.md) — first implementation of three randomly initialised specialist networks, with [architecture notes](orbit/cognitive/COGNITIVE_ARCHITECTURE.md), GPU training code, inference, tests, and a manual Kaggle workflow. Not yet GPU-trained.

## Model lineage

| Version | Student initialization | Training approach |
|---|---|---|
| v0.1 | Random | Small synthetic corpus; 871,552 parameters |
| v0.2 | Pretrained SmolLM2-360M | Supervised fine-tuning and separate teacher DPO |
| v0.3 Scratch | Random | FineWeb-Edu pretraining, assistant tuning, then DPO using teacher-written preferences |

For v0.3, SmolLM3-3B is used only to generate additional training and preference text. Its pretrained weights are not loaded into the ORBIT student. The v0.3 configuration and training report explicitly record that the student is scratch-initialized.

## Training

- [ORBIT v0.2 Kaggle fine-tuning workflow](.github/workflows/train-orbit-v02.yml)
- [ORBIT v0.2 teacher DPO workflow](.github/workflows/train-orbit-v02-dpo.yml)
- [ORBIT v0.3 Scratch Kaggle workflow](.github/workflows/train-orbit-v03-scratch.yml)
- [ORBIT Cognitive v1 GPU training workflow](.github/workflows/train-orbit-cognitive-kaggle.yml)
- [ORBIT Cognitive v1 CI/test workflow](.github/workflows/check-orbit-cognitive.yml)
- [ORBIT v0.3 Scratch DPO workflow](.github/workflows/train-orbit-v03-scratch-dpo.yml)

The v0.3 scratch training workflow is configured to run after the current v0.2 fine-tuning/DPO sequence completes, to avoid two Kaggle jobs competing for the same free GPU allocation. It can also be started manually from GitHub Actions.

## Important limitations

The v0.3 Scratch design is an intentionally small first model (approximately
25M parameters, 512-token context, up to 100M pretraining tokens per initial
run) sized for a free Kaggle T4. It is genuinely trained from random weights,
but one initial run will not make it comparable to large pretrained
assistants. Expect it to need substantially more data and training time, and
judge it from its held-out results and actual generated answers rather than
training loss alone.

## Development handoff

- [README_FOR_AI.md](README_FOR_AI.md) — detailed project context, model lineage, training workflows, verified status, and instructions for future AI assistants/developers.
