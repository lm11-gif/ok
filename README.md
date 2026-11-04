# ORBIT workspace

This repository houses the ORBIT AI Playground and several deliberately
separate ORBIT Seed and ORBIT Cognitive model experiments.

## Projects

- [AI Playground](orbit/ai_playground/README.md) — browser-based front end for Ollama and OpenAI-compatible models.
- [ORBIT Seed v0.1](orbit/seed_model/README.md) — the original 871k-parameter model trained from random initialization. It is an educational prototype, not a coherent general-purpose assistant.
- [ORBIT Seed v0.2](orbit/seed_model_v0.2/README.md) — a separate conversational model fine-tuned from pretrained SmolLM2-360M. This version is **not** scratch-trained.
- [ORBIT Seed v0.3 Scratch](orbit/seed_model_v0.3_scratch/README.md) — a separate model with random student-weight initialization, its own tokenizer, FineWeb-Edu pretraining, assistant tuning, and a scratch-specific teacher DPO stage.
- [ORBIT Cognitive v2.2 Conversational](orbit/cognitive/README.md) — the active conversational rebuild: assistant-only training targets, reconstructed multi-turn histories, a larger context, public dialogue datasets and held-out conversation evaluation. Its GPU run is pending CI; v2.1 remains the latest published Cognitive release until v2.2 is trained and evaluated.

## Model lineage

These are separate experiments, not successive checkpoints of one model. In particular, ORBIT Seed v0.2 uses pretrained weights, while the v0.3 Scratch and Cognitive experiments start their student networks from random initialization.

| Version | Student initialization | Training approach |
|---|---|---|
| Seed v0.1 | Random | Small synthetic corpus; 871,552 parameters |
| Seed v0.2 | Pretrained SmolLM2-360M | Supervised fine-tuning and separate teacher DPO |
| Seed v0.3 Scratch | Random | FineWeb-Edu pretraining, assistant tuning, then DPO using teacher-written preferences |
| Cognitive v1 | Random | Three specialist networks and a small seed knowledge graph; historical release |
| Cognitive v2 | Random | Byte-level causal language generator; historical baseline, with poor observed text coherence |
| Cognitive v2.1 | Random | Learned word-piece generator trained on a small authored corpus; still incoherent on direct tests |
| Cognitive v2.2 Conversational | Random | Assistant-only causal language modelling on reconstructed multi-turn OASST1 and Daily Dialogues, with conversation-level holdouts |
| Cognitive v2.1 | Random | Learned word/punctuation vocabulary with UTF-8 byte fallback; document-isolated next-token training and generated-answer evaluation |

For Seed v0.3, SmolLM3-3B is used only to generate additional training and preference text. Its pretrained weights are not loaded into the ORBIT student. The v0.3 configuration and training report explicitly record that the student is scratch-initialized.

For Cognitive v2.1, all four neural networks (PerceptionNet, AssociationNet, MemoryNet, and NeuralLanguageGenerator) are randomly initialised. No pretrained language-model weights, embeddings, tokenizer, or hidden states are loaded. The learned word-piece vocabulary is built from the training corpus; it is not a pretrained tokenizer.

## Training

- [ORBIT v0.2 Kaggle fine-tuning workflow](.github/workflows/train-orbit-v02.yml)
- [ORBIT v0.2 teacher DPO workflow](.github/workflows/train-orbit-v02-dpo.yml)
- [ORBIT v0.3 Scratch Kaggle workflow](.github/workflows/train-orbit-v03-scratch.yml)
- [ORBIT v0.3 Scratch DPO workflow](.github/workflows/train-orbit-v03-scratch-dpo.yml)
- [ORBIT Cognitive v2.1 Kaggle GPU training workflow](.github/workflows/train-orbit-cognitive-kaggle.yml)
- [ORBIT Cognitive CI/test workflow](.github/workflows/check-orbit-cognitive.yml)

The v0.3 scratch training workflow is configured to run after the current v0.2 fine-tuning/DPO sequence completes, to avoid two Kaggle jobs competing for the same free GPU allocation. It can also be started manually from GitHub Actions.

The Cognitive v2.1 GPU training run completed and its release is preserved. The v2.2 conversational rebuild is in progress: it adds public dialogue datasets and assistant-only loss, and will publish a separate release only after the new GPU run and generated conversation samples are evaluated. Older releases are not overwritten.

## Important limitations

The v0.3 Scratch design is an intentionally small first model (approximately
25M parameters, 512-token context, up to 100M pretraining tokens per initial
run) sized for a free Kaggle T4. It is genuinely trained from random weights,
but one initial run will not make it comparable to large pretrained
assistants. Expect it to need substantially more data and training time, and
judge it from its held-out results and actual generated answers rather than
training loss alone.

Cognitive v2.1 is also an experimental small-scale model. Its dialogue corpus
is synthetic and relatively narrow, and its fact graph contains only 67 seed
facts. Successful GPU training and a lower validation loss do not establish
general language competence, reliable reasoning, or factual accuracy. Inspect
the generated-answer samples in the release's `training_report.json` before
making claims about its coherence.

## Releases

- [ORBIT Cognitive v2.1 — latest published Cognitive model and downloadable ZIP](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v2.1)
- [ORBIT Cognitive v2 — byte-level baseline](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v2)
- [ORBIT Cognitive v1 — original specialist-network release](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v1)

## Development handoff

- [README_FOR_AI.md](README_FOR_AI.md) — detailed project context, model lineage, training workflows, verified status, and instructions for future AI assistants/developers.
