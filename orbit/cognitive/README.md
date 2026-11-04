# ORBIT Cognitive v2.3 Conversational Candidate

**v2.3 is a separate experimental branch, not a replacement for the published v2.1 release.** It is intended to improve real multi-turn conversation by giving the randomly initialised model a broader dialogue corpus, first teaching general transcript language patterns, then focusing training on assistant replies, and finally testing generated-history rollouts.

No pretrained model weights, embeddings, hidden states or pretrained tokenizers are loaded. The student model starts from random initialisation; external resources are used only as text training data.

## Changes in v2.3

- **Broader dialogue data.** Adds a seeded sample of up to 25,000 conversations from UltraChat 200k in addition to OASST1 and English Daily Dialogues 10K.
- **Two-stage language training.** First, the model learns next-token patterns across the entire multi-turn transcript. It then fine-tunes with loss focused on assistant replies and turn-ending markers.
- **Conversation-level splits.** OASST1 validation is held out by conversation tree; UltraChat uses a separate prompt-ID validation split and a separate test_sft file; Daily Dialogues uses its own train/validation/test splits.
- **True multi-turn rollout evaluation.** For each rollout, the model's own generated answer is fed into the following turn. Ground-truth reference replies are never inserted into generated history.
- **Dataset provenance.** The final report records pinned revisions, checksums, downloaded byte counts, accepted rows and split sizes.

## Training data

1. **OpenAssistant OASST1** — English, non-deleted, non-synthetic, review-approved conversation messages from ready-for-export trees. Apache-2.0. [Dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1)
2. **English Daily Dialogues 10K** by Enes Deniz — synthetic, reviewed multi-turn everyday scenarios. CC-BY-4.0. Health/wellness and shopping/errands examples remain excluded. [Dataset card](https://huggingface.co/datasets/3nesdeniz/english-daily-dialogues-10k)
3. **UltraChat 200k** — filtered multi-turn English supervised conversation dataset, MIT. A seeded sample of up to 25,000 conversations is loaded from the first train_sft shard; the distinct test_sft shard is reserved for evaluation. This dataset was used as text for training a scratch-initialised model; no Zephyr or other model weights are loaded. [Dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

The pinned dataset revisions, files, checksums and licence information are written into the training report and `DATA_SOURCES.txt` packaged alongside the model.

## Model and training configuration

The candidate retains the 256-dimensional, 8-layer, 8-head causal Transformer, with 512-token context and a learned word/punctuation vocabulary with UTF-8 byte fallback. The default training settings are:

    --generator-epochs 30
    --generator-batch-size 16
    --generator-vocab-size 8192
    --generator-dim 256
    --generator-layers 8
    --generator-heads 8
    --generator-block-size 512
    --generator-learning-rate 0.00035
    --ultrachat-max-conversations 25000

The first five epochs (or the bounded warmup setting in the training code) model all transcript tokens. Remaining epochs focus loss on assistant output. All language-model student weights are randomly initialised. Kaggle T4 GPU training uses CUDA mixed precision where available, fused AdamW when supported, gradient clipping, a cosine learning-rate schedule and validation-loss checkpoint selection.

## Evaluation

The report stores:
- assistant-only validation loss and data provenance;
- generated replies beside held-out reference replies;
- basic open-ended prompts;
- multi-turn autoregressive rollouts where each generated reply becomes the next turn's history;
- repeated n-gram, unique-word and replacement-character diagnostics.

The diagnostics are not a substitute for human evaluation. Lower loss does not prove coherence. Check whether answers address the latest user message, remain grammatical, use conversation history, and avoid unrelated fact memorisation.

## Release status

The v2.3 branch is under test. The separate candidate GPU workflow publishes a prerelease only after training and archive validation. It does not overwrite v1, v2, or v2.1.

The latest stable Cognitive release remains [ORBIT Cognitive v2.1](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v2.1). The v2.2 baseline training run remains separate.

