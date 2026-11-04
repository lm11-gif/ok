# ORBIT Cognitive v2.2 Conversational

**v2.2 is an active scratch-trained conversational rebuild.** It is intended to replace the poorly coherent v2.1 generator, but it is not considered ready until its GPU training run completes and its generated held-out conversations have been inspected.

All student networks are randomly initialised. No pretrained language-model weights, embeddings, tokenizer or hidden states are loaded.

## What is changing

- **Assistant-only training loss.** The generator is trained to predict assistant replies and response-ending markers, rather than spending most of its objective predicting user prompts and formatting tokens.
- **Real multi-turn histories.** Each supervised assistant reply is paired with the preceding turns from its conversation tree. It is not reduced to an unrelated isolated prompt/answer pair.
- **Conversation-level holdout.** OASST1 validation is split by conversation tree, while the Daily Dialogues test split is held out for generated-reply evaluation.
- **Longer context and more capacity.** The current candidate uses a 256-dimensional, 8-layer, 8-head causal Transformer with a 512-token context and learned word/punctuation vocabulary with UTF-8 byte fallback.
- **Actual conversation checks.** Training reports keep generated replies beside held-out reference answers and include basic open-ended prompts. Heuristic word statistics are diagnostic only.

## Training data

The new candidate uses three sources:

1. **OpenAssistant OASST1** — English, non-deleted, non-synthetic, review-approved messages from conversation trees marked ready for export. The original dataset is licensed Apache-2.0. [Dataset card and license](https://huggingface.co/datasets/OpenAssistant/oasst1)
2. **English Daily Dialogues 10K** by Enes Deniz — 10,000 synthetic English multi-turn dialogues spanning everyday domains, licensed CC-BY-4.0. Health/wellness and shopping/errands domains are excluded from this training run. [Dataset card and license](https://huggingface.co/datasets/3nesdeniz/english-daily-dialogues-10k)
3. **Original ORBIT-authored examples** — additional explanations, stories, roleplay and seed-graph fact prompts.

Dataset revisions are pinned in the training code. The final training report records downloaded file checksums, row counts, filtering and split details. The model package includes `DATA_SOURCES.txt` attribution.

The external datasets add much more varied dialogue than v2.1's small synthetic-only corpus, but they are still finite and imperfect. They do not guarantee broad knowledge, reliable reasoning or human-level conversation.

## Architecture and training

The system contains four randomly initialised neural networks:

- **PerceptionNet** estimates entities and relations in a question.
- **AssociationNet** scores objects associated with subject/relation pairs in the small seed graph.
- **MemoryNet** retrieves relevant stored facts.
- **NeuralLanguageGenerator** generates the response from the current user message, preceding conversation and retrieved context.

The candidate training defaults are:

    --generator-epochs 30
    --generator-batch-size 16
    --generator-vocab-size 8192
    --generator-dim 256
    --generator-layers 8
    --generator-heads 8
    --generator-block-size 512
    --generator-learning-rate 0.00035

Normal training uses Kaggle's NVIDIA Tesla T4 GPU, CUDA mixed precision when available, fused AdamW where supported, gradient clipping, a cosine learning-rate schedule and validation-based checkpoint selection. CPU is used only for CI smoke tests.

The language loss is computed only over assistant response text and response-ending markers. Training batches are selected so each contains assistant targets.

## Current release status

- **Latest published Cognitive release:** [ORBIT Cognitive v2.1](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v2.1)
- **v2.2 candidate source:** available on the repository's `main` branch.
- **v2.2 model ZIP:** not published yet. A separate release will be created only after the Kaggle run succeeds and the held-out generated replies are reviewed.

## Evaluation standard

A lower validation loss is not enough. The report must show the assistant's generated response beside the held-out human or synthetic reference reply, including the previous conversation turns. Check whether the response addresses the latest message, follows conversational context, uses grammatical sentences, avoids irrelevant facts and remains consistent across turns.

Do not describe v2.2 as coherent or ready merely because training finishes. The actual replies must support that conclusion.

## Running a published model

For the currently published v2.1 package, use the [v2.1 release page](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v2.1). The older byte-level v2 and word-piece v2.1 checkpoints are not interchangeable with the v2.2 candidate architecture.
