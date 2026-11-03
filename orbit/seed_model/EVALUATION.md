# ORBIT Seed 0.1 — training and generation evaluation

## Training run

- Randomly initialized decoder-only Transformer; no pretrained weights.
- 871,552 trainable parameters.
- 2,000-piece SentencePiece BPE tokenizer trained on the project corpus.
- 3,186 original synthetic examples.
- Fixed split: 481,532 training tokens and 30,997 validation tokens.
- 2,400 optimizer updates, trained on CPU.

Validation cross-entropy fell from **7.6263** at the initial checkpoint to **0.7339** at update 2,400. This means the model learned to predict repeated patterns in this corpus. It does not imply broad language competence.

## Generation checks

| Prompt | Observed behaviour |
|---|---|
| “What is an algorithm?” | Produced a short fragment rather than a clear definition. |
| “I want to learn programming, but I keep getting stuck. What should I do?” | Produced an unrelated fragment. |
| A simple explorer roleplay setup | Mixed unrelated phrases and did not maintain the scene. |

**Result: generation quality is inadequate.** More steps on the same corpus are unlikely to fix this; it needs a much larger and more diverse high-quality dataset, more parameters, and substantially more compute.

## Why the score and outputs disagree

The corpus is synthetic and contains repeated response structures. A small model can learn repeated token patterns and achieve a low validation loss while still failing to generalize to new prompts. This is why generation checks are reported alongside the loss.
