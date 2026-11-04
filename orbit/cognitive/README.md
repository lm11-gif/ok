# ORBIT Cognitive v2.1

**v2.1 replaces the incoherent byte-level generator from v2 with a larger word-piece language model.** All four neural networks still start from random initialisation. No pretrained weights, embeddings, tokenizer, or hidden states are loaded.

## Architecture

- PerceptionNet identifies likely entities and relations in a question.
- AssociationNet scores objects connected to subject/relation pairs in the small seed graph.
- MemoryNet retrieves relevant stored facts.
- NeuralLanguageGenerator is a causal Transformer trained to predict the next learned word/punctuation token. It has a UTF-8 byte fallback for words not in the learned vocabulary.
- The chat controller adds the user message, recent conversation, and retrieved facts to the prompt.

The v2.1 generator uses a wider/deeper network than v2, a longer token context, a corpus-trained vocabulary, and a training schedule with validation-loss checkpoint selection.

## What changed after v2

The v2 model used a 257-symbol byte vocabulary, a 384-byte context, and random training windows drawn from a flattened stream of documents. Its validation byte loss was not a meaningful test of answer coherence, and direct tests showed that generated sentences were badly garbled.

v2.1 makes four changes:

1. **Learned word/punctuation vocabulary.** Frequent pieces are represented as single tokens; unknown pieces fall back to UTF-8 bytes.
2. **Document-isolated training.** A sequence window is always sampled from one conversation/document. It cannot cross from one unrelated example into another.
3. **More training capacity.** The default generator is 192-dimensional with six decoder layers, six attention heads, 256-token context, and up to 40 epochs. Early stopping retains the best validation checkpoint.
4. **Prompt-based checks.** The training report records actual model-generated answers to held-out-style prompts, plus simple repetition and vocabulary-diversity diagnostics. These checks are not a substitute for human evaluation.

## Training data and limitations

The data is original, locally generated synthetic dialogue and fact-grounded examples from the 67-fact seed graph. It includes everyday conversation, explanations, planning, short stories, roleplay and uncertainty examples. It is still a small, repetitive corpus. Training from scratch on it cannot create broad world knowledge or frontier-level reasoning. Better tokenisation and more epochs can improve fluency, but coherence is not guaranteed.

## GPU training

The Kaggle workflow builds a self-contained training script and trains on an NVIDIA Tesla T4. Normal training requires CUDA; CPU is only used for CI smoke tests.

Default generator settings:

    --generator-epochs 40
    --generator-batch-size 32
    --generator-vocab-size 8192
    --generator-dim 192
    --generator-layers 6
    --generator-heads 6
    --generator-block-size 256

Training uses CUDA mixed precision when available, fused AdamW where supported, gradient clipping, a cosine learning-rate schedule, GPU-resident tokenised documents, and validation-loss checkpoint selection.

## Run the model

Download the **ORBIT Cognitive v2.1** ZIP from the GitHub release, extract it, then double-click `run_orbit.bat`.

Or run from PowerShell in the extracted folder:

    python chat.py --model . --device cpu

On a CUDA-compatible NVIDIA GPU:

    python chat.py --model . --device cuda

The ZIP includes `tokenizer.json`; it must stay beside `generator.pt`. The v2 byte-level checkpoint is not compatible with the v2.1 generator, and the two releases remain separate.

## Honest evaluation

The training report includes generated answer samples. Read those samples before judging the model. A lower next-token loss alone does not prove that responses are coherent, follow instructions, or are factually accurate. The project should only be described as improved if the actual sample answers support that conclusion.

