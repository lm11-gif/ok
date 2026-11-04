# ORBIT Cognitive v2.1

**v2.1 is the latest experimental Cognitive release.** It replaces the badly
incoherent byte-level generator from v2 with a larger word-piece language
model. All four neural networks start from random initialisation. No
pretrained model weights, embeddings, tokenizer, or hidden states are loaded.

<badge>GPU training completed</badge>

- [Download ORBIT Cognitive v2.1 ZIP](https://github.com/lm11-gif/ok/releases/download/orbit-cognitive-v2.1/orbit-cognitive-v2.1.zip)
- [View the v2.1 release and all assets](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v2.1)
- [View the successful Kaggle GPU workflow](https://github.com/lm11-gif/ok/actions/runs/41776143029)

## Architecture

- **PerceptionNet** identifies likely entities and relations in a question.
- **AssociationNet** scores objects connected to subject/relation pairs in the small seed graph.
- **MemoryNet** retrieves relevant stored facts.
- **NeuralLanguageGenerator** is a causal Transformer trained to predict the next learned word/punctuation token. It has a UTF-8 byte fallback for unfamiliar words.
- The chat controller combines the user message, recent conversation, and retrieved facts into the generator prompt.

The generator is wider and deeper than v2, uses a longer token context, and is trained with document-isolated sequences and validation-based checkpoint selection.

## What changed after v2

The v2 model used a 257-symbol byte vocabulary, a 384-byte context, and random training windows drawn from a flattened stream of documents. Its validation byte loss was not a meaningful test of answer coherence, and direct tests showed that generated sentences were badly garbled.

v2.1 makes four main changes:

1. **Learned word/punctuation vocabulary.** Frequent pieces are represented as single tokens; unknown pieces fall back to UTF-8 bytes.
2. **Document-isolated training.** A sequence window is sampled from one conversation/document. It cannot cross from one unrelated example into another.
3. **More training capacity.** The default generator is 192-dimensional with six decoder layers, six attention heads, a 256-token context, and up to 40 epochs. The best validation checkpoint is retained.
4. **Generated-answer evaluation.** The training report records actual model-generated answers to evaluation prompts, alongside repetition and vocabulary-diversity diagnostics. These checks are useful signals, not a substitute for human evaluation.

## Training data and limitations

The data is original, locally generated synthetic dialogue and fact-grounded examples from the 67-fact seed graph. It includes everyday conversation, explanations, planning, short stories, roleplay, and uncertainty examples.

This is still a small, repetitive corpus. Training from scratch on it cannot create broad world knowledge or frontier-level reasoning. Better tokenisation and more training epochs may improve fluency, but coherence is not guaranteed. A successful training job or low next-token loss does not prove that the model follows instructions, answers questions correctly, or produces consistently coherent dialogue. Read the generated samples in `training_report.json` before judging the result.

## GPU training

The v2.1 Kaggle workflow completed successfully. It trains the randomly initialised PerceptionNet, AssociationNet, MemoryNet, and NeuralLanguageGenerator on an NVIDIA Tesla T4. The workflow publishes a separate release and preserves the older v1 and v2 releases.

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

1. Download `orbit-cognitive-v2.1.zip` from the [GitHub release](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v2.1).
2. Extract the ZIP into a folder.
3. Make sure Python and PyTorch are installed in the environment you intend to use.
4. Double-click `run_orbit.bat`.

Or run from PowerShell in the extracted folder:

    python chat.py --model . --device cpu

On a CUDA-compatible NVIDIA GPU with a CUDA-enabled PyTorch installation:

    python chat.py --model . --device cuda

The ZIP includes `tokenizer.json`; it must stay beside `generator.pt`. The v2 byte-level checkpoint is not compatible with the v2.1 generator. Use the matching source files and checkpoint from the same release.

## Evaluation and reporting

The release contains `training_report.json`, which records training configuration, validation results, specialist-network metrics, and generated-answer samples. Use those samples to assess whether the model's output is actually coherent. Do not use training loss alone as a quality score.

