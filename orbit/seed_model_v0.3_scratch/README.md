# ORBIT Seed v0.3 Scratch

**This is the genuinely from-scratch ORBIT track.** It is separate from the
v0.2 SmolLM2 fine-tune. The student model is a decoder-only Transformer whose
weights are initialized randomly. It does not load SmolLM2, SmolLM3, or any
other pretrained student checkpoint.

The larger teacher is used only to write additional training examples. Its
weights are never copied into ORBIT. Learning from text written by a teacher
is still training a randomly initialized student from scratch; it is data
distillation, not weight fine-tuning.

## Planned training stages

1. **Tokenizer training:** ORBIT trains its own SentencePiece BPE tokenizer
   from the pretraining corpus.
2. **Pretraining from random initialization:** ORBIT learns next-token
   prediction from a bounded stream of the educational FineWeb-Edu corpus.
3. **Assistant tuning:** ORBIT is trained on assistant conversations from
   Smol-SmolTalk and extra answers written by SmolLM3-3B. The loss is masked
   to apply to assistant replies, not user prompts.
4. **Evaluation:** the training report records train/validation losses and
   actual generated answers. These are diagnostics, not proof that the model
   is already a good assistant.

## Initial model and compute budget

- Parameters: approximately 25 million (exact count is written to config).
- Layers: 8.
- Hidden width: 448.
- Attention heads: 7.
- Context window: 512 tokens.
- Vocabulary: up to 12,000 pieces, trained from scratch.
- Initial pretraining budget: up to 100 million FineWeb-Edu tokens.
- Student weights: random initialization.
- Teacher: SmolLM3-3B, used to generate text examples only.
- GPU: Kaggle Tesla T4.

This is a deliberately small first scratch model to fit a free T4 run. A
single 100M-token run is only an initial training phase. It will not match a
multi-billion-parameter assistant trained on trillions of tokens. More
training data, compute, and later preference tuning will be needed.

## Train

The GitHub Actions workflow is named "Train ORBIT Seed v0.3 Scratch on Kaggle GPU".
It can be started manually from the Actions tab, or by updating
START_TRAINING.txt. The workflow uses the existing Kaggle repository
secrets; do not put API keys in this folder or commit them.

When it finishes, the checkpoint is published at:

https://github.com/lm11-gif/ok/releases/tag/orbit-seed-v0.3-scratch

## Run locally

Download and extract the release archive, then:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chat.py --model .
```

For CPU inference, run:

```powershell
python chat.py --model . --device cpu
```

## What "from scratch" means here

- The student Transformer starts from newly randomized weights.
- Its tokenizer is trained from the raw training corpus.
- FineWeb-Edu supplies pretraining text; Smol-SmolTalk supplies conversational
  training examples.
- SmolLM3-3B supplies additional text targets only. No teacher model weights
  are loaded into the student or used as its initialization.
- The model's configuration and training report explicitly record
  from_scratch: true and pretrained_student_weights: false.

## Data and model notices

- FineWeb-Edu: ODC-By 1.0. See
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Smol-SmolTalk: see
  https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- Teacher text generator: SmolLM3-3B, Apache-2.0. See
  https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- The release archive includes NOTICE.md and the training report.
