# ORBIT Seed v0.2

ORBIT Seed v0.2 is the conversational branch of ORBIT. It starts from the
Apache-2.0 SmolLM2-360M pretrained model and is fine-tuned for assistant
conversations.

**This version is not trained from scratch.** The original 871k-parameter
from-scratch experiment remains in `orbit/seed_model/` as v0.1. The v0.2
branch uses a pretrained 360M-parameter language model because the old model
was too small and its synthetic training set was too repetitive to generate
reliably coherent responses.

## Training pipeline

1. **Supervised fine-tuning (SFT):** SmolLM2-360M is fine-tuned on
   HuggingFaceTB/smol-smoltalk. The loss is masked so the model learns from
   assistant replies, not from reproducing user messages.
2. **Teacher preference generation:** SmolLM3-3B reviews ORBIT's draft answers
   to a curated set of prompts and writes an improved answer. ORBIT's draft is
   the rejected completion; the teacher-reviewed answer is the chosen one.
3. **Direct Preference Optimization (DPO):** TRL's DPOTrainer uses these
   prompt/chosen/rejected pairs to adjust the SFT model. The teacher is
   unloaded before DPO training to reduce GPU memory use.

The teacher is larger than ORBIT and is Apache-2.0 licensed, but it is not an
infallible judge. Its generated answers may still be wrong. The DPO report
includes the held-out preference loss, smoke-test answers, and limitations;
the release archive also includes the generated preference pairs for review.

## SFT recipe

- Base: `HuggingFaceTB/SmolLM2-360M` (Apache-2.0)
- Conversation tokenizer/template: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Dataset: `HuggingFaceTB/smol-smoltalk`
- Method: LoRA supervised fine-tuning, merged into a standalone model
- Initial recipe: 20,000 training conversations, 1 epoch, sequence length 512
- Hardware: Kaggle Tesla T4 GPU

## DPO recipe

- Student: completed ORBIT Seed v0.2 SFT checkpoint
- Teacher: `HuggingFaceTB/SmolLM3-3B` (Apache-2.0)
- Preference data: up to 145 curated prompts, ORBIT draft vs teacher-reviewed answer
- Method: LoRA DPO with TRL 0.15.2
- Hardware: Kaggle Tesla T4 GPU; teacher generation and DPO training run sequentially

## Download

The first-stage SFT checkpoint is published at:
https://github.com/lm11-gif/ok/releases/tag/orbit-seed-v0.2

The teacher-supervised DPO checkpoint will be published separately at:
https://github.com/lm11-gif/ok/releases/tag/orbit-seed-v0.2-dpo

Extract the selected archive, install the packages, and run:

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch transformers safetensors accelerate sentencepiece
python chat.py --model .
```

For CPU inference on the first laptop, the 360M model is a much lighter
starting point than large desktop models. It can still be slower than a
quantized GGUF build; the release currently provides the Transformers
checkpoint.

## Evaluation

Do not judge ORBIT solely by training or DPO loss. Inspect the training
reports, generated smoke-test answers, and `preferences.jsonl`. The teacher
may produce mistakes, and DPO preference loss is not a factuality score.
Compare the SFT and DPO checkpoints on the same prompts before deciding
whether the DPO version is actually better.

## Attribution and licenses

- Base model: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
- Conversation data: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- Teacher model: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- See `NOTICE.md` and the upstream license material in the release archives.
