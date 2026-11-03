# ORBIT Seed v0.2

ORBIT Seed v0.2 is the conversational branch of ORBIT. It is built by
fine-tuning the Apache-2.0 SmolLM2-360M pretrained base model on the
Apache-2.0 Smol-SmolTalk dataset.

**This version is not trained from scratch.** The original 871k-parameter
from-scratch experiment remains in `orbit/seed_model/` as v0.1. The v0.2
branch uses a pretrained 360M-parameter language model because the old model
was too small and its synthetic training set was too repetitive to generate
reliably coherent responses.

## Training recipe

- Base: `HuggingFaceTB/SmolLM2-360M` (Apache-2.0)
- Conversation tokenizer/template: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Dataset: `HuggingFaceTB/smol-smoltalk` (Apache-2.0; adapted for models below 1B parameters)
- Method: LoRA supervised fine-tuning, merged into a standalone model
- Initial recipe: 20,000 training conversations, 1 epoch, sequence length 512
- Hardware: Kaggle Tesla T4 GPU

The release archive contains the merged model, tokenizer, notices, and a
training report with held-out evaluation and generated sample answers.

## Download

When training completes, get the `orbit-seed-v0.2.zip` asset from:
https://github.com/lm11-gif/ok/releases/tag/orbit-seed-v0.2

Extract it, install the packages, and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch transformers safetensors accelerate sentencepiece
python chat.py --model .
```

For CPU inference on the first laptop, the 360M model is a much lighter
starting point than large desktop models. It can still be slower than a
quantized GGUF build; the release currently provides the Transformers
checkpoint.

## Evaluation

Do not judge v0.2 solely by training loss. Inspect
`training_report.json` for held-out evaluation loss and generated smoke-test
answers. Run your own conversations and compare them with v0.1.

## Attribution and licenses

- Base model: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
- Conversation data: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- See `NOTICE.md` and `UPSTREAM_LICENSE` in the release archive.
