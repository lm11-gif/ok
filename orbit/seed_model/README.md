# ORBIT Seed 0.1 — experimental from-scratch language model

**This is a real trained neural language model, but it is not yet a coherent general-purpose chatbot.** The original evaluation found fragmented or irrelevant replies to held-out chat and roleplay prompts.

## Files

- `model.safetensors` — trained weights (**must be added separately; binary upload is not supported by the connected GitHub action**)
- `tokenizer.model` — trained SentencePiece tokenizer (**must be added separately**)
- `tokenizer.vocab` — tokenizer vocabulary text
- `config.json` — architecture and model dimensions
- `chat.py` — minimal local inference
- `make_corpus.py` — generator for the original synthetic training corpus
- `corpus.txt`, `train_corpus.txt`, `validation_corpus.txt` — original text and fixed train/validation split
- `train.py` — from-scratch training script
- `resume_train.py` — CPU continuation
- `resume_train_gpu.py` — CUDA continuation for Kaggle/Colab
- `training_log.json` and `EVALUATION.md` — recorded training history and evaluation

## Model details

| Property | Value |
|---|---:|
| Architecture | Decoder-only causal Transformer |
| Parameters | 871,552 |
| Layers | 3 |
| Attention heads | 4 |
| Hidden size | 128 |
| Context window | 160 tokens |
| Tokenizer | SentencePiece BPE, 2,000 pieces |
| Training updates | 2,400 |
| Training examples | 3,186 synthetic examples |
| Training tokens | 481,532 |
| Validation tokens | 30,997 |

The data is a small, locally authored synthetic corpus. No external pretrained weights were used.

## Run on Windows

Install Python 3.10 or newer, then:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chat.py
```

## Continue training

The checkpoint is already trained. To continue it on CPU:

```powershell
python resume_train.py --steps 100
```

For a hosted GPU, follow [KAGGLE_GPU_GUIDE.md](KAGGLE_GPU_GUIDE.md) and use [resume_train_gpu.py](resume_train_gpu.py).

The small synthetic corpus will overfit quickly. Better and more diverse training data is more important than simply increasing the number of updates.
