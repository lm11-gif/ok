# ORBIT Seed 0.1 — experimental from-scratch language model

**This is a real trained neural language model, but it is not yet a coherent general-purpose chatbot.** The original evaluation found fragmented or irrelevant replies to held-out chat and roleplay prompts.

## Files tracked in this repository

- `model.safetensors` — trained weights (**binary file still needs to be added; see [UPLOAD_REMAINING.md](../UPLOAD_REMAINING.md)**)
- `tokenizer.model` — matching SentencePiece tokenizer (**binary file still needs to be added**)
- `config.json` — architecture and model dimensions
- `chat.py` — minimal local CPU inference loop
- `make_corpus.py` — generator for the original synthetic training corpus
- `prepare_training_data.py` — deterministic preparation of train/validation text splits
- `train.py` — from-scratch training script
- `resume_train.py` — CPU continuation from existing weights
- `resume_train_gpu.py` — CUDA continuation for Kaggle/Colab
- `requirements.txt`, Windows setup scripts, `training_log.json`, and `EVALUATION.md`

The generated `corpus.txt`, `train_corpus.txt`, and `validation_corpus.txt` are not checked in as multi-megabyte text files. They can be recreated using the included script:

```powershell
python make_corpus.py
```

The original split is produced by `train.py`. To regenerate just the split without starting a training run:

```powershell
python prepare_training_data.py
```

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
| Training corpus | 3,186 synthetic examples |
| Training tokens | 481,532 |
| Validation tokens | 30,997 |

The training text is a small, locally authored synthetic corpus of short conversations, explanations, roleplay exchanges, and stories. No external pretrained weights or copied training corpus were used.

## Run it on Windows

1. Extract the GPU-ready package or the original model archive.
2. Add the two binary files listed above if they are not already present.
3. Install Python 3.10 or newer.
4. Double-click `setup_windows.bat` once.
5. Double-click `run_windows.bat` to start a chat.

Or use a terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chat.py
```

A one-shot test can be run with:

```powershell
python chat.py --one-shot "What is an algorithm?"
```

## Continue training

The supplied checkpoint is already trained. For a short CPU continuation:

```powershell
python resume_train.py --steps 100
```

For hosted GPU continuation, see [KAGGLE_GPU_GUIDE.md](KAGGLE_GPU_GUIDE.md) and use [ORBIT_Kaggle_GPU_Training.ipynb](../training/ORBIT_Kaggle_GPU_Training.ipynb).

The first GPU run creates a fresh optimizer state because the original archive did not include the old optimizer state. Subsequent runs can save and resume it.

## Limitations

- Not suitable for coding, factual question answering, or dependable open-ended conversation.
- The model has very little world knowledge because the training corpus is small and synthetic.
- Its 160-token context is short.
- The validation score is only a next-token metric on a dataset with repeated structures; it does not measure whether answers are coherent or useful.
- The checkpoint is an educational prototype and a starting point for experiments, not a replacement for a modern pretrained model.
