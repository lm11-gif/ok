# Continue ORBIT Seed on a free Kaggle GPU

This procedure continues the existing ORBIT Seed checkpoint; it does not initialize a new model.

## 1. Upload ORBIT as a private Kaggle Dataset

Upload the `orbit-seed-gpu-ready.zip` package or extracted `orbit_seed_model` folder as a private Kaggle dataset. Ensure it contains `model.safetensors`, `tokenizer.model`, `train_corpus.txt`, `validation_corpus.txt`, and `resume_train_gpu.py`.

## 2. Create a GPU notebook

1. Create a notebook at https://www.kaggle.com/code.
2. Attach the private ORBIT dataset using **Add Input**.
3. In Notebook **Settings → Accelerator**, select a GPU.
4. Run:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    raise RuntimeError("Enable a GPU accelerator and restart the session.")
```

## 3. Run continuation training

Use the notebook [ORBIT_Kaggle_GPU_Training.ipynb](../training/ORBIT_Kaggle_GPU_Training.ipynb), which:
- verifies that CUDA is enabled,
- extracts the GPU-ready archive safely,
- installs `sentencepiece` and `safetensors`,
- runs `resume_train_gpu.py` for an initial 100-step smoke test,
- packages the updated checkpoint and optimizer state for download.

The first GPU run creates a fresh optimizer because the original archive did not include optimizer state. Subsequent runs save and resume optimizer state.

## Important

The initial 100-step run verifies GPU continuation; it will not make the model a capable chatbot. Its original evaluation found that generated text was incoherent. Better training data and a larger architecture are needed for meaningful improvements.
