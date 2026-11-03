# ORBIT training

The notebook in this folder continues the existing ORBIT Seed 0.1 checkpoint on a Kaggle GPU.

1. Upload the complete GPU-ready model package as a private Kaggle Dataset.
2. Attach it to a Kaggle Notebook.
3. Enable a GPU accelerator.
4. Import and run [ORBIT_Kaggle_GPU_Training.ipynb](ORBIT_Kaggle_GPU_Training.ipynb).

The first run is a 100-step smoke test. Download the resulting checkpoint ZIP from the notebook output and retain it for the next training session.

The notebook requires the binary `model.safetensors` weights and `tokenizer.model`; it cannot train if either is missing.
