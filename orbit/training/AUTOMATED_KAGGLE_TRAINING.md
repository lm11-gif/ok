# ORBIT Seed GPU training automation

The workflow `.github/workflows/train-orbit-kaggle.yml` submits ORBIT Seed to a
private Kaggle GPU kernel, waits for it to finish, downloads its checkpoint,
and commits the trained weights and optimizer state back to `main`.

## One-time account authorization

The workflow runs under your Kaggle account, so GitHub Actions needs your
Kaggle API credentials. Do not paste them into chat or commit them to the repo.

1. Open https://www.kaggle.com/settings/api and generate an API token.
2. Open https://github.com/lm11-gif/ok/settings/secrets/actions.
3. Add two **repository secrets**:
   - `KAGGLE_API_TOKEN`: the API token you just generated
   - `KAGGLE_USERNAME`: your Kaggle username (the account slug in your Kaggle profile URL)

The token stays in GitHub Actions secrets and is passed only to the Kaggle CLI
during the workflow. It is not written to the repository or notebook source.

## Start the run

After both secrets exist, the repository's training workflow can be triggered
by updating `orbit/training/START_TRAINING.txt`. It will request 500 extra
optimizer updates on a Kaggle GPU by default. You can also use **Actions →
Train ORBIT Seed on Kaggle GPU → Run workflow** and select a step count.

The workflow:
- submits a private Kaggle kernel with GPU and internet enabled,
- verifies that CUDA is available before training,
- clones the current public repository and starts from its committed
  `model.safetensors`,
- recreates the deterministic synthetic train/validation split,
- continues the existing checkpoint rather than initializing a new model,
- downloads the trained checkpoint and commits
  `model.safetensors`, `optimizer_state.pt`, `training_log.json`, and
  `training_report.json` to `main`.

Kaggle GPU allocation is subject to account quota and availability. Kaggle
documents free GPU notebooks, but access may queue during busy periods:
https://www.kaggle.com/docs/notebooks

## Results and limits

A completed workflow should have a `training_report.json` describing the GPU,
step range, validation loss, and checkpoint hashes. The run is a conservative
continuation experiment on the current small synthetic corpus. It cannot by
itself turn this 871k-parameter prototype into a capable general-purpose
chatbot; the existing evaluation shows that dataset quality and model scale
are the larger limitations.
