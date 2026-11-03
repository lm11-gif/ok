# Finish adding ORBIT's binary model assets

The source code, training scripts, and Kaggle notebook are already committed. Two binary files remain:

- `orbit/seed_model/model.safetensors` — original trained ORBIT Seed 0.1 weights (4,513,744 bytes).
- `orbit/seed_model/tokenizer.model` — matching SentencePiece tokenizer (270,765 bytes).

## Easiest option: use the browser uploader

This repository now includes a standalone browser utility: [GitHub binary uploader](tools/github-binary-uploader.html).

You can also download the same utility from the ChatGPT conversation attachment. Open it in your browser, then select the two files extracted from `orbit-seed-gpu-ready.zip`.

The uploader sends the files directly from your browser to GitHub's Contents API. It does not use an intermediate server, external libraries, analytics, or browser storage.

### Temporary token required

The existing ChatGPT GitHub connection can read and commit source text but does not expose a local-file binary upload action. The browser utility therefore uses a short-lived fine-grained GitHub personal access token created by you:

1. Create a fine-grained token at https://github.com/settings/personal-access-tokens/new.
2. Set Repository access to **Only select repositories** and select `ok`.
3. Grant only **Contents: Read and write**.
4. Paste the token into the local uploader, choose the two files, and start upload.
5. After both commits succeed, verify the files appear in this folder and **revoke the token immediately**.

Never paste the token into ChatGPT or share it with anyone. The uploader does not save it and sends it only to `api.github.com`.

## Manual alternative

1. Open https://github.com/lm11-gif/ok/tree/main/orbit/seed_model
2. Choose **Add file → Upload files**.
3. Select the two files from the extracted `orbit_seed_model` folder.
4. Commit directly to `main`.

Keep the exact filenames. The model weights and tokenizer must come from the same archive. Do not use a different tokenizer or reinitialize the model.

## After upload

The repository will contain the source, matching tokenizer, and trained weights needed to run ORBIT Seed and continue training it from the existing checkpoint. The first GPU continuation run creates a fresh optimizer state because the original training archive did not include optimizer state. Subsequent runs can resume that optimizer state if it is kept with the checkpoint.
