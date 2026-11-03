# Finish adding ORBIT's binary model assets

The source code and notebook are already committed. Two files are still missing from the GitHub repository because the connected GitHub integration in this session only supports UTF-8 text-file writes, not binary uploads.

## Required files

From the provided `orbit-seed-gpu-ready.zip`, extract:

- `orbit_seed_model/model.safetensors` — the trained ORBIT Seed 0.1 weights (4,513,744 bytes).
- `orbit_seed_model/tokenizer.model` — its matching SentencePiece tokenizer model (270,765 bytes).

Upload both files to `orbit/seed_model/` on the `main` branch.

## GitHub web upload

1. Open https://github.com/lm11-gif/ok/tree/main/orbit/seed_model
2. Choose **Add file → Upload files**.
3. Select `model.safetensors` and `tokenizer.model` from the extracted `orbit_seed_model` folder.
4. Commit directly to `main`.

Keep the exact filenames. The model weights and tokenizer must be from the same checkpoint package. Do not use a different tokenizer or reinitialize the model.

## Optional tokenizer vocabulary

`tokenizer.vocab` is a human-readable vocabulary listing. It is useful for inspection but not required by `chat.py` or the GPU continuation script.

## After upload

The repository will contain the source, tokenizer assets, and trained weights needed to run ORBIT Seed and continue training it from the existing checkpoint. The first GPU continuation run creates a new optimizer state because the original training archive did not include one. Subsequent runs can resume that optimizer state if it is kept with the checkpoint.
