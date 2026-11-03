# Binary model files — upload required

The GitHub connector available in this session can commit source text and notebook files, but does not expose a raw binary upload operation. These files therefore still need to be added through GitHub's web upload interface.

From the original `orbit-seed-gpu-ready.zip`, extract the `orbit_seed_model` folder and upload these files into this directory:

- `model.safetensors` (4,513,744 bytes) — the existing trained checkpoint.
- `tokenizer.model` (270,765 bytes) — the matching SentencePiece tokenizer model.

Do not substitute a different tokenizer or initialize new weights. The model and tokenizer must remain paired.

After uploading, this README can be removed or retained as provenance documentation.
