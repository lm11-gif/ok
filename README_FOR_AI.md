# README_FOR_AI — ORBIT Project Handoff

**Purpose:** This file is the durable handoff/context document for any AI assistant or developer continuing work on this repository. Read it before modifying model code, datasets, training scripts, or GitHub Actions. It records the user's actual goal, the model lineage, current implementation, training automation, verified status, and known limitations.

**Last status snapshot:** 2026-11-04 (Australia/Brisbane).
**Repository:** https://github.com/lm11-gif/ok
**Main branch:** `main`

---

## 1. The user's goal and important preferences

The user is building **ORBIT**, an AI playground and an independently developed language-model project named **ORBIT Seed**.

The user's central requirement is a **genuinely scratch-trained AI**, not merely a small pretrained model with a new name or a fine-tuned checkpoint. For the scratch student:

- Initialize every student-model weight randomly.
- Train ORBIT's own tokenizer from the raw training text.
- Pretrain the student on text using next-token prediction.
- Then teach it conversational behaviour using assistant examples.
- Optionally use a stronger teacher to write training examples and preference answers. Teacher-written text is allowed; teacher weights must not initialize or be copied into the ORBIT student.
- Keep a clear record of the model's origin and actual training history.

A student can still honestly be described as trained from scratch when it learns from text written by another model. That is text/data distillation, not pretrained-weight fine-tuning. Be transparent that teacher-generated data influences what the student learns.

The user strongly values:
- Honest status reporting. Never say a model trained, a release exists, a test passed, or a workflow succeeded unless the corresponding run/output confirms it.
- Real repository edits, file changes, and runnable training workflows rather than plans or simulated actions.
- Avoiding wasted GPU quota. Do not start duplicate Kaggle jobs or run scratch and v0.2 jobs simultaneously unless the user explicitly chooses that.
- Clear, direct language. Avoid exaggerated slang, unnecessary emojis, and overconfident claims.

Never ask the user to paste Kaggle secrets. The repository already has the Actions secrets `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`.

---

## 2. Model lineage — do not mix these branches up

There are three distinct model tracks. They are not interchangeable.

| Track | Student initialization | What it is |
|---|---|---|
| ORBIT Seed v0.1 | Random | Original tiny scratch experiment: 871,552 parameters, 3 layers, hidden size 128, 4 heads, context 160, 2,000-piece SentencePiece tokenizer. |
| ORBIT Seed v0.2 | Pretrained SmolLM2-360M | Separate conversational fine-tune. This is **not** scratch-trained. It has an SFT stage and a teacher-supervised DPO stage. |
| ORBIT Seed v0.3 Scratch | Random | New independent scratch project. Own SentencePiece tokenizer, FineWeb-Edu pretraining, assistant-only conversation tuning, and a scratch-specific DPO stage. |

### v0.1 — original scratch proof of concept

Location: `orbit/seed_model/`

- Original student weights were randomly initialized.
- It was trained on a small, repetitive, synthetic corpus.
- The documented original run used 2,400 training steps and about 481,532 training tokens.
- It has 871,552 parameters.
- It generated fragmented or irrelevant responses and is not a capable general-purpose assistant.
- It is useful as proof that ORBIT already had a genuinely scratch-trained experiment, but v0.3 is intended to be the serious replacement.

### v0.2 — pretrained comparison branch

Location: `orbit/seed_model_v0.2/`

- Base student: `HuggingFaceTB/SmolLM2-360M`.
- SFT data: `HuggingFaceTB/smol-smoltalk`.
- The student is fine-tuned from pretrained weights. Never call this branch scratch-trained.
- Teacher: `HuggingFaceTB/SmolLM3-3B`.
- DPO workflow: `.github/workflows/train-orbit-v02-dpo.yml`.
- DPO trainer: `orbit/seed_model_v0.2/train_dpo.py`.
- It is useful as a conversational baseline/comparison, not as the user's final scratch model.

### v0.3 Scratch — primary scratch objective

Location: `orbit/seed_model_v0.3_scratch/`

- The student Transformer is newly initialized with random weights.
- It does not load SmolLM2, SmolLM3, or any other pretrained student checkpoint.
- ORBIT's tokenizer is trained from the FineWeb-Edu text corpus.
- FineWeb-Edu supplies base language pretraining text.
- Smol-SmolTalk supplies conversation examples.
- SmolLM3-3B writes some additional assistant training examples and later teacher-reviewed preference answers. It is used as a text generator/judge only; no SmolLM3 weights, hidden states, or embeddings are copied into ORBIT.
- Its config/report explicitly include `from_scratch: true` and `pretrained_student_weights: false`.

The distinction is important: v0.3 is a randomly initialized student trained on text, some of which is synthetic teacher output. It is not a distilled weight checkpoint.

---

## 3. v0.3 Scratch architecture and training recipe

Main trainer: `orbit/seed_model_v0.3_scratch/train_scratch.py`

Local inference: `orbit/seed_model_v0.3_scratch/chat.py`

Current initial configuration in the trainer:

| Setting | Value |
|---|---:|
| Architecture | Decoder-only causal Transformer |
| Parameters | Approximately 24.9 million (exact count written to config) |
| Layers | 8 |
| Hidden width | 448 |
| Attention heads | 7 |
| Context length | 512 tokens |
| SentencePiece BPE vocabulary | Up to 12,000 pieces |
| Dropout | 0.08 |
| Initial pretraining budget | Up to 100 million tokens per run |
| Pretraining learning rate | 3e-4 |
| Pretraining batch | 8, gradient accumulation 4 |
| Instruction examples | Up to 4,000 Smol-SmolTalk examples, plus teacher-written answers |
| Assistant tuning loss | Masked to assistant-answer tokens |

Approximate parameter count with a 12,000-token vocabulary is about 24.9M. The exact value is calculated by the script and written to `config.json`.

### Data

- Pretraining dataset: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed with a bounded character budget.
- Trainer limits raw collected text using `MAX_RAW_CHARS = 420_000_000` and truncates long documents to 16,000 characters.
- It creates a train/validation split before tokenization.
- SentencePiece is trained from the pretraining train split. Its model file must always stay paired with the checkpoint weights.
- Conversation dataset: `HuggingFaceTB/smol-smoltalk`.
- The trainer masks system and user tokens with label `-100` and calculates SFT loss on assistant replies. Preserve this behaviour.

### Compute and scale expectations

The first run is designed for a Kaggle Tesla T4, with up to 100M pretraining tokens. This is an initial small-model run, not enough compute to recreate SmolLM3 or another modern multi-billion-parameter assistant. Do not promise comparable capability.

The code includes continuation logic: later runs can download the last `orbit-seed-v0.3-scratch` release, restore ORBIT's own pretraining checkpoint and optimizer state, and continue the same student. Keep that logic intact unless deliberately redesigning checkpoint/resume.

The release archive is intended to include:
- `model.safetensors` — assistant-tuned scratch student weights.
- `pretraining_model.safetensors` — best raw-pretraining checkpoint used for continuation.
- `optimizer_state.pt` and `pretraining_log.json` — continuation state/history.
- `tokenizer.model` and optional `tokenizer.vocab`.
- `config.json`, `training_report.json`, and `NOTICE.md`.

Do not accidentally publish only the inference weights if doing so would break continuation.

---

## 4. Teacher supervision and DPO

### v0.3 scratch DPO

Trainer: `orbit/seed_model_v0.3_scratch/train_scratch_dpo.py`

Workflow: `.github/workflows/train-orbit-v03-scratch-dpo.yml`

This is separate from v0.2's DPO.

Process:
1. Load the published v0.3 scratch checkpoint and verify its scratch markers.
2. Ask the scratch ORBIT student to answer a curated prompt bank. These drafts are the rejected answers.
3. Load SmolLM3-3B as a teacher and ask it to review each ORBIT draft and produce an improved answer. These are the chosen answers.
4. Unload the teacher before preference optimization.
5. Load the v0.3 scratch-trained student as the policy and make a frozen copy of that same student as the reference policy.
6. Optimize the scratch student using the DPO objective. This implementation calculates the log-probability margins directly in PyTorch; it does not use TRL's DPOTrainer.
7. Save preference pairs, training/evaluation metrics, and side-by-side SFT-versus-DPO answer samples.

Current parameters: up to 96 prompt pairs, minimum 70 usable pairs, 2 epochs, beta 0.1, learning rate 1e-5.

Teacher-generated preferences can be wrong. DPO loss is not a factuality score. Review `preferences.jsonl`, held-out preference metrics, and generated answers before concluding that DPO improved the model.

### v0.2 DPO

Do not confuse the above with `orbit/seed_model_v0.2/train_dpo.py`. v0.2 DPO adjusts the pretrained SmolLM2-derived student. It does not affect the scratch lineage.

---

## 5. GitHub Actions / Kaggle automation

Repository Actions workflows:

| Workflow file | Purpose |
|---|---|
| `.github/workflows/train-orbit-v02.yml` | Pretrained v0.2 SFT on Kaggle |
| `.github/workflows/train-orbit-v02-dpo.yml` | v0.2 teacher DPO after successful v0.2 SFT |
| `.github/workflows/train-orbit-v03-scratch.yml` | Genuine scratch pretraining and assistant tuning |
| `.github/workflows/train-orbit-v03-scratch-dpo.yml` | Scratch-specific DPO after successful v0.3 scratch training |
| `.github/workflows/check-orbit-v03-scratch.yml` | Python syntax validation for v0.3 scripts |

Kaggle GitHub Actions secrets already configured:
- `KAGGLE_API_TOKEN`
- `KAGGLE_USERNAME`

Never print, retrieve, or request the values of these secrets.

### Workflow sequence currently configured

1. v0.2 SFT is running first.
2. When v0.2 SFT completes successfully, the v0.2 DPO workflow should run.
3. When the v0.2 DPO workflow itself completes through a `workflow_run` event, the v0.3 Scratch training workflow is configured to start. It is intentionally serialized to avoid simultaneous Kaggle GPU jobs.
4. When v0.3 Scratch training succeeds and publishes its release, v0.3 Scratch DPO should start.
5. v0.3 DPO publishes a separate scratch-DPO release.

The v0.3 scratch workflow also has `workflow_dispatch`, so a manual run from GitHub Actions can start it without waiting for the v0.2 chain. Avoid doing this while another Kaggle job is using the same free quota unless the user explicitly wants that.

### Kaggle notebook reference warning

A previous v0.2 run failed while polling a guessed notebook slug. Kaggle normalizes notebook titles, and the leaderboard's task slug is not necessarily the notebook's actual ID.

**Always resolve the canonical notebook `ref` from the authenticated account's own kernel list** (e.g. `kaggle kernels list --mine --search ... --format json`) and use that exact `owner/slug` for `kaggle kernels status`, `logs`, and `output`. Do not infer or reconstruct it from the title if Kaggle already returns the ref.

The observed v0.2 kernel reference was `dud12345/orbit-seed-v0-2-fine-tuning`. Its status was confirmed as `KernelWorkerStatus.RUNNING` during the most recent checks. Check the live account listing rather than assuming this remains current.

---

## 6. Verified status snapshot — 2026-11-04

This section is a historical snapshot, not a promise that the state is still current. Always query GitHub Actions before continuing.

### v0.2 SFT

- GitHub Actions run: [41744997703](https://github.com/lm11-gif/ok/actions/runs/41744997703)
- Last checked state: `in_progress`.
- The workflow passed script syntax, Kaggle credential validation, CLI installation, notebook preparation, and Kaggle submission.
- It was still in the `Wait for Kaggle to finish` step at the last status check.
- Kaggle notebook ref observed: `dud12345/orbit-seed-v0-2-fine-tuning`.
- Kaggle status observed: `KernelWorkerStatus.RUNNING`.
- No successful v0.2 release/checkpoint was confirmed at the snapshot time.

Earlier v0.2 attempt:
- Kaggle error: incompatible `torchao` version (installed 0.10.0 while PEFT expected >=0.16).
- Another attempt also used an incorrect guessed kernel slug.
- The active corrected trainer uninstalls the unused optional `torchao` package and resolves the notebook ref via Kaggle's own listing.
- Do not assume those fixes guarantee the current run succeeds. Inspect Kaggle logs/output and the actual training report.

### v0.2 DPO

- Push-triggered checks such as runs 41747479304, 41747564172, and 41747699008 passed their syntax-check jobs, but the actual `dpo` job was **skipped**. An overall workflow `success` does not mean DPO training ran.
- The real DPO job should only run on a `workflow_run` event after the v0.2 SFT run successfully completes.
- Verify the event and job steps before saying it has trained.

### v0.3 Scratch

- Main scratch trainer and chat runner passed Python syntax checks in GitHub Actions (e.g. run [41748906393](https://github.com/lm11-gif/ok/actions/runs/41748906393)).
- The scratch DPO trainer also passed Python syntax checks (e.g. run [41748906376](https://github.com/lm11-gif/ok/actions/runs/41748906376)).
- Push-triggered scratch-DPO workflow runs had the DPO training job **skipped**, as expected because no scratch release existed yet. One earlier push-triggered DPO attempt failed at the checkpoint-release verification step. These were not DPO GPU training runs.
- No successful v0.3 scratch Kaggle training run or v0.3 model release was confirmed at the snapshot time.
- Do not tell the user the scratch model has trained until Kaggle successfully completes, the archive is downloaded/validated, and the release/report exist.

### What "success" means

GitHub Actions workflow status alone can be misleading if one job is skipped. Inspect jobs and steps. A syntax-check-only run may be green while the actual GPU job is skipped. A Kaggle notebook can be running while the GitHub workflow is still polling it. A notebook submission is not evidence that training completed.

---

## 7. Important files

### Root
- `README.md` — short human-facing project overview.
- `README_FOR_AI.md` — this durable AI handoff document.

### v0.1
- `orbit/seed_model/` — original small scratch model.

### v0.2
- `orbit/seed_model_v0.2/train_smollm2.py` — pretrained SmolLM2 SFT.
- `orbit/seed_model_v0.2/train_dpo.py` — v0.2 teacher DPO.
- `orbit/seed_model_v0.2/chat.py` — v0.2 inference.
- `orbit/seed_model_v0.2/README.md` — v0.2 explanation.

### v0.3 Scratch
- `orbit/seed_model_v0.3_scratch/train_scratch.py` — random-init pretraining and assistant SFT.
- `orbit/seed_model_v0.3_scratch/train_scratch_dpo.py` — teacher preference generation and scratch-specific DPO.
- `orbit/seed_model_v0.3_scratch/chat.py` — local inference model class and chat runner.
- `orbit/seed_model_v0.3_scratch/README.md` — human-facing scratch branch guide.
- `orbit/seed_model_v0.3_scratch/requirements.txt` — local inference dependencies.
- `orbit/seed_model_v0.3_scratch/START_TRAINING.txt` — push trigger for scratch training.

### Workflows
- `.github/workflows/train-orbit-v02.yml`
- `.github/workflows/train-orbit-v02-dpo.yml`
- `.github/workflows/train-orbit-v03-scratch.yml`
- `.github/workflows/train-orbit-v03-scratch-dpo.yml`
- `.github/workflows/check-orbit-v03-scratch.yml`

---

## 8. What the next assistant should do

When resuming work:

1. Read this file first.
2. Query the live GitHub Actions runs for `lm11-gif/ok`; specifically inspect v0.2 SFT run 41744997703 and any later v0.2 DPO/v0.3 runs.
3. If SFT is still running, inspect its GitHub job steps and Kaggle notebook status/logs. Do not start another competing GPU job.
4. If SFT succeeds, verify its release asset and training report. Then confirm that the v0.2 DPO job actually ran (not merely that a syntax-check job succeeded).
5. Confirm v0.3 Scratch training is queued/started only after the intended dependency. Manual dispatch is available, but check GPU usage first.
6. On v0.3 success, inspect:
   - `from_scratch: true`
   - `pretrained_student_weights: false`
   - model parameter count and tokenizer size
   - actual pretraining token count and validation-loss trend
   - assistant-only SFT counts/loss
   - generated sample answers
   - presence and validity of `model.safetensors`, `pretraining_model.safetensors`, `optimizer_state.pt`, tokenizer, config, and report.
7. Only after the v0.3 scratch release exists should its scratch DPO workflow run. Inspect whether its DPO job actually ran, the preference-pair count, training/held-out metrics, and SFT-vs-DPO sample answers.
8. If a failure occurs, report the exact failing step and the actual traceback. Fix the cause and verify syntax before resubmitting. Avoid wasting Kaggle GPU quota on repeated unverified submissions.

Do not silently replace the v0.3 random-init student with SmolLM2/SmolLM3 weights, or relabel v0.2 as scratch-trained. Keep model lineage and checkpoint/tokenizer pairing explicit in every release.

---

## 9. Licensing and provenance notes

- The FineWeb-Edu dataset is documented as ODC-By 1.0; keep attribution/provenance notices with releases.
- Smol-SmolTalk has its own dataset/model card terms; check its current card before redistributing substantial data.
- SmolLM3-3B is used as a teacher and is documented under Apache 2.0; retain attribution and notices.
- A dataset's license and the teacher model's license are distinct. Do not assume they automatically cover each other or every generated output.
- Keep the release notices and data-source names/links in the training report.

**Final principle:** ORBIT's goal is an independently trained student. Improve it through more genuine data, compute, evaluation, and carefully controlled teacher-generated text — not by quietly substituting a pretrained model.
