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

### Workflow sequence and latest real execution

The workflows were originally configured as a serialized v0.2 SFT → v0.2 DPO → v0.3 Scratch → v0.3 Scratch DPO chain. That complete chain did **not** execute: the user manually cancelled v0.2 before a v0.2 checkpoint or DPO run existed. The v0.3 Scratch run was then submitted directly. Its Kaggle notebook completed, but its GitHub report-commit step failed due to a non-fast-forward push, so the v0.3 release was skipped. The v0.3 Scratch DPO stage has not been verified as run. See the latest handoff in Section 6A below.

Do not infer that a configured `workflow_run` chain completed simply because its workflow files exist. Verify every job and step.

The v0.3 scratch workflow also has `workflow_dispatch`, so a manual run from GitHub Actions can start it without waiting for the v0.2 chain. Avoid doing this while another Kaggle job is using the same free quota unless the user explicitly wants that.

### Kaggle notebook reference warning

A previous v0.2 run failed while polling a guessed notebook slug. Kaggle normalizes notebook titles, and the leaderboard's task slug is not necessarily the notebook's actual ID.

**Always resolve the canonical notebook `ref` from the authenticated account's own kernel list** (e.g. `kaggle kernels list --mine --search ... --format json`) and use that exact `owner/slug` for `kaggle kernels status`, `logs`, and `output`. Do not infer or reconstruct it from the title if Kaggle already returns the ref.

The observed v0.2 kernel reference was `dud12345/orbit-seed-v0-2-fine-tuning`. Its status was confirmed as `KernelWorkerStatus.RUNNING` during the most recent checks. Check the live account listing rather than assuming this remains current.

---

## 6A. Latest training handoff — 2026-11-04 (after Kaggle completion)

This section supersedes the 01:10 UTC snapshot and all older running-status notes below. It reflects the latest verified state from GitHub Actions job 126314079773 in workflow run 41750482331. Check current status again before starting any new run.

### User-requested stop-v0.2 / start-v0.3 transition — verified successful

- User manually stopped the v0.2 Kaggle session from Kaggle's Active Events
  controls.
- Kaggle status verification workflow
  [41752074440](https://github.com/lm11-gif/ok/actions/runs/41752074440)
  completed successfully at 2026-11-04 01:09:41 UTC.
- The exact status output at 01:09:38 UTC:
  - `dud12345/orbit-seed-v0-2-fine-tuning`:
    `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`
  - `dud12345/orbit-seed-v0-3-scratch-training`:
    `KernelWorkerStatus.RUNNING`
- v0.2 GitHub workflow
  [41744997703](https://github.com/lm11-gif/ok/actions/runs/41744997703)
  is also completed/cancelled. It did not download or publish a v0.2
  checkpoint. Do not infer a v0.2 model was produced.
- Earlier API cancellation attempts failed due missing
  `kernelSessions.cancel` permission. These attempts are no longer needed;
  the user's manual cancellation succeeded and Kaggle confirmed it.
- There should now be only the v0.3 Kaggle training job consuming GPU.

### v0.3 Scratch GPU job — completed, but release/reporting incomplete

- Scratch workflow run
  [41750482331](https://github.com/lm11-gif/ok/actions/runs/41750482331)
  reached a completed GitHub Actions job with overall conclusion **failure**.
- Kaggle notebook: `dud12345/orbit-seed-v0-3-scratch-training`.
- The job log reports `KernelWorkerStatus.COMPLETE` at 2026-11-04 01:44:55 UTC.
- The workflow downloaded `pretraining_model.safetensors`, `tokenizer.model`, `training_report.json`, `pretraining_log.json`, train/validation token bins, and the tokenizer training corpus.
- The next step copied `training_report.json` and `config.json` into the runner checkout and attempted to commit/push them. GitHub rejected the push with `fetch first` / `non-fast-forward` because `main` had newer commits. The report commit was not pushed.
- The `Publish the scratch model release` step was skipped. A subsequent artifact listing returned no GitHub Actions artifacts. The report/config files were not present on `main` when fetched.
- Therefore Kaggle execution and download of its outputs are confirmed, but a durable report in GitHub and a published release are **not** confirmed. Do not claim the release exists or quote training metrics until the Kaggle output is recovered and inspected.
- The student model was configured for random initialization; SmolLM3 was intended as a text-data teacher only. Verify the actual downloaded report/config before making claims about the exact trained result.
- No v0.3 DPO training run is confirmed because its prerequisite release was not published. Do not trigger DPO without recovering and validating the checkpoint first.
- The user's observed chat samples were poor/incoherent: “By choosing the snow that is rain.”; “I've got a Discovery Worksheet?”; a longer desk response containing “introitulation”; and “To create a comprehensive sentence...” These samples have not been tied to a verified checkpoint hash or fixed decoding settings. Treat them as qualitative observations, not a controlled evaluation.

Do not submit another Kaggle training job merely to fix the GitHub push. First determine whether the completed Kaggle output can be downloaded again, then recover the report/checkpoint without repeating training.

### One-off helper workflow

`.github/workflows/cancel-v02-and-start-scratch.yml` is now a read-only
diagnostic workflow named "Check ORBIT Kaggle GPU sessions". It can query
current Kaggle statuses and v0.2 logs. Keep the two Kaggle secrets in its
environment. It does not cancel sessions.

---

## 6B. Historical verified status snapshot — 2026-11-04 (superseded)

This section records what was known before the v0.3 Kaggle run completed. It is retained for audit/history only; Section 6A above is the latest verified status. Always query GitHub Actions before continuing.

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

### ORBIT Cognitive proposal
- `orbit/cognitive/COGNITIVE_ARCHITECTURE.md` — proposed multi-network, neuro-symbolic cognitive architecture. Design document only; no implementation or training is claimed.

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

1. Read this file and `orbit/cognitive/COGNITIVE_ARCHITECTURE.md` first.
2. Query live GitHub Actions and Kaggle output availability. Treat v0.2 as manually cancelled and v0.3 as Kaggle-completed but not released until current evidence says otherwise.
3. Try to recover the existing v0.3 Kaggle output/report and checkpoint before considering another GPU run. The last GitHub job downloaded these files but failed only when pushing the report commit; do not waste GPU quota by blindly retraining.
4. Inspect the recovered `training_report.json`, `config.json`, `pretraining_log.json`, and checkpoint set. Verify actual tokens/steps, validation trend, SFT stage counts/loss, exact parameter count, tokenizer size, and file integrity. Do not infer these from the Kaggle `COMPLETE` status alone.
5. Do not run v0.3 DPO until the scratch checkpoint has been recovered, validated, and published through a durable release or other controlled artifact path.
6. For Cognitive v1, inspect the active CI run and fix failures first. The typed fact store, dataset generator, specialist models, trainer, inference code, tests, and Kaggle workflow now exist in orbit/cognitive/. Only after CI passes, trigger the manual Kaggle GPU workflow. Review GPU hardware, held-out scores, per-network timing, and the archived checkpoints. Do not state that training or speedups are verified until a successful Kaggle report exists.
7. Build a held-out question set covering basic definitions, paraphrases, related concepts, multi-step questions, unknown facts, and contradictions. Track factual accuracy, evidence coverage, unsupported claims, abstention, latency, and memory use.
8. Keep v0.3 as a separate baseline; do not delete it or silently combine its weights with the cognitive modules. Keep all design and implementation updates in GitHub.
9. If a failure occurs, report the exact failing step and actual traceback. Fix it and verify before resubmitting. Avoid wasting Kaggle GPU quota on repeated unverified submissions.

Do not silently replace the v0.3 random-init student with SmolLM2/SmolLM3 weights, or relabel v0.2 as scratch-trained. Keep model lineage and checkpoint/tokenizer pairing explicit in every release.

---

## 9. Licensing and provenance notes

- The FineWeb-Edu dataset is documented as ODC-By 1.0; keep attribution/provenance notices with releases.
- Smol-SmolTalk has its own dataset/model card terms; check its current card before redistributing substantial data.
- SmolLM3-3B is used as a teacher and is documented under Apache 2.0; retain attribution and notices.
- A dataset's license and the teacher model's license are distinct. Do not assume they automatically cover each other or every generated output.
- Keep the release notices and data-source names/links in the training report.

**Final principle:** ORBIT's goal is an independently trained student. Improve it through more genuine data, compute, evaluation, and carefully controlled teacher-generated text — not by quietly substituting a pretrained model.


## 10. ORBIT Cognitive v1 — implementation and GPU training

The initial implementation, two Kaggle GPU runs, and a model release are complete. The latest chat-controller refinements and tests are in CI. The current released weights are trained with the asymmetric AssociationNet and multi-positive MemoryNet objectives.

### Repository files

- orbit/cognitive/README.md — end-user instructions, run metrics, limits.
- orbit/cognitive/COGNITIVE_ARCHITECTURE.md — design and implementation history.
- orbit/cognitive/core.py — 67-fact seed graph, shared vocabulary/entity/relation IDs, multi-fact grouping, deterministic question templates.
- orbit/cognitive/models.py — PerceptionNet, asymmetric AssociationNet, MemoryNet.
- orbit/cognitive/train.py — GPU training, timing/throughput reporting, validation, checkpoint saving, archive generation.
- orbit/cognitive/chat.py — inference/controller code; singular/plural entity matching, symbolic relation hints, association+memory ranking, template response composition.
- orbit/cognitive/bundle_kaggle.py — self-contained Kaggle script builder. Kaggle script mode did not automatically expose adjacent .py files.
- orbit/cognitive/tests/test_cognitive.py — unit tests for seed data, multi-positive targets, module shapes, plural entities, relation cues, and natural definition articles.
- .github/workflows/check-orbit-cognitive.yml — CPU-only CI: compilation, unit tests, bundled-script smoke training, checkpoint load/inference, archive validation.
- .github/workflows/train-orbit-cognitive-kaggle.yml — manual Kaggle T4 training, archive validation, Actions artifact upload, release publishing.
- .github/workflows/recover-orbit-cognitive-kaggle.yml — monitor/retrieve an already-submitted Kaggle kernel; it does not submit a new training job.

### Architecture and objectives

All three student networks are randomly initialised; no pretrained weights, embeddings, hidden states, or teacher model are loaded.

- PerceptionNet: predicts subject entity and relation from question tokens.
- AssociationNet: asymmetric bilinear scoring with independent subject/object tables. It trains on multi-positive objects for all subject/relation pairs in the known graph.
- MemoryNet: ranks fact records for a query. Every valid fact ID for a subject/relation is a positive target, so a query can have several correct answers.
- Controller: exact word-boundary concept matching with simple plural aliases, explicit relation wording hints plus neural fallback, association-assisted ranking of relevant stored facts, and constrained answer templates.

The seed graph contains 67 facts, 67 entities, and 18 relations. There are 59 subject/relation groups. The starter dataset and question templates are intentionally small. Perception and Memory test queries are held-out template variants over facts already in the graph. Association uses all 59 subject/relation groups for training and reconstruction evaluation; its 100% score is in-sample graph memorisation, not held-out link-prediction/generalisation. Never present these synthetic-template metrics as broad intelligence or factual reasoning benchmarks.

### Latest successful GPU run and release

GitHub Actions run [41764168830](https://github.com/lm11-gif/ok/actions/runs/41764168830) succeeded and updated the [orbit-cognitive-v1 GitHub release](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v1). The workflow downloaded and verified the model archive, uploaded a durable Actions artifact, and uploaded the ZIP, report, and config to the release.

GPU/runtime:
- NVIDIA Tesla T4, 14.56 GB reported memory, compute capability 7.5.
- PyTorch 2.11.0+cu130, CUDA 13.0.
- FP16 autocast enabled; fused AdamW active.
- TF32 disabled because T4 is compute capability 7.5.
- Batch size 256; tokenised tensors kept on GPU; retrieval is vectorised; models train sequentially in one process.
- Total trainer-reported time: 6.776 s.
- Peak allocated GPU memory: 0.029 GB.

Per-specialist report:
- PerceptionNet: 93,877 parameters; 35 epochs; 1.152 s; 10,755 training examples/s; test subject accuracy 100%, relation accuracy 72.0%.
- AssociationNet: 14,659 parameters; 37 epochs; 0.189 s; 11,543 training subject/relation groups/s; known-graph reconstruction top-1 and top-5 both 100% (in-sample).
- MemoryNet: 85,632 parameters; 32 epochs; 0.438 s; 25,848 training queries/s; multi-positive test Recall@1 71.2%, Recall@5 89.8%.
- Total: 194,168 parameters.

The first GPU baseline (run 41763096394) reported 7.84 s but used earlier objectives and did not synchronise CUDA at timing boundaries. The revised run reports 6.776 s, but this is not a controlled speedup comparison because model objectives/architecture and timing method changed. Do not claim a measured percentage gain.

### Inference spot-check and recent controller changes

The released archive was downloaded and all three state dictionaries were loaded into their matching architectures in a local sanity harness. It returned sensible seed-graph answers for “What is rain?”, “What is a car?”, “What does a tree need?” (both sunlight and water), “What does RAM store?”, “Where does a river flow?”, “What is photosynthesis?”, “What does Python do?”, and “What does fire need?”. It abstained for an unknown concept (“dragon”). This was a narrow spot-check, not a broad benchmark.

After that spot-check, chat.py was updated to:
- recognise basic plurals such as cats, trees, countries, buses;
- use explicit query wording (“made of”, “need”, “where”, “flow”, etc.) as relation-routing hints, retaining PerceptionNet as fallback;
- compose more natural definitions (e.g. “rain is precipitation”, “computer is an electronic device”, “desk is furniture”);
- retry MemoryNet retrieval with a matching training-style canonical query when the relation hint is explicit but the original wording receives a low similarity score.

The canonical fallback was checked against the released checkpoints in a local harness: “How are cats classified?” retrieved “cat is a mammal” at 0.979 cosine similarity, while an unknown “dragon” question remained rejected. This is a narrow spot-check, not an independent benchmark. Unit tests cover plural forms, relation cues, canonical template construction, and definition grammar. CI run [41764965347](https://github.com/lm11-gif/ok/actions/runs/41764965347) passed the plural/relation/grammar changes. The additional canonical-query fallback and its tests are in run [41765177984](https://github.com/lm11-gif/ok/actions/runs/41765177984), still running during dependency installation, and run [41765192200](https://github.com/lm11-gif/ok/actions/runs/41765192200), queued for the latest test commit. Do not claim the newest canonical-fallback code passed until one of those latest runs is green.

These chat.py-only changes do not require retraining; the checkpoint architecture is unchanged. The current release contains the corrected training objectives and latest trained weights. If a future model class or checkpoint tensor shape changes, retrain and publish a matching archive.

### Next meaningful work

1. Confirm the latest CI run passes the new plural, relation-cue, and grammar regression tests.
2. Expand the fact graph substantially, with independent provenance and review. The current 67 facts are only a bootstrap.
3. Build manually authored held-out questions unrelated to training templates. Cover definitions, paraphrases, relation confusion, multi-step rule application, unknowns, contradictions, and ambiguous inputs.
4. Add an explicit rule/constraint engine and provenance/conflict tracking so the system can combine facts, explain chains, and abstain on missing or inconsistent evidence.
5. Measure exact factual accuracy, evidence coverage, unsupported-answer rate, unknown abstention, latency, and memory use.
6. Only add a ControlNet or further specialist networks when a simple rule-based baseline shows a concrete need.

Keep ORBIT Seed v0.3 as a separate scratch-language-model experiment. Its Kaggle execution completed earlier, but its report commit was rejected and no v0.3 release was confirmed. Do not mix its weights or status into Cognitive v1.


## 12. ORBIT Cognitive v2 — neural language generator (2026-11-04)

The user tested Cognitive v1 and correctly observed that it returned predefined answer templates, not generated language. The user explicitly requested a neural language generator. Keep v1 release/checkpoints intact and develop the separate v2 path.

Initial v2 source commit: `aca78cd5a8b2c9607ec192a7894752c296851122`. Chat integration commit: `fbc5938d42a8ef9f6529f29d99f5e2af473867d0`. A following fix commit addresses the Kaggle bundler syntax error reported by CI.

- `orbit/cognitive/models.py`: `NeuralLanguageGenerator`, a randomly initialised causal Transformer that predicts UTF-8 bytes (IDs 0..255 plus EOS=256) and samples output autoregressively.
- `orbit/cognitive/train.py`: GPU-first next-byte cross-entropy stage. It builds training data from the locally authored `orbit/seed_model/make_corpus.py` multi-turn dialogue corpus plus fact-grounded examples. Default generator settings: width 128, 4 decoder layers, 4 heads, 384-byte context, 12 epochs, batch size 32. Uses AMP/GradScaler, fused AdamW when supported, gradient clipping, validation loss and best checkpoint save.
- `orbit/cognitive/bundle_kaggle.py`: embeds a compressed generated corpus in the self-contained offline Kaggle entry point. Use base64 (not base85) because base85 output can contain apostrophes that corrupt a Python literal. The inserted import statement must use a real newline escape, not a literal backslash followed by n.
- `orbit/cognitive/chat.py`: v2 chat sends user text, short history, and retrieved memory to the neural generator. No template composer determines the final answer.
- v1 archive has no generator checkpoint; v2 chat requires `generator.pt`. Keep releases and Kaggle kernel names separate.
- The corpus is small synthetic data; expect weak initial language quality. Do not promise coherent answers. Verify sample outputs after training.
- Latest failed earlier CI is run 41768956965: bundle builder generated invalid Python because base85 compressed data contained a quote, and the injected import string contained an escaped literal backslash-n. Fix both before proceeding.
- Run CI before the v2 Kaggle GPU job. Only trigger via `orbit/cognitive/START_TRAINING.txt` after CI success. Do not claim v2 trained until Kaggle output archive is validated and release exists.
