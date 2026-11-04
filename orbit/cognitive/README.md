# ORBIT Cognitive v1

**Status: implemented, GPU-trained, and released.** The revised three-network checkpoint was trained on a Kaggle Tesla T4 and published as orbit-cognitive-v1. The inference controller has since received plural matching, relation-cue routing, and grammar fixes; its newest CI run is in progress. This is an early, narrow prototype—not a general-purpose assistant or a literal brain.

## Architecture

- PerceptionNet classifies the subject concept and requested relation from a question.
- AssociationNet is an asymmetric bilinear graph model with separate subject and object embeddings. It learns multi-positive object targets for each subject/relation pair.
- MemoryNet embeds a question and ranks the fact records most relevant to it.
- A small Python controller combines explicit entity matching, PerceptionNet, AssociationNet scores, and MemoryNet retrieval. It uses relation cues as symbolic routing hints and constrained answer templates; it does not generate unrestricted prose.

All network weights are initialised randomly. No pretrained model weights, embeddings, or hidden states are loaded.

## Data

The initial dataset is a small curated seed knowledge graph in core.py. The seed graph contains 67 manually curated facts, 67 entities, and 18 relation types. Training, validation, and test questions use separate template groups over the same underlying concepts/facts. MemoryNet groups every valid fact for a subject/relation into a multi-positive target set. AssociationNet is trained to reconstruct all 59 subject/relation groups in the known graph; its perfect score is an in-sample reconstruction result, not held-out link prediction. This is reproducible and avoids a large dataset download, but the system only knows facts in the starter graph. Template-generated benchmark scores can overstate real-world language ability.

## GPU training

Kaggle's GPU image supplies PyTorch. The workflow bundles all three source modules and the trainer into a single self-contained script because Kaggle script kernels execute the designated entry-point file rather than automatically exposing adjacent Python modules. Normal training refuses to run on CPU.

    python -m pip install -r orbit/cognitive/requirements.txt
    python -m orbit.cognitive.train --device cuda --output artifacts/orbit-cognitive-v1

CPU smoke test for CI only:

    python -m orbit.cognitive.train --device cpu --smoke --output /tmp/orbit-cognitive-smoke

### Speed techniques

- Tokenise the small corpus once and keep tensors resident on the GPU to avoid repeated Python/DataLoader/host-to-device overhead.
- Use a batch size of 256 to reduce per-step kernel-launch and optimiser overhead for these compact models.
- Synchronise CUDA at timing boundaries so per-network durations and throughput are measured after GPU work actually finishes.
- Use a single self-contained Kaggle script to avoid multi-file packaging/import failures and extra setup work.
- CI uses CPU-only PyTorch wheels because its tests are CPU-only; Kaggle uses its preinstalled CUDA PyTorch build.
- Use CUDA FP16 autocast and GradScaler when enabled.
- Use fused AdamW when the installed PyTorch/CUDA build supports it; automatically fall back otherwise.
- Enable TF32 matmul on Ampere-or-newer GPUs only.
- Vectorise MemoryNet retrieval over the complete fact table for each batch.
- Run compact specialist models sequentially in one GPU session to avoid competing GPU processes and model reload overhead.
- Early-stop on each specialist's selected validation/reconstruction metric and retain the best checkpoint. AssociationNet's early-stop score is in-sample graph reconstruction, not a held-out metric.
- Do not enable torch.compile by default: compilation overhead may outweigh gains for these small networks.

The Kaggle workflow records GPU model, VRAM, CUDA/PyTorch versions, timing, peak allocated GPU memory, and specialist validation/test scores. It uploads an Actions artifact before publishing a versioned release. It does not rely on pushing model files into the Git branch.

## Train through GitHub Actions

1. Open Actions → Train ORBIT Cognitive v1 on Kaggle GPU and choose Run workflow, or update START_TRAINING.txt in this directory.
2. The workflow uses existing KAGGLE_API_TOKEN and KAGGLE_USERNAME repository secrets. Never put credentials in files.
3. Download the Actions artifact named orbit-cognitive-v1-training-<run id>, or use the orbit-cognitive-v1 release after a successful run.

Only one Kaggle GPU job should run at a time; all ORBIT training workflows share a GitHub Actions concurrency lock. Ordinary code and documentation pushes do not start GPU training. The first submission failed during script startup because Kaggle did not import adjacent module files; the self-contained bundler fixed that. The revised run completed successfully in GitHub Actions run [41764168830](https://github.com/lm11-gif/ok/actions/runs/41764168830).

## Run the trained model

Download the orbit-cognitive-v1.zip asset from the [GitHub release](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v1) and extract it. The ZIP contains the checkpoints, config, vocabulary, entity/relation tables, fact store, and training report at its root. From a checkout of this repository, install a suitable PyTorch build and run:

    python -m pip install -r orbit/cognitive/requirements.txt
    python -m orbit.cognitive.chat --model path/to/extracted

The model weights/data are in the release ZIP; chat.py, models.py, and core.py are in the repository source.

## Limitations and next steps

This release is a small neuro-symbolic prototype, not a general-purpose assistant or a literal brain. The networks are randomly initialised, but all knowledge comes from the manually curated 67-fact seed graph. The language coverage is limited to template-style phrasings, the controller handles only the supported relations, and the output composer is template-based rather than a free-form language model. Perception's test relation accuracy is about 72%; unsupported or differently phrased questions can still route incorrectly.

The next meaningful improvements are a much larger reviewed fact graph, independently written held-out questions (including multi-step and contradiction cases), better open-set and relation parsing, explicit provenance/conflict handling, and a rule-based reasoning layer. Do not add more neural specialists until the current three show measurable benefit on those tests.

## Revised T4 training run (current release)

GitHub Actions run [41764168830](https://github.com/lm11-gif/ok/actions/runs/41764168830) completed successfully. It trained the three randomly initialised networks on an NVIDIA Tesla T4, then validated and uploaded the checkpoint archive, training report, and config to the [orbit-cognitive-v1 release](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v1).

| Specialist | Parameters | Epochs run | Measured stage time | Test result |
|---|---:|---:|---:|---|
| PerceptionNet | 93,877 | 35 | 1.152 s | Subject accuracy 100%; relation accuracy 72.0% |
| AssociationNet | 14,659 | 37 | 0.189 s | Known-graph reconstruction top-1/top-5: 100% |
| MemoryNet | 85,632 | 32 | 0.438 s | Multi-positive Recall@1 71.2%; Recall@5 89.8% |
| **Total** | **194,168** | — | **6.776 s reported total** | — |

AssociationNet's result is intentionally labelled *known-graph reconstruction*: its train/validation/test entries are the same 59 subject/relation groups, so 100% is evidence that it memorised this tiny graph, not evidence of generalisation. Perception and Memory use held-out question templates, but the underlying facts are already known to training; these scores measure limited paraphrase/template transfer, not independent factual reasoning.

GPU/runtime details: Tesla T4, 14.56 GB reported device memory, compute capability 7.5, PyTorch 2.11.0+cu130, CUDA 13.0, FP16 autocast enabled, fused AdamW active, TF32 disabled (appropriate for this GPU), and 0.029 GB peak allocated GPU memory. The trainer synchronises CUDA before timing boundaries and reports approx. 10,755 perception examples/s, 11,543 association groups/s, and 25,848 memory queries/s.

The first-run baseline (7.84 s) used earlier model/objective code and did not synchronise CUDA at timing boundaries. The revised run reports 6.776 s, but because the objectives, architecture, and timing method changed, this is **not a controlled speed comparison** and should not be described as a measured speedup. The applied speed techniques are FP16, fused AdamW, GPU-resident token tensors, vectorised retrieval, one process for sequential stages, batch size 256, and avoiding torch.compile overhead.

### Inference sanity check

I downloaded the released archive and loaded all three trained state dictionaries with their matching architecture. A local inference harness using the current model architecture and the controller flow produced sensible seed-graph answers for questions including “What is rain?”, “What is a car?”, “What does a tree need?”, “What does RAM store?”, “Where does a river flow?”, “What is photosynthesis?”, and “What does fire need?”. It correctly abstained for an unknown concept (“dragon”). The controller has since been improved to handle plural entity forms (e.g. “cats”), explicit relation cues (e.g. “where ...?”), and natural articles (“an electronic device”); the matching unit tests are running in CI.

## First T4 run (historical baseline)

The first successful GPU run was GitHub Actions [41763096394](https://github.com/lm11-gif/ok/actions/runs/41763096394), on an NVIDIA Tesla T4 (14.56 GB reported VRAM), PyTorch 2.11.0+cu130 / CUDA 13.0. It used FP16 autocast and fused AdamW. The three stages took 7.84 seconds in total and reported 0.029 GB peak allocated GPU memory.

That first run is a baseline, not a successful general-learning result: PerceptionNet test subject accuracy was 100% and relation accuracy about 75.4%; AssociationNet's earlier held-out top-1 score was 0% and top-5 hit rate 15.4%; MemoryNet's earlier Recall@5 was 97.8%, but those initial memory labels treated each member of a multi-answer fact group separately and therefore undercounted valid answers. The follow-up code changes correct these objective/evaluation issues. Do not compare metrics across the old and revised objective without noting the changed target definitions.

The revised code makes AssociationNet asymmetric and trains it to reconstruct known graph pairs (a memorisation/cache task over the small seed graph, explicitly not held-out generalisation). MemoryNet now groups all valid facts for a subject/relation into one positive target set. CUDA timing boundaries now synchronise before recording stage time. A new GPU run is required to measure these changes; do not quote new speed or accuracy until that run finishes.
