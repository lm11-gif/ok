# ORBIT Cognitive v1

**Status: implemented, CI-tested, and first T4 training run completed.** A revised association/retrieval training pass is pending. Three small specialist neural networks are randomly initialised and set up to train on a Kaggle GPU. This is an early, narrow prototype—not a general-purpose assistant or a literal brain.

## Architecture

- PerceptionNet classifies the subject concept and requested relation from a question.
- AssociationNet is an asymmetric bilinear graph model with separate subject and object embeddings. It learns multi-positive object targets for each subject/relation pair.
- MemoryNet embeds a question and ranks the fact records most relevant to it.
- A small Python controller combines explicit entity matching, perception, graph scores, and fact retrieval. Constrained answer templates keep unsupported free-form generation out of the initial evaluation.

All network weights are initialised randomly. No pretrained model weights, embeddings, or hidden states are loaded.

## Data

The initial dataset is a small curated seed knowledge graph in core.py. Training, validation, and test questions use separate template groups. Memory queries use different template groups for train/validation/test. AssociationNet is trained to reconstruct the complete small seed graph; its graph-reconstruction metric is not a held-out link-prediction score. MemoryNet uses multi-positive targets when one query maps to several valid fact records. This is reproducible and avoids a large dataset download, but the system only knows facts in the starter graph. Template-generated benchmark scores can overstate real-world language ability.

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
- Early-stop on held-out validation scores and retain the best checkpoint.
- Do not enable torch.compile by default: compilation overhead may outweigh gains for these small networks.

The Kaggle workflow records GPU model, VRAM, CUDA/PyTorch versions, timing, peak allocated GPU memory, and specialist validation/test scores. It uploads an Actions artifact before publishing a versioned release. It does not rely on pushing model files into the Git branch.

## Train through GitHub Actions

1. Open Actions → Train ORBIT Cognitive v1 on Kaggle GPU and choose Run workflow, or update START_TRAINING.txt in this directory.
2. The workflow uses existing KAGGLE_API_TOKEN and KAGGLE_USERNAME repository secrets. Never put credentials in files.
3. Download the Actions artifact named orbit-cognitive-v1-training-<run id>, or use the orbit-cognitive-v1 release after a successful run.

Only one Kaggle GPU job should run at a time; all ORBIT training workflows share a GitHub Actions concurrency lock. Ordinary code and documentation pushes do not start GPU training. The first submission was accepted by Kaggle but failed during script startup because adjacent module files were not importable. A self-contained bundler and canonical kernel-ID handling have been added; the corrected run has not yet been confirmed.

## Run the trained model

Extract orbit-cognitive-v1.zip, then run:

    python -m orbit.cognitive.chat --model path/to/extracted/model

The archive contains three checkpoints, config, vocabulary, entity/relation tables, fact store, and training report.

## Limitations and next steps

The first release can demonstrate specialist training, retrieval, and graph association over its small fact store. It cannot answer arbitrary questions, learn all facts by itself, or reliably reason about concepts absent from its graph. Expand and independently review the fact corpus, add robust unknown detection, measure on manually authored held-out questions, and only then add a learned controller or answer generator.

## First T4 run (historical baseline)

The first successful GPU run was GitHub Actions [41763096394](https://github.com/lm11-gif/ok/actions/runs/41763096394), on an NVIDIA Tesla T4 (14.56 GB reported VRAM), PyTorch 2.11.0+cu130 / CUDA 13.0. It used FP16 autocast and fused AdamW. The three stages took 7.84 seconds in total and reported 0.029 GB peak allocated GPU memory.

That first run is a baseline, not a successful general-learning result: PerceptionNet test subject accuracy was 100% and relation accuracy about 75.4%; AssociationNet's earlier held-out top-1 score was 0% and top-5 hit rate 15.4%; MemoryNet's earlier Recall@5 was 97.8%, but those initial memory labels treated each member of a multi-answer fact group separately and therefore undercounted valid answers. The follow-up code changes correct these objective/evaluation issues. Do not compare metrics across the old and revised objective without noting the changed target definitions.

The revised code makes AssociationNet asymmetric and trains it to reconstruct known graph pairs (a memorisation/cache task over the small seed graph, explicitly not held-out generalisation). MemoryNet now groups all valid facts for a subject/relation into one positive target set. CUDA timing boundaries now synchronise before recording stage time. A new GPU run is required to measure these changes; do not quote new speed or accuracy until that run finishes.
