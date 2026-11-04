# ORBIT Cognitive v1 — Multi-Network Architecture and Implementation

**Status:** The revised three-network implementation has passed CI, completed a Tesla T4 training run, and been published as the orbit-cognitive-v1 release. The latest inference-controller fixes (plural forms, lexical relation routing, and natural articles) are undergoing CI. See README.md for measured GPU results and limitations.

**Purpose:** Explore a different direction for ORBIT: several small, specialised neural networks cooperating through shared memory and a controlled reasoning loop, rather than asking one small autoregressive language model to do every task.

## 1. Core idea

ORBIT Cognitive would be a modular, neuro-symbolic system. Small neural networks would handle learned pattern-recognition tasks; explicit data structures and algorithms would handle memory, bookkeeping, rule application, and orchestration.

The networks would not simply chat with one another in free-form text. They would exchange typed, structured records through a shared workspace. A coordinator would decide which module should act next, preserve evidence, detect loops, and stop when it has enough support or must admit uncertainty.

This is still computation, not a claim of human-like consciousness or a literal biological brain.

## 2. Proposed modules

Start with a small set of modules and add specialists only when evaluations demonstrate a need.

| Module | Main responsibility | Possible implementation |
|---|---|---|
| Input encoder | Turn a question into a compact representation: intent, entities, relations, and constraints | Small randomly initialised text encoder or embedding network |
| Concept and association network | Suggest related concepts and likely relationships | Small link-prediction / metric-learning network |
| Memory retriever | Rank stored facts and past experiences relevant to the current question | Small query-key network; the actual memory store is ordinary structured data |
| Reasoning / action policy | Choose the next operation: retrieve, compare, apply a rule, ask for clarification, or stop | Small classifier or policy network |
| Evaluator | Check whether a proposed answer is supported, contradictory, incomplete, or uncertain | Small classifier plus explicit consistency checks |
| Answer composer | Express supported conclusions in clear language | Initially constrained templates; optionally a small language generator later |

A module does not need to be neural just because it is part of a cognitive architecture. The memory database, rule engine, task scheduler, and provenance tracker should initially be conventional code. This makes their behaviour inspectable and easier to debug.

## 3. Shared workspace and memory

All modules should use a common typed representation rather than passing paragraphs between one another.

Example fact record:

    {
      "subject": "rain",
      "relation": "is_a",
      "object": "precipitation",
      "source": "approved_training_fact_001",
      "confidence": 0.98
    }

The first prototype should distinguish:

- **Working memory:** temporary facts, the current question, candidate answers, and the steps taken for this question.
- **Semantic memory:** reusable concepts and relationships, with source/provenance and confidence.
- **Episodic memory:** a log of previous interactions and outcomes. Do not automatically turn every user statement into a trusted fact.
- **Procedural memory:** known operations and rules the controller can apply.

A memory entry should retain where it came from. Repetition alone must not make an unsupported statement become true.

## 4. Example: answering "What is rain?"

1. The input encoder identifies a definition request and the concept rain.
2. The retriever searches semantic memory for facts about rain.
3. The association network proposes useful related concepts such as water, cloud, and precipitation.
4. The reasoning controller follows stored relationships and constructs a candidate definition.
5. The evaluator checks that the candidate is supported by retrieved facts and does not contradict them.
6. The answer composer expresses the supported result, for example: "Rain is water that falls from clouds as precipitation."
7. If the required facts are missing or conflict, ORBIT should say that it does not know rather than inventing a definition.

The neural networks help interpret, associate, retrieve, and select operations. The evidence-bearing memory and explicit checks make the answer auditable.

## 5. How the modules learn

Each network needs a clear training objective and held-out evaluation set. Do not train all modules against one vague "intelligence" score.

- Input encoder: map varied phrasings to the same intent and concepts; evaluate entity and relation extraction accuracy.
- Association network: the starter version reconstructs the known 59 subject/relation groups with multi-positive object targets. This is a compact learned cache, not held-out link prediction. Evaluate generalisation only after building a larger graph and genuinely held-out edge split.
- Retriever: rank relevant facts above irrelevant ones; evaluate recall at small top-k values.
- Reasoning policy: learn which operation helps on small, verifiable tasks; compare it against a simple rule-based baseline.
- Evaluator: distinguish supported answers, unsupported answers, and contradictions; measure false acceptance as well as false rejection.
- Answer composer: produce short, grammatical responses that preserve the facts supplied to it; separately measure factual fidelity and readability.

Synthetic examples or teacher-written text may be used as training data, but record their origin and review samples. Do not copy teacher-model weights into any student network if the project is meant to preserve scratch-trained lineage.

## 6. Keep the first version small, but train it on a GPU

ORBIT Cognitive is intended to be a GPU-trained neural system. CPU execution is useful for unit tests, data inspection, and possibly lightweight inference, but it is **not** the training target for the specialist networks.

Start with three separately trained, randomly initialised specialists:

1. **PerceptionNet:** a small text encoder that maps a question to a structured frame (intent, entities, relations, and constraints).
2. **AssociationNet:** a small network that predicts relationships between concepts and proposes useful related concepts.
3. **MemoryNet:** a small retrieval/ranking network that selects relevant records from the external fact store.

After these work reliably, add a **ControlNet** to select the next operation and an optional learned evaluator. The first evaluator can combine deterministic evidence checks with a small classifier. Do not build five or six networks before the first three demonstrate measurable value.

The neural specialists should exchange typed records using one shared concept vocabulary, relation schema, and identifier format. Do not have them communicate by generating free-form paragraphs. That would make their interfaces ambiguous and harder to train.

Their exact widths and parameter counts should be selected after profiling on the actual Kaggle GPU assigned to the session. A modest combined model budget is preferable to arbitrary oversized networks. More modules increase coordination and training complexity; they do not automatically increase intelligence.

## 7. GPU training plan

**Training target:** Kaggle GPU sessions, using the repository's existing GitHub Actions-to-Kaggle approach where practical. The workflow must print and record the actual GPU model, CUDA/PyTorch versions, memory capacity, and whether CUDA is being used. Do not silently fall back to CPU training.

All Cognitive v1 student networks should begin from random initialisation. Do not load pretrained model weights into them. Teacher-generated examples may be used as labelled text data if provenance is recorded, but teacher weights and hidden states must not be copied into the students.

### Training stages

1. **Build and validate datasets.** Create labelled examples for intent/entity/relation parsing, concept-link prediction, retrieval ranking, and operation selection. Keep train/validation/test splits separate by source or underlying fact so near-duplicate examples cannot leak across splits.
2. **Train PerceptionNet on GPU.** Supervise structured parses from varied phrasings. Evaluate intent classification, entity extraction, and relation extraction separately.
3. **Train AssociationNet on GPU.** Use multi-positive object sets for each subject/relation pair. In the starter release, train it as a compact reconstruction cache of the known graph and label the metric as graph reconstruction, not held-out link prediction. Evaluate generalisation only after creating a genuinely held-out graph split with unseen facts.
4. **Train MemoryNet on GPU.** Use query-to-record relevance sets; when a query has multiple valid fact records, every valid record must count as a positive. Evaluate Recall@k on held-out question phrasings, with all valid records in the target set.
5. **Integrate and jointly tune.** Connect the networks through the shared typed workspace. Train or fine-tune the controller with labelled operation traces, then optionally tune the modules together with a weighted task loss. Retain each specialist's individual validation metrics so joint training cannot hide a regression.
6. **Package and evaluate.** Save every specialist checkpoint, shared schema/vocabulary, optimizer and scheduler state, random seeds, dataset fingerprints, metrics, and an integrated test report.

### Compute and checkpoint requirements

- Use the Kaggle GPU, and verify torch.cuda.is_available() before starting any training epoch. Log torch.cuda.get_device_name(0) and available memory.
- Use mixed precision only when supported by the assigned GPU; keep a safe full-precision fallback for unsupported operations.
- Synchronise the CUDA device at both timing boundaries before calculating per-stage runtime or examples/groups/queries per second.
- Use gradient accumulation and conservative batch sizes so the run fits the actual device.
- Save resumable checkpoints regularly and after each specialist stage. A workflow failure during GitHub publishing must not destroy the only copy of trained weights.
- Upload checkpoints and reports as workflow artifacts and/or a versioned release before attempting to update repository documentation. A failed README/report commit must not discard the model files.
- Use one Kaggle GPU job at a time. Do not run separate specialist jobs concurrently unless there is a clear quota and hardware reason.
- Do not start GPU training until the dataset generator, training script, validation script, and checkpoint recovery path have passed CPU-side tests.

The first release should be judged on held-out task performance, not on the fact that it ran on a GPU. GPU acceleration makes training practical; it does not guarantee useful learning.

## 8. Output generation

Do not make a general-purpose language generator the first milestone. Start with constrained, fact-preserving response templates driven by the verified structured result. This lets us test whether the cognitive modules retrieve and reason correctly without confusing that with fluent writing. A small answer-generation network can be added later and trained on GPU once the evidence pipeline is dependable.

## 9. Evaluation gates

Use a fixed held-out test set, including questions with different wording and questions whose answers are intentionally absent from memory.

Minimum early tests:

- Definition: "What is rain?"
- Paraphrase: "Explain what rainfall is."
- Related concepts: "How are clouds and rain connected?"
- Retrieval: "What fact do you have about cars?"
- Simple multi-step: "If rain is precipitation and precipitation is water falling from the atmosphere, what category does rain belong to?"
- Unknown: ask about a made-up object with no stored facts.
- Contradiction: provide two conflicting records and verify that ORBIT flags the conflict.

Track exact answer accuracy, evidence coverage, unsupported-claim rate, unknown-question abstention, latency, and memory use. Do not call the system intelligent just because it produces plausible sentences.

## 10. Relationship to ORBIT Seed v0.3

Keep orbit/seed_model_v0.3_scratch/ as a separate scratch-trained language-model experiment and baseline. Do not delete or silently repurpose its checkpoint.

The current v0.3 Kaggle workflow run is [41750482331](https://github.com/lm11-gif/ok/actions/runs/41750482331). Verified from its GitHub Actions job logs:

- Kaggle reported the notebook as KernelWorkerStatus.COMPLETE.
- The workflow downloaded the pretraining checkpoint, tokenizer, token bins, and training_report.json.
- The later step that tried to commit training_report.json and config.json to main failed because the push was rejected as non-fast-forward (fetch first).
- The model-release step was skipped. The run has no GitHub Actions artifacts, and the two report/config files were not present on main when checked.
- Therefore, a Kaggle run completed and output files were downloaded by the runner, but a published v0.3 release and a durable report in the repository have not been verified. Do not invent loss values, token counts, or SFT completion details from the workflow's overall status alone.

The user's sample outputs from ORBIT ("By choosing the snow that is rain", "I've got a Discovery Worksheet?", and a long sentence about "introitulation") are incoherent. The exact checkpoint and generation settings used for those samples have not yet been established. Treat them as qualitative evidence of poor observed output, not as a controlled evaluation.

Cognitive v1 should initially be developed independently of the v0.3 training pipeline. The v0.3 model could later be tested as an optional language-composition component, but its text-generation behaviour should not be allowed to invent or override facts in the structured memory.

## 11. Implementation status and next steps

### Implemented system

- core.py: seed graph, shared entity/relation IDs, vocabulary, grouped fact queries, and deterministic template splits.
- models.py: randomly initialised PerceptionNet, asymmetric AssociationNet, and MemoryNet.
- train.py: GPU-first staged training, validation/early stopping, checkpointing, CUDA-synchronised timing, parameter and throughput metrics, archive creation.
- chat.py: inference controller, relation cue routing, whole-word singular/plural concept matching, retrieval ranking, and constrained answer composition.
- bundle_kaggle.py: self-contained Kaggle script bundler.
- tests/test_cognitive.py: data/model tests and inference helper regression tests.
- .github/workflows/check-orbit-cognitive.yml: CPU-only CI with tests, bundled-script smoke training, checkpoint load/inference, and archive validation.
- .github/workflows/train-orbit-cognitive-kaggle.yml: manual Kaggle T4 training, validation, durable artifact upload, release update.

### Latest successful GPU run

GitHub Actions [41764168830](https://github.com/lm11-gif/ok/actions/runs/41764168830) successfully trained the current released checkpoints and updated the [orbit-cognitive-v1 release](https://github.com/lm11-gif/ok/releases/tag/orbit-cognitive-v1).

- GPU: Tesla T4, 14.56 GB reported memory, compute capability 7.5.
- Runtime: PyTorch 2.11.0+cu130 / CUDA 13.0.
- Optimisations: FP16 autocast, fused AdamW, tensors preloaded on GPU, batch size 256, vectorised retrieval, sequential model training; TF32 correctly disabled on the T4.
- Model sizes: PerceptionNet 93,877 parameters; AssociationNet 14,659; MemoryNet 85,632; total 194,168.
- PerceptionNet: 35 epochs, 1.152 s stage time, test subject accuracy 100%, test relation accuracy 72.0%.
- AssociationNet: 37 epochs, 0.189 s stage time, known-graph reconstruction top-1/top-5 100%. This is in-sample reconstruction across 59 known subject/relation groups, not generalisation.
- MemoryNet: 32 epochs, 0.438 s stage time, multi-positive test Recall@1 71.2%, Recall@5 89.8%.
- Overall trainer-reported duration 6.776 s; peak allocated GPU memory 0.029 GB.
- Throughput: approx. 10,755 perception examples/s, 11,543 association groups/s, 25,848 memory queries/s.

The earlier T4 run took 7.84 s but used different objectives and unsynchronised CUDA timing. Do not claim a speedup from comparing those two totals; there is no controlled unoptimised baseline.

### Limits / interpretation

The seed graph has only 67 facts, 67 entities, and 18 relation types. Perception and Memory test examples are held-out question templates over facts already in the graph. Association is explicitly trained and evaluated as memorisation/reconstruction over the same 59 graph groups. These results prove that the small specialist pipeline trains and retrieves some stored data; they do not show broad factual knowledge, independent reasoning, or human-like cognition.

A local smoke harness loaded the published state dictionaries and checked common prompts such as “What is rain?”, “What does a tree need?”, “What does RAM store?”, “Where does a river flow?”, and the unknown concept “dragon”. Known seed facts were returned, including multiple objects for tree/needs; the unknown concept was rejected. This is a narrow sanity check, not a broad benchmark.

### Current CI and next steps

- Revised training code, timing, and the release have passed CI at [41763985893](https://github.com/lm11-gif/ok/actions/runs/41763985893) and the Kaggle training run [41764168830](https://github.com/lm11-gif/ok/actions/runs/41764168830).
- The newest inference-controller edits and tests are in the current CI run triggered by commit b56d28376e0c84d9b59c14fde2edd4f42d0e2cb0; inspect the live conclusion before declaring the latest source fully green.
- No retraining is needed for changes limited to chat.py and tests. The release checkpoint architecture remains the same as models.py.
- Next substantive development: expand and review the fact graph; create independently authored, manually verified question sets; evaluate relation parsing, unknown abstention, evidence coverage, contradiction detection, and simple multi-step inference; then add an explicit rule engine/controller. Avoid adding more neural models until those tests establish a need.

Keep ORBIT Seed v0.3 as a separate scratch-language-model experiment. Its Kaggle execution completed earlier, but its report commit was rejected and no v0.3 release was confirmed.
