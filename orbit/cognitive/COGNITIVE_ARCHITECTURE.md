# ORBIT Cognitive v1 — Multi-Network Architecture Proposal

**Status:** Design proposal only. This document does not claim that the architecture has been implemented or trained.

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
- Association network: predict held-out concept links; evaluate precision and recall, not just training loss.
- Retriever: rank relevant facts above irrelevant ones; evaluate recall at small top-k values.
- Reasoning policy: learn which operation helps on small, verifiable tasks; compare it against a simple rule-based baseline.
- Evaluator: distinguish supported answers, unsupported answers, and contradictions; measure false acceptance as well as false rejection.
- Answer composer: produce short, grammatical responses that preserve the facts supplied to it; separately measure factual fidelity and readability.

Synthetic examples or teacher-written text may be used as training data, but record their origin and review samples. Do not copy teacher-model weights into any student network if the project is meant to preserve scratch-trained lineage.

## 6. Keep the first prototype small

The first milestone should be a CPU-friendly proof of concept, not a large collection of expensive models.

Suggested order:

1. Build the typed fact store, provenance records, working memory, and deterministic controller.
2. Add one small input encoder and one small retriever/associator.
3. Test a narrow set of factual questions with known answers.
4. Add a reasoning-policy network only after the retrieval baseline works.
5. Add the evaluator and measure whether it catches unsupported answers.
6. Add a language composer last; begin with templates so answer quality is not confused with reasoning quality.

Start with perhaps two or three small neural modules, not six independently trained models. Their size should be chosen after measuring CPU speed and memory use on the target machine. More modules increase integration and training complexity; they do not automatically increase intelligence.

## 7. Evaluation gates

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

## 8. Relationship to ORBIT Seed v0.3

Keep orbit/seed_model_v0.3_scratch/ as a separate scratch-trained language-model experiment and baseline. Do not delete or silently repurpose its checkpoint.

The current v0.3 Kaggle workflow run is [41750482331](https://github.com/lm11-gif/ok/actions/runs/41750482331). Verified from its GitHub Actions job logs:

- Kaggle reported the notebook as KernelWorkerStatus.COMPLETE.
- The workflow downloaded the pretraining checkpoint, tokenizer, token bins, and training_report.json.
- The later step that tried to commit training_report.json and config.json to main failed because the push was rejected as non-fast-forward (fetch first).
- The model-release step was skipped. The run has no GitHub Actions artifacts, and the two report/config files were not present on main when checked.
- Therefore, a Kaggle run completed and output files were downloaded by the runner, but a published v0.3 release and a durable report in the repository have not been verified. Do not invent loss values, token counts, or SFT completion details from the workflow's overall status alone.

The user's sample outputs from ORBIT ("By choosing the snow that is rain", "I've got a Discovery Worksheet?", and a long sentence about "introitulation") are incoherent. The exact checkpoint and generation settings used for those samples have not yet been established. Treat them as qualitative evidence of poor observed output, not as a controlled evaluation.

Cognitive v1 should initially be developed independently of the v0.3 training pipeline. The v0.3 model could later be tested as an optional language-composition component, but its text-generation behaviour should not be allowed to invent or override facts in the structured memory.

## 9. Immediate next steps

- Preserve v0.3 and recover/record its actual Kaggle training report if the Kaggle output is still accessible.
- Create a small CPU prototype of the memory store, typed workspace, and controller.
- Add a tiny scratch-trained encoder/retriever only when there is a labelled dataset and a measurable baseline.
- Test the system on a fixed question set before deciding whether to expand it.
- Keep each experiment, dataset, checkpoint, and training lineage documented in GitHub.

No new Kaggle training session is required for the initial cognitive-architecture prototype.
