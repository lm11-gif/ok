#!/usr/bin/env python3
"""Teacher-supervised preference generation and DPO for ORBIT Seed v0.2.

  Uses a sequential teacher-generation then DPO-training schedule to fit a T4.
  Preference data is retained in the output archive for human inspection.
"""

from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Preserve Kaggle's CUDA-enabled PyTorch, but pin the Hugging Face stack.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet",
        "transformers==4.53.3",
        "datasets==3.3.2",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "trl==0.15.2",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
    ],
    check=True,
)

# Kaggle images may include an old optional torchao build which makes PEFT
# raise during adapter setup. ORBIT does not use torchao.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

SFT_ARCHIVE_URL = (
    "https://github.com/lm11-gif/ok/releases/download/"
    "orbit-seed-v0.2/orbit-seed-v0.2.zip"
)
TEACHER_MODEL = "HuggingFaceTB/SmolLM3-3B"
TEACHER_REVISION = "main"
SEED = 20261104
MAX_PAIRS = 160
MIN_PAIRS = 80
MAX_PROMPT_TOKENS = 768
MAX_ANSWER_TOKENS = 220
WORK = Path("/kaggle/working")
SFT_DIR = WORK / "orbit-seed-v0.2-sft"
OUTPUT_DIR = WORK / "orbit-seed-v0.2-dpo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Explain the difference between RAM and storage to someone who has never built a computer.",
    "Why does a metal spoon feel colder than a wooden spoon when both are in the same room?",
    "Explain how a rainbow forms without using complicated science terms.",
    "What is the difference between weather and climate?",
    "Explain recursion using a simple everyday analogy and a tiny Python example.",
    "Give me a practical, beginner-friendly plan for learning Python over two weeks.",
    "Explain what an API is using a restaurant analogy, then give a small real example.",
    "What is the difference between a list, tuple, and dictionary in Python?",
    "Write a Python function that checks whether a string is a palindrome. Include two tests.",
    "Write a Python function that removes duplicate items from a list while preserving order.",
    "Explain why this Python loop never stops: while x != 10: x += 2. Suggest a fix.",
    "Show how to read a JSON file in Python and handle a missing or invalid file safely.",
    "Explain the difference between a class and an object with a short example.",
    "A program works on my computer but fails on another. Give me a sensible debugging checklist.",
    "Explain what a Git commit is and how it differs from a Git push.",
    "Describe a safe way to add autosaving to a small game.",
    "Design a simple save-file format for a small single-player game. Explain the trade-offs.",
    "A Minecraft mod crashes on startup with a missing dependency error. What should I check first?",
    "Explain the difference between client-side and server-side logic in a multiplayer game.",
    "Suggest a lightweight architecture for 100 villagers who need to gather resources and build homes.",
    "How could a game simulate distant NPCs without keeping every one fully active?",
    "Explain pathfinding with A* in simple terms and list its main steps.",
    "What is a behavior tree, and when might it be better than a giant collection of if statements?",
    "Explain the difference between an LLM, a tokenizer, and an embedding.",
    "What does quantization do to a language model, and what trade-offs does it create?",
    "Explain why a model with more parameters is not automatically better at every task.",
    "What is the difference between pretraining, supervised fine-tuning, and preference tuning?",
    "Explain Direct Preference Optimization in plain language, including what chosen and rejected answers mean.",
    "Why should an AI assistant admit uncertainty instead of making up an answer?",
    "Give an example of a good clarifying question when a programming request is ambiguous.",
    "Rewrite this sentence more clearly without changing its meaning: The implementation was done in a way that was not ideal for the use case.",
    "Summarize the purpose of a technical README in five concise points.",
    "Write a polite message explaining that a software update is delayed because testing found a bug.",
    "Write a short story about a robot that learns to repair an old lighthouse.",
    "Create a vivid but concise description of a quiet village during the first snowfall.",
    "Give three distinct names for a science-fiction research station and explain the tone of each.",
    "Help me turn a vague idea for a puzzle game into a small, testable prototype plan.",
    "Suggest five original mechanics for a cooperative survival game. Avoid generic crafting-only ideas.",
    "Explain how to balance a game mechanic that is powerful but has a long cooldown.",
    "A player says my game feels unfair. What questions should I ask before changing the difficulty?",
    "Explain how binary search works and when it is useful.",
    "What is the time complexity of checking every pair in a list of n items? Explain why.",
    "Explain the difference between a process and a thread.",
    "What is a race condition? Give a small illustrative example.",
    "Why is input validation important even in a local application?",
    "Explain the principle of least privilege with a practical software example.",
    "Describe how to back up a project before making a risky refactor.",
    "A user reports that an app is slow. Give a methodical way to locate the bottleneck.",
    "Explain what a database index does and why too many indexes can be harmful.",
    "Describe a simple schema for storing users and their saved game worlds.",
    "What is an HTTP status code, and what do 200, 404, and 500 mean?",
    "Explain the difference between authentication and authorization.",
    "How can I make an AI-generated answer easier to verify?",
    "Give an example of a response that is honest about not having executed code.",
    "Explain why test coverage is not the same as proving a program is correct.",
    "Create a small checklist for reviewing a pull request that changes core game logic.",
    "What should a good bug report include?",
    "Explain why random seeds are useful when debugging procedural generation.",
    "Describe a simple algorithm for generating a connected maze.",
    "How could I prevent NPCs from all choosing the same resource node?",
    "Explain how a priority queue helps with pathfinding.",
    "A game has frame-rate drops when many NPCs are active. Suggest profiling steps before rewriting the AI.",
    "Explain the difference between a hard-coded rule and a learned behavior.",
    "Give a balanced comparison of using a local model versus a hosted API for a hobby project.",
    "Explain why context length and generation speed are different model performance measures.",
    "How would you test whether a new model version is actually better than the previous one?",
    "Explain how data leakage can make a model evaluation look better than it really is.",
    "Describe three ways a training dataset can accidentally teach unwanted behavior.",
    "Why can synthetic training data amplify a teacher model's mistakes?",
    "Give a practical method for filtering low-quality synthetic preference pairs.",
    "Explain why DPO needs a reference policy and what role it plays.",
    "What is catastrophic forgetting in fine-tuning, and how can it be reduced?",
    "Explain why a low training loss does not guarantee good conversations.",
    "Give a simple example of overfitting in language-model fine-tuning.",
    "Explain gradient accumulation and why it helps when GPU memory is limited.",
    "What does a learning-rate warmup do?",
    "Why might a smaller learning rate be safer when fine-tuning an already capable model?",
    "Explain the difference between validation data and test data.",
    "A model's replies became repetitive after fine-tuning. List plausible causes and checks.",
    "Help me break a large programming task into milestones that can each be tested.",
    "Explain how to write a minimal reproducible example for a software bug.",
    "Write a concise explanation of why an application needs a restart after updating a library.",
    "Explain the difference between a compiler error and a runtime error.",
    "What is a memory leak? Give an example in a game loop.",
    "Describe a safe approach to loading user-created plugins.",
    "Explain why an AI coding agent should inspect project files before proposing edits.",
    "What should an AI coding agent do when it cannot access a required tool?",
    "Write a response to a user who asks for a feature that is outside the current software's capabilities. Be direct but helpful.",
    "Explain how to distinguish a confirmed fact from an assumption in a technical answer.",
    "Give an example of a concise troubleshooting answer that starts with the most likely cause.",
    "Explain why changing one variable at a time can make debugging easier.",
    "What are the risks of automatically accepting every suggestion from an AI reviewer?",
    "Describe how to compare two implementations fairly using the same inputs and environment.",
    "How should an assistant respond when the user corrects a mistaken assumption?",
    "Explain how to preserve backward compatibility when changing a configuration file.",
    "Suggest a clear folder structure for a small Python game project.",
    "Write a basic Python dataclass for a 2D position with x and y coordinates.",
    "Explain why deterministic tests are valuable for game AI.",
    "How can a developer prevent an NPC's decision loop from running every frame?",
    "Explain the difference between local simulation and abstract simulation for distant agents.",
    "A player can clip through the floor after a physics update. Suggest a systematic debugging approach.",
    "Explain why collision detection and collision response are separate parts of a physics engine.",
    "How could an agent plan a sequence of actions to gather wood, build a tool, and construct a shelter?",
    "Explain the difference between breadth-first search and depth-first search.",
    "Give a simple example of an invariant in a resource-management simulation.",
    "What information should an AI agent log when it repeatedly fails to complete a task?",
    "How can a system avoid having multiple agents reserve the same limited resource?",
    "Explain why a model should not claim that a test passed unless the test actually ran.",
    "Write a short, friendly explanation of what an open-source license allows and what it does not automatically guarantee.",
    "Give an example of a useful answer when the question lacks enough information for a confident conclusion.",
    "Explain the difference between correlation and causation with an everyday example.",
    "A friend is overwhelmed by a large project. Suggest a realistic way to choose the next small step.",
    "Explain how to compare two GPUs for a workload without relying only on their marketing names.",
    "What are the main reasons a laptop may slow down during a long compute task?",
    "Explain why thermal throttling can reduce performance even when the software is unchanged.",
    "How should a developer decide whether a bug needs a quick patch or a deeper redesign?",
    "Write a concise release note for a bug fix that prevents saved games from being overwritten.",
    "Explain what a checksum is and how it can detect file corruption.",
    "Describe how to make a backup-and-restore feature safer for user data.",
    "What is the difference between a model checkpoint and an inference-ready model export?",
    "Explain why tokenizer files must match the language model weights.",
    "Give a simple explanation of why padding tokens should not accidentally be included in a language-model training loss.",
    "What should an evaluation report include so another person can reproduce the result?",
    "Explain why a teacher model should not be treated as an unquestionable source of truth.",
    "Suggest a way to mix easy and difficult examples during preference tuning.",
    "How can an AI assistant keep answers concise without omitting important caveats?",
    "Write a small Python example that safely handles division by zero.",
    "Explain the purpose of type hints in Python.",
    "What is the difference between a shallow copy and a deep copy?",
    "Describe a simple strategy for caching expensive calculations in a simulation.",
    "A web request times out intermittently. Suggest likely causes and a way to investigate.",
    "Explain why retries should use limits and backoff rather than looping forever.",
    "Give a brief checklist for publishing a small open-source Python package.",
    "How should an AI agent report partial completion when a multi-step task fails halfway through?",
    "Explain how to tell a user that a result is an estimate rather than a measured value.",
    "Write a clear step-by-step explanation of how to create a new Git branch and switch to it.",
    "Explain the difference between merging and rebasing in Git at a beginner level.",
    "Describe how a continuous-integration workflow can catch syntax errors before GPU training starts.",
    "Why should training workflows avoid printing API tokens or other credentials?",
    "Explain the benefit of separating data generation, training, and evaluation into different stages.",
    "Suggest a sensible first benchmark for a small conversational assistant.",
    "How can a benchmark accidentally reward verbosity instead of correctness?",
    "Explain why a model can sound confident while being wrong.",
    "Give a short, practical explanation of precision, recall, and F1 score.",
    "Explain how to make a technical answer useful to both beginners and experienced developers.",
];

def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available. Enable a Kaggle GPU accelerator.")
    return torch.cuda.get_device_name(0)

def download_sft_checkpoint() -> None:
    import requests

    archive = WORK / "orbit-seed-v0.2-sft.zip"
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        print("Downloading the completed SFT checkpoint from the GitHub release...", flush=True)
        with requests.get(SFT_ARCHIVE_URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with archive.open("wb") as target:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        target.write(chunk)
    if SFT_DIR.exists():
        shutil.rmtree(SFT_DIR)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(SFT_DIR)
    if not list(SFT_DIR.glob("*.safetensors")):
        nested = next((p for p in SFT_DIR.rglob("*.safetensors")), None)
        if nested is None:
            raise RuntimeError("The SFT release archive contains no safetensors model weights.")
        extracted_root = nested.parent
    else:
        extracted_root = SFT_DIR
    # Store the real checkpoint directory for subsequent stages.
    (WORK / "sft_model_path.txt").write_text(str(extracted_root), encoding="utf-8")

def load_student(path: Path, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = True
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model

def make_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are ORBIT, a helpful, honest, clear assistant. Stay on topic and do not invent facts.",
        },
        {"role": "user", "content": user_text},
    ]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "User: " + user_text + "\nAssistant:"

def generate_answer(model, tokenizer, prompt_text: str, max_new_tokens: int, temperature: float):
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.08,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs.update({"temperature": temperature, "top_p": 0.9})
    with torch.inference_mode():
        generated = model.generate(**encoded, **kwargs)
    new_tokens = generated[0, encoded["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer

def teacher_review_prompt(user_request: str, student_draft: str) -> str:
    return (
        "/no_think\nYou are a careful senior AI assistant supervising a smaller assistant called ORBIT. "
        "Produce the final answer that ORBIT should have given. The draft may be incomplete, "
        "unclear, or incorrect. Correct mistakes, answer the actual request, explain important "
        "steps, and keep the response proportionate. Do not invent facts; clearly state uncertainty "
        "where appropriate. Do not mention this review process. Output only the improved final answer.\n\n"
        "USER REQUEST:\n" + user_request.strip() + "\n\n"
        "ORBIT'S DRAFT:\n" + student_draft.strip() + "\n\n"
        "IMPROVED FINAL ANSWER:"
    )

def clean_answer(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<\|[^|]+\|>", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()

def main():
    started = time.time()
    gpu_name = require_cuda()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(json.dumps({
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "teacher": TEACHER_MODEL,
        "max_pairs": MAX_PAIRS,
    }, indent=2), flush=True)

    download_sft_checkpoint()
    model_path = Path((WORK / "sft_model_path.txt").read_text(encoding="utf-8").strip())
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    prompts = PROMPTS[:MAX_PAIRS]
    if len(prompts) < MIN_PAIRS:
        raise RuntimeError(f"Prompt bank has only {len(prompts)} items; need at least {MIN_PAIRS}.")

    # Stage 1: capture ORBIT's current answers as rejected candidates.
    print("Generating baseline ORBIT drafts...", flush=True)
    student = load_student(model_path, tokenizer)
    drafts = []
    for index, prompt in enumerate(prompts):
        formatted = make_chat_prompt(tokenizer, prompt)
        draft = generate_answer(student, tokenizer, formatted, MAX_ANSWER_TOKENS, 0.65)
        drafts.append(draft)
        if (index + 1) % 16 == 0:
            print(f"Student drafts: {index + 1}/{len(prompts)}", flush=True)
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2: teacher reviews the draft and supplies the preferred answer.
    print("Loading SmolLM3-3B teacher...", flush=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL,
        revision=TEACHER_REVISION,
        use_fast=True,
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        revision=TEACHER_REVISION,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    teacher.eval()
    rows = []
    for index, (prompt, draft) in enumerate(zip(prompts, drafts)):
        if len(draft.strip()) < 12:
            continue
        review_text = teacher_review_prompt(prompt, draft)
        if teacher_tokenizer.chat_template:
            teacher_messages = [
                {
                    "role": "system",
                    "content": "/no_think\nYou are a careful, accurate assistant improving a junior model's draft.",
                },
                {"role": "user", "content": review_text},
            ]
            formatted = teacher_tokenizer.apply_chat_template(
                teacher_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = "User: " + review_text + "\nAssistant:"
        improved = clean_answer(
            generate_answer(teacher, teacher_tokenizer, formatted, MAX_ANSWER_TOKENS, 0.25)
        )
        rejected = clean_answer(draft)
        if len(improved) < 24 or len(rejected) < 12:
            continue
        if improved.casefold() == rejected.casefold():
            continue
        if len(improved) > 6000 or len(rejected) > 6000:
            continue
        rows.append({
            "prompt": make_chat_prompt(tokenizer, prompt),
            "chosen": improved,
            "rejected": rejected,
            "source_prompt": prompt,
        })
        if (index + 1) % 16 == 0:
            print(f"Teacher-reviewed pairs: {index + 1}/{len(prompts)}; kept={len(rows)}", flush=True)

    del teacher
    del teacher_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if len(rows) < MIN_PAIRS:
        raise RuntimeError(
            f"Only {len(rows)} usable preference pairs were generated; need at least {MIN_PAIRS}. "
            "Refusing to train DPO on a tiny or low-quality dataset."
        )

    # Save the exact synthetic preference data for inspection and reproducibility.
    with (OUTPUT_DIR / "preferences.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset_rows = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    dataset = Dataset.from_list(dataset_rows).train_test_split(test_size=0.1, seed=SEED)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    # Stage 3: DPO. The teacher is unloaded; only the 360M policy and its
    # reference behavior are needed during this stage.
    print("Loading ORBIT SFT checkpoint for DPO...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    args = DPOConfig(
        output_dir=str(WORK / "orbit-v02-dpo-training"),
        beta=0.1,
        loss_type="sigmoid",
        max_length=512,
        max_prompt_length=256,
        num_train_epochs=2.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5.0e-6,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=0,
        max_grad_norm=0.5,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(
        json.dumps({
            "preference_pairs": len(rows),
            "train_pairs": len(train_ds),
            "eval_pairs": len(eval_ds),
            "teacher": TEACHER_MODEL,
            "beta": args.beta,
            "epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
        }, indent=2),
        flush=True,
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    print("Merging DPO LoRA adapter into ORBIT...", flush=True)
    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="500MB",
    )
    tokenizer.save_pretrained(OUTPUT_DIR)
    # Preserve the upstream SFT notices/license material with the DPO export.
    for notice_name in ("NOTICE.md", "UPSTREAM_LICENSE"):
        source_notice = model_path / notice_name
        if source_notice.is_file():
            shutil.copy2(source_notice, OUTPUT_DIR / notice_name)

    # Compare SFT and DPO on the same held-out-from-optimization prompts.
    # These are generated from the original student drafts captured above.
    smoke_prompts = prompts[:5]
    generations = []
    merged.eval()
    for index, prompt in enumerate(smoke_prompts):
        formatted = make_chat_prompt(tokenizer, prompt)
        answer = generate_answer(merged, tokenizer, formatted, 160, 0)
        comparison = {
            "prompt": prompt,
            "sft_answer": drafts[index],
            "dpo_answer": answer,
        }
        generations.append(comparison)
        print(
            f"COMPARISON PROMPT: {prompt}\nSFT: {drafts[index]}\nDPO: {answer}\n",
            flush=True,
        )

    report = {
        "name": "ORBIT Seed v0.2 DPO",
        "training_type": "Teacher-generated preference pairs followed by Direct Preference Optimization",
        "base_checkpoint": "ORBIT Seed v0.2 SFT",
        "teacher_model": TEACHER_MODEL,
        "teacher_model_license": "Apache-2.0",
        "teacher_model_source": "https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
        "dpo_library": "trl==0.15.2",
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "prompt_count": len(prompts),
        "preference_pairs_kept": len(rows),
        "train_pairs": len(train_ds),
        "eval_pairs": len(eval_ds),
        "beta": args.beta,
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "train_metrics": train_result.metrics,
        "heldout_metrics": eval_result,
        "elapsed_seconds": round(time.time() - started, 1),
        "smoke_test_generations": generations,
        "limitations": [
            "Preference pairs are synthetic and teacher-generated; the teacher can still be wrong.",
            "DPO preference loss is not a factuality score.",
            "Inspect preferences.jsonl and compare smoke tests before treating this checkpoint as better.",
        ],
    }
    (OUTPUT_DIR / "dpo_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "NOTICE.md").write_text(
        "# ORBIT Seed v0.2 DPO notices\n\n"
        "This is a preference-tuned derivative of the ORBIT Seed v0.2 SFT checkpoint, "
        "which itself is a fine-tune of HuggingFaceTB/SmolLM2-360M.\n\n"
        "Teacher: HuggingFaceTB/SmolLM3-3B, Apache-2.0: "
        "https://huggingface.co/HuggingFaceTB/SmolLM3-3B\n\n"
        "See dpo_training_report.json for settings, evaluation metrics and limitations. "
        "See preferences.jsonl for the generated preference pairs.\n",
        encoding="utf-8",
    )

    archive = WORK / "orbit-seed-v0.2-dpo.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=OUTPUT_DIR)
    print("ORBIT_DPO_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    print(f"ORBIT_DPO_ARCHIVE={archive} bytes={archive.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
