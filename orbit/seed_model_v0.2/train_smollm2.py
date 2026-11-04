#!/usr/bin/env python3
"""Fine-tune SmolLM2-360M into ORBIT Seed v0.2 on a Kaggle GPU.

This is an instruction fine-tune of an Apache-2.0 pretrained base model,
not training a language model from random initialization.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Kaggle kernels do not all ship with the same NLP packages. Install pinned,
# mutually compatible versions without replacing the preinstalled CUDA PyTorch.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet",
        "transformers==4.48.3",
        "datasets==3.3.2",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
    ],
    check=True,
)

# Kaggle currently ships torchao 0.10 on some images, while the installed
# PEFT version expects torchao >=0.16 if that optional backend is present.
# ORBIT does not use torchao, so remove the incompatible optional package.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
CHAT_TOKENIZER = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATASET_ID = "HuggingFaceTB/smol-smoltalk"
TRAIN_EXAMPLES = 20_000
EVAL_EXAMPLES = 1_000
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 1.0
LEARNING_RATE = 1.0e-4
SEED = 20261104

WORK = Path("/kaggle/working")
OUTPUT_DIR = WORK / "orbit-seed-v0.2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. Enable a Kaggle GPU accelerator and restart."
        )
    return torch.cuda.get_device_name(0)

def make_messages(raw):
    cleaned = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        cleaned.append({"role": role, "content": content.strip()})
    if not cleaned or not any(m["role"] == "assistant" for m in cleaned):
        return None
    return cleaned

def main():
    gpu_name = require_cuda()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if not hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        pass
    torch.backends.cuda.matmul.allow_tf32 = True

    print(json.dumps({
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "base_model": BASE_MODEL,
        "dataset": DATASET_ID,
        "training_examples": TRAIN_EXAMPLES,
        "evaluation_examples": EVAL_EXAMPLES,
        "max_length": MAX_LENGTH,
    }, indent=2), flush=True)

    started = time.time()

    print("Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CHAT_TOKENIZER, use_fast=True)
    if not tokenizer.chat_template:
        raise RuntimeError("The selected SmolLM2 instruct tokenizer has no chat template.")

    # The upstream SmolLM2 template has no {generation} markers. Add them
    # around assistant messages so we can train only on assistant responses,
    # not on the user/system prompt tokens.
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\\nYou are a helpful AI assistant named ORBIT<|im_end|>\\n' }}"
        "{% endif %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% generation %}{{ message['content'] }}{{ '<|im_end|>\\n' }}{% endgeneration %}"
        "{% else %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )

    # Use a distinct padding token so padding can be excluded from the loss.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|orbit_pad|>"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("Downloading the Apache-2.0 Smol-SmolTalk dataset...", flush=True)
    train_raw = load_dataset(DATASET_ID, split="train")
    test_raw = load_dataset(DATASET_ID, split="test")

    train_raw = train_raw.shuffle(seed=SEED).select(
        range(min(TRAIN_EXAMPLES, len(train_raw)))
    )
    test_raw = test_raw.shuffle(seed=SEED + 1).select(
        range(min(EVAL_EXAMPLES, len(test_raw)))
    )

    def tokenize_batch(batch):
        all_ids = []
        all_masks = []
        all_labels = []
        for raw_messages in batch["messages"]:
            messages = make_messages(raw_messages)
            if messages is None:
                continue
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                ids = encoded["input_ids"]
                assistant_mask = encoded.get("assistant_masks")
                if assistant_mask is None:
                    raise RuntimeError(
                        "Tokenizer did not return assistant_masks; refusing to train on user text."
                    )
                if len(ids) < 8 or not any(assistant_mask):
                    continue
                labels = [
                    token_id if is_assistant else -100
                    for token_id, is_assistant in zip(ids, assistant_mask)
                ]
                all_ids.append(ids)
                all_masks.append(encoded["attention_mask"])
                all_labels.append(labels)
            except Exception as exc:
                print(f"Skipping malformed conversation: {exc}", flush=True)
        return {
            "input_ids": all_ids,
            "attention_mask": all_masks,
            "labels": all_labels,
        }

    class AssistantOnlyCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            model_inputs = [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ]
            batch = self.tokenizer.pad(
                model_inputs,
                padding=True,
                return_tensors="pt",
            )
            labels = torch.full_like(batch["input_ids"], -100)
            for row, feature in enumerate(features):
                row_labels = torch.tensor(feature["labels"], dtype=torch.long)
                labels[row, : row_labels.numel()] = row_labels
            batch["labels"] = labels
            return batch

    print("Tokenizing train and held-out test conversations...", flush=True)
    train_ds = train_raw.map(
        tokenize_batch,
        batched=True,
        batch_size=128,
        remove_columns=train_raw.column_names,
        desc="Tokenizing ORBIT training conversations",
    )
    eval_ds = test_raw.map(
        tokenize_batch,
        batched=True,
        batch_size=128,
        remove_columns=test_raw.column_names,
        desc="Tokenizing held-out evaluation conversations",
    )
    if len(train_ds) < 1000 or len(eval_ds) < 100:
        raise RuntimeError(
            f"Not enough usable conversations after filtering: train={len(train_ds)}, eval={len(eval_ds)}"
        )

    collator = AssistantOnlyCollator(tokenizer)
    supervised_tokens = sum(
        sum(1 for token_id in row["labels"] if token_id != -100)
        for row in train_ds
    )
    if supervised_tokens < 10_000:
        raise RuntimeError(
            f"Too few assistant-only target tokens after preprocessing: {supervised_tokens}"
        )
    print(f"Assistant-only supervised tokens: {supervised_tokens:,}", flush=True)
    args = TrainingArguments(
        output_dir=str(WORK / "orbit-v02-training"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="no",
        save_strategy="no",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
        group_by_length=True,
        max_grad_norm=1.0,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Running the supervised fine-tuning stage...", flush=True)
    train_result = trainer.train()
    print("Evaluating on the held-out test split...", flush=True)
    eval_result = trainer.evaluate()

    # Merge LoRA into the base weights so local inference needs only one model.
    print("Merging LoRA adapters into the base model...", flush=True)
    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="500MB",
    )
    tokenizer.save_pretrained(OUTPUT_DIR)

    prompts = [
        "Hi! What can you help me with?",
        "Explain why the sky looks blue in simple terms.",
        "I am stuck learning Python. What is one practical first step?",
        "Write a short story about a lost dog finding its way home.",
    ]
    generations = []
    merged.eval()
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are ORBIT, a helpful and clear assistant. Answer naturally and stay on topic."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            generated = merged.generate(
                input_ids,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            generated[0, input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        generations.append({"prompt": prompt, "answer": answer})
        print(f"PROMPT: {prompt}\nORBIT: {answer}\n", flush=True)

    # Preserve upstream license text with the merged model.
    try:
        from huggingface_hub import hf_hub_download
        license_file = hf_hub_download(repo_id=BASE_MODEL, filename="LICENSE")
        shutil.copy2(license_file, OUTPUT_DIR / "UPSTREAM_LICENSE")
    except Exception as exc:
        print(f"Could not fetch upstream LICENSE file automatically: {exc}", flush=True)

    report = {
        "name": "ORBIT Seed v0.2",
        "training_type": "LoRA supervised fine-tuning of pretrained SmolLM2-360M",
        "from_scratch": False,
        "base_model": BASE_MODEL,
        "base_model_license": "Apache-2.0",
        "chat_tokenizer": CHAT_TOKENIZER,
        "dataset": DATASET_ID,
        "dataset_license": "Apache-2.0",
        "dataset_source": "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk",
        "base_model_source": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M",
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "training_examples_requested": TRAIN_EXAMPLES,
        "training_examples_used": len(train_ds),
        "heldout_examples_used": len(eval_ds),
        "max_sequence_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_metrics": train_result.metrics,
        "heldout_metrics": eval_result,
        "elapsed_seconds": round(time.time() - started, 1),
        "generations": generations,
        "warning": "A low evaluation loss does not prove response quality. Inspect the generations and test the model interactively.",
    }
    (OUTPUT_DIR / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    notice = """# ORBIT Seed v0.2 notices

This model is a fine-tuned derivative of HuggingFaceTB/SmolLM2-360M.
The base model is released under Apache License 2.0:
https://huggingface.co/HuggingFaceTB/SmolLM2-360M

The conversation fine-tuning data is HuggingFaceTB/smol-smoltalk,
which is listed under Apache License 2.0:
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

See UPSTREAM_LICENSE for the upstream model license. See training_report.json
for exact training settings and generated smoke-test outputs.
"""
    (OUTPUT_DIR / "NOTICE.md").write_text(notice, encoding="utf-8")

    archive = WORK / "orbit-seed-v0.2.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=OUTPUT_DIR)
    print("ORBIT_V02_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    print(f"ORBIT_V02_ARCHIVE={archive} bytes={archive.stat().st_size}", flush=True)

if __name__ == "__main__":
    main()
