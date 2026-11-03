#!/usr/bin/env python3
"""Run ORBIT Seed v0.2 locally using Transformers."""

from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, args.cpu_threads))
        dtype = torch.float32
        device = "cpu"
    else:
        dtype = torch.float16
        device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    messages = [
        {"role": "system", "content": "You are ORBIT, a helpful and clear assistant. Answer naturally, accurately, and stay on topic."}
    ]
    print("ORBIT Seed v0.2 ready. Type /exit to quit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            break

        messages.append({"role": "user", "content": prompt})
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            output[0, inputs.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        print("\nORBIT:", answer)
        messages.append({"role": "assistant", "content": answer})

        # Keep the dialogue context bounded on long sessions.
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

if __name__ == "__main__":
    main()
