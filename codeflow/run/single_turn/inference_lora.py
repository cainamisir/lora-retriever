import os
import json
import sys
import argparse
from typing import List
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lora_manager import LoRAPerTagManager, extract_target_code
from src.utils import (
    get_filenames_without_extension,
    extract_code,
    get_input_single,
    ensure_python_code_block,
)


def select_tag(problem: dict) -> str:
    tags = problem.get("tags") or []
    if isinstance(tags, list) and tags:
        return str(tags[0])
    return "default"


def load_data(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(args):
    use_lora = args.update_interval > 0
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.input_file)
    data = sorted(data, key=lambda p: (select_tag(p), p.get("problem-id", "")))

    filename_list = get_filenames_without_extension(args.output_dir)
    tag_counts = defaultdict(int)

    lora_mgr = LoRAPerTagManager(
        model_path=args.model_path,
        adapter_root=args.adapter_root,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        device=args.device,
        train_device=args.train_device,
        train_max_length=args.train_max_length,
        train_lr=args.train_lr,
        device_map=args.device_map,
        target_modules=args.target_modules,
        load_in_8bit=args.load_in_8bit,
        bnb_compute_dtype=args.bnb_compute_dtype,
        load_in_4bit=args.load_in_4bit,
    )

    for problem in data:
        problem_description_now = problem["problem-description"]
        subproblems = problem["subproblems"]
        problemid = problem["problem-id"]
        overall_turns = problem["overall-turns"]
        tag = select_tag(problem) if use_lora else None

        if use_lora:
            tag_counts[tag] += 1
        should_update = use_lora and (tag_counts[tag] % args.update_interval == 0)

        if problemid in filename_list:
            continue

        history: List[str] = []
        turn_number = 1

        for subproblem in subproblems:
            user_input = get_input_single(subproblem, turn_number, overall_turns, problem_description_now, history)
            turn_number += 1

            # Accumulate context until final turn
            if turn_number != overall_turns + 1:
                history.append(user_input)
                continue

            messages = [
                {"role": "system", "content": "You are a Programming Expert. You always provide correct and reliable code solutions."},
                {"role": "user", "content": user_input},
            ]
            prompt_text = lora_mgr.format_prompt(messages)
            generated_list = []
            code_outputs = []
            for _ in range(max(1, args.num_samples)):
                g = lora_mgr.generate(messages, tag=tag, max_new_tokens=args.max_new_tokens)
                g = ensure_python_code_block(g)
                generated_list.append(g)
                code_outputs.append(extract_code(g))

            # Keep first candidate for backward compatibility, store all candidates separately
            subproblem.update({"original_output": generated_list[0]})
            subproblem.update({"generated": code_outputs[0]})
            subproblem.update({"generated_candidates": code_outputs})
            subproblem.update({"prompt": user_input})

            target_code = extract_target_code(subproblem, problem=problem)
            if should_update and target_code:
                adapter_dir = lora_mgr.finetune(tag=tag, prompt_text=prompt_text, target_text=target_code)
                if adapter_dir:
                    print(f"[LoRA] Updated adapter for tag='{tag}' at {adapter_dir}")
            elif should_update and not target_code:
                print(f"[LoRA] No reference code found; skipping update for tag='{tag}' (problem-id={problemid})")

        with open(f"{args.output_dir}/{problemid}.json", "w") as f:
            json.dump(problem, f, ensure_ascii=False, indent=4)

        print(f"Finished: saved to {args.output_dir}/{problemid}.json")
        print(f"[INFER] problem={problemid} tag={tag} update={'yes' if should_update else 'no'} adapter_active={adapter_dir if should_update and target_code else ('current' if use_lora else 'base')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-turn code generation with per-tag LoRA adaptation")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--adapter_root", type=str, default="output/adapters", help="Directory to store LoRA adapters")
    parser.add_argument("--device", type=str, default=None, help="Device for inference model (e.g., cuda:0)")
    parser.add_argument("--train_device", type=str, default=None, help="Device for LoRA fine-tune (defaults to inference device)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for LoRA updates")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="all-linear", help="LoRA target modules (e.g., all-linear or comma-separated list)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--train_max_length", type=int, default=2048, help="Truncate prompt+target tokens for LoRA training")
    parser.add_argument("--update_interval", type=int, default=8, help="Apply a LoRA update every N problems per tag")
    parser.add_argument("--train_lr", type=float, default=None, help="Override learning rate for LoRA updates")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of generations per subproblem (for pass@k)")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for model loading (e.g., auto, balanced, balanced_low_0)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit (bitsandbytes)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (bitsandbytes, NF4)")
    parser.add_argument("--bnb_compute_dtype", type=str, default="bfloat16", help="bitsandbytes compute dtype: bfloat16 or float16")
    args = parser.parse_args()
    main(args)
