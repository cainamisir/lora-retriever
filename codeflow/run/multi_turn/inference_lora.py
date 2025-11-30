import os
import sys
import argparse
import json
from typing import List
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lora_manager import LoRAPerTagManager, extract_target_code
from src.utils import (
    get_filenames_without_extension,
    extract_code,
    get_input,
    ensure_python_code_block,
    ensure_python_code_block_main,
    clean_code_block,
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
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.input_file)
    # Process problems grouped by primary tag to keep adapter updates coherent.
    data = sorted(data, key=lambda p: (select_tag(p), p.get("problem-id", "")))
    tag_counts = defaultdict(int)
    log_lines = []
    filename_list = get_filenames_without_extension(args.output_dir)

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
    )

    for problem in data:
        problem_description_now = problem["problem-description"]
        subproblems = problem["subproblems"]
        problemid = problem["problem-id"]
        print(f"Solving Problem {problemid}...")
        overall_turns = problem["overall-turns"]
        tag = select_tag(problem)

        # Only update the adapter every N problems per tag.
        tag_counts[tag] += 1
        should_update = args.update_interval > 0 and (tag_counts[tag] % args.update_interval == 0)

        if problemid in filename_list:
            continue

        turn_number = 1
        history: List[str] = []

        for subproblem in subproblems:
            input_text = get_input(subproblem, turn_number, overall_turns, problem_description_now, history)
            turn_number += 1

            messages = [
                {"role": "system", "content": "You are a Programming Expert. You always provide correct and reliable code solutions."},
                {"role": "user", "content": input_text},
            ]
            prompt_text = lora_mgr.format_prompt(messages)

            generated = lora_mgr.generate(messages, tag=tag, max_new_tokens=args.max_new_tokens)
            if turn_number == overall_turns + 1:
                generated = ensure_python_code_block_main(generated, subproblem)
            else:
                generated = ensure_python_code_block(generated)

            subproblem.update({"original_output": generated})
            code_output = extract_code(generated)
            code_output = clean_code_block(code_output)
            subproblem.update({"prompt": input_text})
            subproblem.update({"generated": code_output})

            target_code = extract_target_code(subproblem, problem=problem)
            if should_update and target_code:
                adapter_dir = lora_mgr.finetune(tag=tag, prompt_text=prompt_text, target_text=target_code)
                if adapter_dir:
                    print(f"[LoRA] Updated adapter for tag='{tag}' at {adapter_dir}")
            elif should_update and not target_code:
                print(f"[LoRA] No reference code found; skipping update for tag='{tag}' (problem-id={problemid})")

            history.append(code_output)

        log_entry = {
            "problem_id": problemid,
            "tag": tag,
            "should_update": should_update,
            "adapter_dir": adapter_dir if should_update and target_code else None,
        }
        log_lines.append(log_entry)
        print(f"[INFER] problem={problemid} tag={tag} update={'yes' if should_update else 'no'} adapter={log_entry['adapter_dir'] or 'none'}")

        output_path = os.path.join(args.output_dir, f"{problemid}.json")
        with open(output_path, "w") as f:
            json.dump(problem, f, ensure_ascii=False, indent=4)

        print(f"Processing completed, results saved to {output_path}")

    # Write inference log
    log_file = os.path.join(args.output_dir, "inference_log.jsonl")
    with open(log_file, "w", encoding="utf-8") as lf:
        for line in log_lines:
            lf.write(json.dumps(line) + "\n")
    print(f"[INFER] log written to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn code generation with per-tag LoRA adaptation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output")
    parser.add_argument("--adapter_root", type=str, default="output/adapters", help="Directory to store LoRA adapters")
    parser.add_argument("--device", type=str, default=None, help="Device for inference model (e.g., cuda:0)")
    parser.add_argument("--train_device", type=str, default=None, help="Device for LoRA fine-tune (defaults to inference device)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for LoRA updates")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--train_max_length", type=int, default=2048, help="Truncate prompt+target tokens for LoRA training")
    parser.add_argument("--update_interval", type=int, default=8, help="Apply a LoRA update every N problems per tag")
    parser.add_argument("--train_lr", type=float, default=None, help="Override learning rate for LoRA updates")

    args = parser.parse_args()
    main(args)
