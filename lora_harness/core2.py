import os
import json
import argparse
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME, MAX_LEN as CONFIG_MAX_LEN, TASK_TYPES, LORA_PARAMS
from classify_task import classify_task
from lora_adapters import create_model_with_adapters, set_active_adapter

# Total training sequence length will be ~2 * MAX_LEN (prompt + target)
MAX_LEN = 2048  # bump up if stable, shrink if you still see issues


def build_prompt(problem_statement: str) -> str:
    """
    Prompt tailored for SWE-bench-style program repair:
    - Asks for a single unified git diff patch
    - Requires the output to start with 'diff --git'
    - No explanations or extra text
    """
    return (
        "You are an automated program repair tool.\n\n"
        "Given the bug report below, output a single unified git diff patch that fixes the bug.\n"
        "ONLY OUTPUT THE DIFF.\n"
        "NO COMMENTS, NO ADDITIONAL INFORMATION, ONLY DIFF, NO EXTRA TEXT.\n"
        "YOU MUST START THE DIFF WITH 'diff --git'.\n\n"
        "Bug report:\n"
        f"{problem_statement}\n\n"
        "Diff:\n"
    )


def prepare_training_example(tokenizer, prompt: str, target: str):
    """
    Standard causal LM training:
    - Input = [prompt tokens][target tokens]
    - Labels = same as input, but prompt portion masked as -100
    """
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    )
    target_enc = tokenizer(
        target,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    )

    # Concatenate along sequence dimension
    input_ids = torch.cat([prompt_enc.input_ids, target_enc.input_ids], dim=1)
    attention_mask = torch.cat(
        [prompt_enc.attention_mask, target_enc.attention_mask],
        dim=1,
    )

    labels = input_ids.clone()
    # Mask prompt region so loss is only on the generated diff
    labels[:, : prompt_enc.input_ids.shape[1]] = -100

    return {
        "input_ids": input_ids.to(torch.long),
        "attention_mask": attention_mask.to(torch.long),
        "labels": labels.to(torch.long),
    }


def extract_diff(generated_text: str) -> str:
    """
    Extract a clean diff string starting from the first 'diff --git'.
    If none found, return the raw text (SWE-bench harness will likely fail it).
    """
    marker = "diff --git"
    if marker in generated_text:
        _, after = generated_text.split(marker, 1)
        diff = marker + after
    else:
        diff = generated_text

    return diff.strip()


def eval_with_lora(
    model,
    tokenizer,
    dataset,
    task_types: List[str],
    max_new_tokens: int,
    per_example_results: List[dict],
    predictions_path: str,
    device: torch.device,
) -> int:
    """
    Evaluate LoRA-enabled model.

    - Computes 'solved_with_lora' via substring match of gold patch (debug only).
    - Writes SWE-bench harness-compatible predictions JSONL to `predictions_path`.
    """
    solved = 0
    model.eval()

    predictions = []

    for idx, (ex, task_type) in enumerate(zip(dataset, task_types)):
        prompt = build_prompt(ex["problem_statement"])
        gold_patch = ex["patch"].strip()

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        set_active_adapter(model, task_type)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # force greedy
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
            )

        # Only decode tokens *after* the prompt
        generated_ids = out[0]
        prompt_len = enc["input_ids"].shape[1]
        new_token_ids = generated_ids[prompt_len:]

        pred_text = tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        diff_pred = extract_diff(pred_text)

        # Debug metric: does the gold patch appear as substring?
        is_solved = gold_patch in diff_pred
        if is_solved:
            solved += 1

        per_example_results[idx]["lora_pred"] = diff_pred
        per_example_results[idx]["solved_with_lora"] = is_solved

        predictions.append(
            {
                "instance_id": ex["instance_id"],
                "model_patch": diff_pred,
                "model_name_or_path": f"{MODEL_NAME}-lora",
            }
        )

    # Write SWE-bench predictions file
    with open(predictions_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    return solved


# def eval_without_lora(
#     model,
#     tokenizer,
#     dataset,
#     max_new_tokens: int,
#     per_example_results: List[dict],
#     predictions_path: str,
#     device: torch.device,
# ) -> int:
#     """
#     Evaluate base model behavior using the same PEFT model with adapters disabled.

#     - Writes SWE-bench harness-compatible predictions JSONL to `predictions_path`.
#     """
#     solved = 0
#     model.eval()
#     model.disable_adapter()  # disable all adapters so only base weights are used

#     predictions = []

#     for idx, ex in enumerate(dataset):
#         prompt = build_prompt(ex["problem_statement"])
#         gold_patch = ex["patch"].strip()

#         enc = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=MAX_LEN,
#         )
#         enc = {k: v.to(device) for k, v in enc.items()}

#         with torch.no_grad():
#             out = model.generate(
#                 **enc,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 num_beams=1,
#                 temperature=1.0,
#                 top_p=1.0,
#             )

#         generated_ids = out[0]
#         prompt_len = enc["input_ids"].shape[1]
#         new_token_ids = generated_ids[prompt_len:]

#         pred_text = tokenizer.decode(
#             new_token_ids,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )

#         diff_pred = extract_diff(pred_text)

#         is_solved = gold_patch in diff_pred
#         if is_solved:
#             solved += 1

#         per_example_results[idx]["base_pred"] = diff_pred
#         per_example_results[idx]["solved_without_lora"] = is_solved

#         predictions.append(
#             {
#                 "instance_id": ex["instance_id"],
#                 "model_patch": diff_pred,
#                 "model_name_or_path": f"{MODEL_NAME}-base",
#             }
#         )

#     # Write SWE-bench predictions file
#     with open(predictions_path, "w") as f:
#         for rec in predictions:
#             f.write(json.dumps(rec) + "\n")

#     return solved

def eval_without_lora(
    model,
    tokenizer,
    dataset,
    max_new_tokens: int,
    per_example_results: List[dict],
    predictions_path: str,
    device: torch.device,
) -> int:
    """
    Evaluate base model behavior using the same PEFT model with adapters disabled.
    """
    solved = 0
    model.eval()
    
    # REMOVED: model.disable_adapter() 

    predictions = []

    # Use the context manager here to force the base model weights
    with model.disable_adapter():
        for idx, ex in enumerate(dataset):
            prompt = build_prompt(ex["problem_statement"])
            gold_patch = ex["patch"].strip()

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                )

            generated_ids = out[0]
            prompt_len = enc["input_ids"].shape[1]
            new_token_ids = generated_ids[prompt_len:]

            pred_text = tokenizer.decode(
                new_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            diff_pred = extract_diff(pred_text)

            is_solved = gold_patch in diff_pred
            if is_solved:
                solved += 1

            per_example_results[idx]["base_pred"] = diff_pred
            per_example_results[idx]["solved_without_lora"] = is_solved

            predictions.append(
                {
                    "instance_id": ex["instance_id"],
                    "model_patch": diff_pred,
                    "model_name_or_path": f"{MODEL_NAME}-base",
                }
            )

    # Write SWE-bench predictions file
    with open(predictions_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    return solved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory for logs, adapters, cached data, etc.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of SWE-bench Lite problems to use.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    data_root = args.data_root
    logs_dir = os.path.join(data_root, "logs")
    adapters_dir = os.path.join(data_root, "adapters")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(adapters_dir, exist_ok=True)

    classification_log_path = os.path.join(logs_dir, "classification_log.jsonl")
    solutions_log_path = os.path.join(logs_dir, "per_example_solutions.jsonl")

    # SWE-bench predictions paths (for harness)
    predictions_base_path = os.path.join(data_root, "predictions_base.jsonl")
    predictions_lora_path = os.path.join(data_root, "predictions_lora.jsonl")

    # 1. Load dataset
    dataset = load_dataset(
        "princeton-nlp/SWE-bench_Lite",
        split=f"test[:{args.num_samples}]",
    )

    # 2. Load tokenizer and base instruct model
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # Multi-GPU fp16 model with automatic sharding
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16,   # non-deprecated arg
        device_map="auto",     # shard across available GPUs
    )

    # Disable KV cache during training to save memory / avoid weirdness
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    # Also force greedy decoding as default to avoid sampling kernel asserts
    if hasattr(base_model, "generation_config"):
        gen_cfg = base_model.generation_config
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        gen_cfg.temperature = 1.0
        gen_cfg.top_p = 1.0

    # Use the device of the first parameter as our "main" device for feeding inputs
    device = next(base_model.parameters()).device
    print(f"Using main device: {device}")

    # 3. Classify each example once using the same instruct model
    print(f"Classifying {len(dataset)} problems into {len(TASK_TYPES)} task types...")
    ex_task_types: List[str] = []
    base_model.eval()

    with open(classification_log_path, "w") as clf_log_f:
        for idx, ex in enumerate(dataset):
            task_type = classify_task(
                ex["problem_statement"],
                tokenizer=tokenizer,
                model=base_model,
            )
            ex_task_types.append(task_type)

            record = {
                "idx": idx,
                "instance_id": ex.get("instance_id", None),
                "repo": ex.get("repo", None),
                "task_type": task_type,
                "problem_statement": ex["problem_statement"],
            }
            clf_log_f.write(json.dumps(record) + "\n")

    # 4. Attach LoRA adapters to the same base model
    print("Attaching LoRA adapters...")
    model = create_model_with_adapters(base_model, TASK_TYPES, LORA_PARAMS)
    # IMPORTANT: do NOT call model.to(device) here with device_map="auto"
    model.train()

    # Slightly safer LR to reduce chance of divergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Prepare per-example results structure
    per_example_results: List[dict] = []
    for idx, (ex, task_type) in enumerate(zip(dataset, ex_task_types)):
        per_example_results.append(
            {
                "idx": idx,
                "instance_id": ex.get("instance_id", None),
                "repo": ex.get("repo", None),
                "problem_statement": ex["problem_statement"],
                "gold_patch": ex["patch"].strip(),
                "task_type": task_type,
                "lora_pred": None,
                "base_pred": None,
                "solved_with_lora": False,
                "solved_without_lora": False,
            }
        )

    # 5. Simple one-epoch training over the dataset (with NaN guard + grad clipping)
    print("Training LoRA adapters...")
    model.train()
    for step_idx, (ex, task_type) in enumerate(zip(dataset, ex_task_types)):
        prompt = build_prompt(ex["problem_statement"])
        target = ex["patch"].strip()

        batch = prepare_training_example(tokenizer, prompt, target)
        batch = {k: v.to(device) for k, v in batch.items()}

        set_active_adapter(model, task_type)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Guard against NaN / Inf loss
        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at step {step_idx}: {loss}. Skipping update.")
            optimizer.zero_grad()
            continue

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        if step_idx % 10 == 0:
            print(f"[Train] step {step_idx} | loss = {loss.item():.4f}")

    # Save full PEFT model (base + adapters)
    print("Saving adapters...")
    model.save_pretrained(adapters_dir)

    # 6. Evaluate with LoRA
    print("Evaluating with LoRA-enabled model...")
    solved_with_lora = eval_with_lora(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        task_types=ex_task_types,
        max_new_tokens=args.max_new_tokens,
        per_example_results=per_example_results,
        predictions_path=predictions_lora_path,
        device=device,
    )

    # 7. Evaluate without LoRA (same model, adapters disabled)
    print("Evaluating base behavior (adapters disabled)...")
    solved_without_lora = eval_without_lora(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        per_example_results=per_example_results,
        predictions_path=predictions_base_path,
        device=device,
    )

    # 8. Save aggregate results
    results = {
        "model_name": MODEL_NAME,
        "num_samples": len(dataset),
        "solved_with_lora": solved_with_lora,
        "solved_without_lora": solved_without_lora,
        "predictions_base_path": predictions_base_path,
        "predictions_lora_path": predictions_lora_path,
    }
    eval_path = os.path.join(data_root, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    # 9. Save per-example solutions log
    with open(solutions_log_path, "w") as f:
        for rec in per_example_results:
            f.write(json.dumps(rec) + "\n")

    print("Done. Aggregate results:")
    print(json.dumps(results, indent=2))
    print(f"Per-example solutions written to: {solutions_log_path}")
    print(f"Classification log written to: {classification_log_path}")
    print(f"Base predictions written to: {predictions_base_path}")
    print(f"LoRA predictions written to: {predictions_lora_path}")


if __name__ == "__main__":
    main()
