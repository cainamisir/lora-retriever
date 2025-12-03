import os
import json
import argparse
from typing import List, Set

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Assuming config has these. If not, define them or import them.
from config import MODEL_NAME, LORA_PARAMS
# from classify_task import classify_task # <--- Removed, we don't need this anymore
from lora_adapters import create_model_with_adapters, set_active_adapter

# Total training sequence length will be ~2 * MAX_LEN (prompt + target)
MAX_LEN = 2048 

def get_adapter_key(instance_id: str) -> str:
    """
    Extracts a key from the instance_id to determine which LoRA adapter to use.
    SWE-bench format is usually: repo_owner__repo_name-issue_id
    
    Example: 'django__django-11001' -> returns 'django'
    Example: 'astropy__astropy-1234' -> returns 'astropy'
    """
    # Split by '__' to get the repo part
    if "__" in instance_id:
        repo_part = instance_id.split("__")[1]
        # Split by '-' to remove the issue number
        repo_name = repo_part.split("-")[0]
        return repo_name
    return "default"

def build_prompt(problem_statement: str) -> str:
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
    max_new_tokens: int,
    per_example_results: List[dict],
    predictions_path: str,
    device: torch.device,
) -> int:
    solved = 0
    model.eval()
    predictions = []

    print("Running LoRA Evaluation...")
    for idx, ex in enumerate(dataset):
        prompt = build_prompt(ex["problem_statement"])
        gold_patch = ex["patch"].strip()
        
        # Switch to the correct adapter for this repo
        adapter_key = get_adapter_key(ex["instance_id"])
        set_active_adapter(model, adapter_key)

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

        # Update results dict
        per_example_results[idx]["lora_pred"] = diff_pred
        per_example_results[idx]["solved_with_lora"] = is_solved
        per_example_results[idx]["adapter_used"] = adapter_key

        predictions.append(
            {
                "instance_id": ex["instance_id"],
                "model_patch": diff_pred,
                "model_name_or_path": f"{MODEL_NAME}-lora-{adapter_key}",
            }
        )

    with open(predictions_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    return solved

def eval_without_lora(
    model,
    tokenizer,
    dataset,
    max_new_tokens: int,
    per_example_results: List[dict],
    predictions_path: str,
    device: torch.device,
) -> int:
    solved = 0
    model.eval()
    predictions = []

    print("Running Base Model Evaluation (Adapters Disabled)...")
    
    # Context manager to ensure base weights are used
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

    with open(predictions_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    return solved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    data_root = args.data_root
    logs_dir = os.path.join(data_root, "logs")
    adapters_dir = os.path.join(data_root, "adapters")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(adapters_dir, exist_ok=True)

    solutions_log_path = os.path.join(logs_dir, "per_example_solutions.jsonl")
    predictions_base_path = os.path.join(data_root, "predictions_base.jsonl")
    predictions_lora_path = os.path.join(data_root, "predictions_lora.jsonl")

    # 1. Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=f"test[:{args.num_samples}]")

    # 2. Identify required adapters based on the dataset
    # This replaces the old "classification" step with deterministic extraction
    required_adapters: Set[str] = set()
    for ex in dataset:
        required_adapters.add(get_adapter_key(ex["instance_id"]))
    
    print(f"Detected {len(required_adapters)} required adapters: {required_adapters}")

    # 3. Load tokenizer and base model
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # IMPORTANT for generation stability
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    device = next(base_model.parameters()).device
    print(f"Using main device: {device}")

    # 4. Attach LoRA adapters (Create one adapter per repo found)
    print("Attaching LoRA adapters...")
    # Convert set to list for stable ordering or compatibility
    adapter_list = list(required_adapters)
    model = create_model_with_adapters(base_model, adapter_list, LORA_PARAMS)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Prepare results structure
    per_example_results = []
    for idx, ex in enumerate(dataset):
        per_example_results.append({
            "idx": idx,
            "instance_id": ex.get("instance_id"),
            "repo": ex.get("repo"),
            "problem_statement": ex["problem_statement"],
            "gold_patch": ex["patch"].strip(),
            "adapter_key": get_adapter_key(ex["instance_id"]), # Log which one we expect to use
            "lora_pred": None,
            "base_pred": None,
            "solved_with_lora": False,
            "solved_without_lora": False,
        })

    # 5. Training Loop
    print("Training LoRA adapters...")
    model.train()
    
    for idx, ex in enumerate(dataset):
        prompt = build_prompt(ex["problem_statement"])
        target = ex["patch"].strip()
        adapter_key = get_adapter_key(ex["instance_id"])

        batch = prepare_training_example(tokenizer, prompt, target)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Activate the specific adapter for this repo/language
        set_active_adapter(model, adapter_key)
        
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        
        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at step {idx}: {loss}. Skipping.")
            optimizer.zero_grad()
            continue

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if idx % 10 == 0:
            print(f"[Train] step {idx} ({adapter_key}) | loss = {loss.item():.4f}")

    print("Saving adapters...")
    model.save_pretrained(adapters_dir)

    # 6. Evaluate with LoRA
    solved_with_lora = eval_with_lora(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        per_example_results=per_example_results,
        predictions_path=predictions_lora_path,
        device=device,
    )

    # 7. Evaluate without LoRA
    solved_without_lora = eval_without_lora(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        per_example_results=per_example_results,
        predictions_path=predictions_base_path,
        device=device,
    )

    # 8. Save results
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

    with open(solutions_log_path, "w") as f:
        for rec in per_example_results:
            f.write(json.dumps(rec) + "\n")

    print("Done. Aggregate results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()