# dataset_loader.py
from datasets import load_dataset
from classify_task import classify_task
from config import TASK_TYPES, MAX_LEN
from torch.utils.data import DataLoader
import torch


def preprocess(example, tokenizer):
    task_type = classify_task(example["problem_statement"])
    prompt = f"Issue: {example['problem_statement']}\nProposed Solution:"
    target = example["patch"]

    input_enc = tokenizer(prompt, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    target_enc = tokenizer(target, truncation=True, max_length=MAX_LEN, return_tensors="pt")

    input_ids = torch.cat([input_enc.input_ids, target_enc.input_ids[:, 1:]], dim=1)
    attention_mask = torch.cat([input_enc.attention_mask, target_enc.attention_mask[:, 1:]], dim=1)
    labels = torch.cat([
        torch.full_like(input_enc.input_ids, -100),
        target_enc.input_ids[:, 1:]
    ], dim=1)

    return {
        "input_ids": input_ids.squeeze(0),
        "attention_mask": attention_mask.squeeze(0),
        "labels": labels.squeeze(0),
        "task_type": task_type
    }


def get_task_dataloader(tokenizer):
    raw_dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="train")
    processed = [preprocess(ex, tokenizer) for ex in raw_dataset]
    return DataLoader(processed, batch_size=1, shuffle=True)