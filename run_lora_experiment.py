#!/usr/bin/env python
"""
Example driver that runs the general-purpose LoRA harness on MemoryBench using
plain Hugging Face transformers + PEFT.

Usage:
    python run_lora_experiment.py \
        --dataset_type single \
        --set_name DialSim-friends \
        --model_name_or_path qwen/Qwen2-7B-Instruct \
        --limit_per_dataset 2
"""

import argparse
import contextlib
import os
import shutil
import uuid
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_harness import LoRAHandle, LoRAHarness
from lora_harness.adapters import MemoryBenchExampleSource

TARGET_TEXT_KEYS = [
    "golden_answer",
    "reference_answer",
    "reference",
    "target",
    "answer",
    "output",
    "expected_response",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-example LoRA harness demo.")
    parser.add_argument("--dataset_type", choices=["single", "domain", "task"], required=True)
    parser.add_argument("--set_name", required=True, help="Dataset/domain/task identifier.")
    parser.add_argument("--model_name_or_path", required=True, help="Base HF model to load.")
    parser.add_argument("--model_cache_dir", default="models", help="Local directory for Hugging Face cache.")
    parser.add_argument("--output_dir", default="lora_outputs", help="Where to save adapters/results.")
    parser.add_argument("--limit_per_dataset", type=int, default=5, help="Number of samples to process per dataset.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation length during inference.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="LoRA fine-tune learning rate.")
    parser.add_argument("--metric_threshold", type=float, default=0.99, help="Re-train adapter when metric falls below this.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None, help="Override torch device (e.g., cuda:0).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_cache_dir = os.path.abspath(args.model_cache_dir)
    os.makedirs(model_cache_dir, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", model_cache_dir)
    os.environ.setdefault("HF_HOME", model_cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(model_cache_dir, "datasets"))

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model {args.model_name_or_path} to {device} for inference...")
    base_inference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)
    base_inference_model.eval()

    base_lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=None,
        task_type="CAUSAL_LM",
    )

    def clone_lora_config():
        return LoraConfig(**base_lora_config.to_dict())

    base_inference_model = get_peft_model(base_inference_model, clone_lora_config())
    base_inference_model.to(device)
    base_inference_model.eval()
    base_inference_model.disable_adapter()
    for param in base_inference_model.parameters():
        param.requires_grad = False

    def format_prompt(messages):
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        formatted = []
        for msg in messages:
            formatted.append(f"{msg['role']}: {msg['content']}")
        formatted.append("assistant:")
        return "\n".join(formatted)

    def toggle_adapter_gradients(adapter_name: str, requires_grad: bool):
        for name, param in base_inference_model.named_parameters():
            if f".{adapter_name}" in name and "lora_" in name:
                param.requires_grad = requires_grad

    def ensure_adapter_loaded(handle: LoRAHandle, trainable: bool = False) -> str:
        adapter_name = handle.identifier
        if not adapter_name:
            if handle.path:
                adapter_name = os.path.basename(os.path.normpath(handle.path))
            else:
                adapter_name = f"adapter-{len(base_inference_model.peft_config)}"
        adapter_exists = adapter_name in base_inference_model.peft_config
        adapter_files_ready = False
        if handle.path:
            adapter_files_ready = os.path.exists(os.path.join(handle.path, "adapter_config.json"))

        if not adapter_exists:
            if adapter_files_ready:
                print(f"Loading adapter {adapter_name} from {handle.path}")
                base_inference_model.load_adapter(
                    handle.path,
                    adapter_name=adapter_name,
                    is_trainable=trainable,
                )
            else:
                print(f"Creating new adapter {adapter_name}")
                base_inference_model.add_adapter(adapter_name, clone_lora_config())
        if trainable:
            base_inference_model.set_adapter(adapter_name)
            toggle_adapter_gradients(adapter_name, True)
            base_inference_model.train()
        else:
            toggle_adapter_gradients(adapter_name, False)
            base_inference_model.set_adapter(adapter_name)
            base_inference_model.eval()
        return adapter_name

    def retrieve_lora(example, harness):
        handle = harness.store.get_last(example.chat_id)
        if handle:
            return handle
        # fallback: best adapter seen for this dataset
        best = None
        best_score = -1.0
        for record in harness.store.records:
            if record["dataset"] != example.dataset_name:
                continue
            metrics = record.get("metrics", {})
            for value in metrics.values():
                if isinstance(value, (int, float)) and value > best_score and record["lora"] is not None:
                    best_score = value
                    best = record["lora"]
        return best

    def inference_fn(example, lora_handle, harness):
        prompt = format_prompt(example.messages)
        model = base_inference_model
        if lora_handle and lora_handle.path:
            adapter_name = ensure_adapter_loaded(lora_handle, trainable=False)
            model.set_adapter(adapter_name)
        else:
            model.disable_adapter()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        except Exception as e:
            print(f"Inference failed with adapter; disabling adapter and retrying. Error: {e}")
            model.disable_adapter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text

    def should_finetune(metrics: dict) -> bool:
        if not metrics:
            return False
        for value in metrics.values():
            if value is None:
                continue
            if isinstance(value, (int, float)) and value < args.metric_threshold:
                return True
        return False

    def extract_target_text(example) -> Optional[str]:
        for key in TARGET_TEXT_KEYS:
            value = example.info.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def finetune_fn(example, lora_handle, response, metrics, harness):
        if not should_finetune(metrics):
            return None
        target_text = extract_target_text(example)
        if target_text is None:
            print(f"No reference text for test_idx={example.test_idx}, skipping fine-tune.")
            return None

        if lora_handle is None:
            adapter_id = f"{example.dataset_name}-{example.test_idx}-{uuid.uuid4().hex[:8]}"
            adapter_dir = os.path.join(args.output_dir, "adapters", adapter_id)
            os.makedirs(adapter_dir, exist_ok=True)
            lora_handle = LoRAHandle(identifier=adapter_id, path=adapter_dir)
        else:
            if lora_handle.path is None:
                adapter_dir = os.path.join(
                    args.output_dir, "adapters", lora_handle.identifier or uuid.uuid4().hex[:8]
                )
                os.makedirs(adapter_dir, exist_ok=True)
                lora_handle.path = adapter_dir
            adapter_dir = lora_handle.path

        adapter_name = ensure_adapter_loaded(lora_handle, trainable=False)
        print(f"Fine-tuning adapter {adapter_name} at {adapter_dir}")

        # Train a fresh copy on CPU to avoid GPU instability
        train_device = torch.device("cpu")
        train_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,
            device_map={"": train_device},
        )
        train_model = get_peft_model(train_model, clone_lora_config())
        train_model.add_adapter(adapter_name, clone_lora_config())
        train_model.set_adapter(adapter_name)
        train_model.train()

        prompt_text = format_prompt(example.messages)
        if not target_text.endswith(tokenizer.eos_token or ""):
            target_text = target_text + tokenizer.eos_token
        full_text = prompt_text + target_text
        inputs = tokenizer(full_text, return_tensors="pt")
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(train_device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        prompt_len = prompt_ids["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        trainable_params = [p for p in train_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        outputs = train_model(**inputs, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            print("Non-finite loss detected, skipping this fine-tune step.")
            optimizer.zero_grad(set_to_none=True)
            del train_model, optimizer, inputs, labels, outputs, loss
            return lora_handle
        loss.backward()
        grads_finite = True
        for p in trainable_params:
            if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                grads_finite = False
                break
        if not grads_finite:
            print("Non-finite gradients detected, skipping this fine-tune step.")
            optimizer.zero_grad(set_to_none=True)
            del train_model, optimizer, inputs, labels, outputs, loss
            return lora_handle
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        weights_finite = True
        for name, param in train_model.named_parameters():
            if adapter_name in name and "lora_" in name:
                if not torch.all(torch.isfinite(param)):
                    weights_finite = False
                    break
        if not weights_finite:
            print(f"Non-finite weights detected for adapter {adapter_name}, discarding this update.")
            if adapter_dir and os.path.exists(adapter_dir):
                shutil.rmtree(adapter_dir, ignore_errors=True)
            del train_model, optimizer, inputs, labels, outputs, loss
            return None

        train_model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        del train_model, optimizer, inputs, labels, outputs, loss
        torch.cuda.empty_cache()

        # Load the freshly trained adapter into the inference model
        ensure_adapter_loaded(lora_handle, trainable=False)

        return lora_handle

    example_source = MemoryBenchExampleSource(
        dataset_type=args.dataset_type,
        name=args.set_name,
        limit_per_dataset=args.limit_per_dataset,
    )

    harness = LoRAHarness(
        example_source=example_source,
        retrieve_lora_fn=retrieve_lora,
        inference_fn=inference_fn,
        finetune_fn=finetune_fn,
    )

    print("Starting harness run...")
    for idx, result in enumerate(harness.run(), start=1):
        print(
            f"[{idx}] dataset={result.example.dataset_name} "
            f"test_idx={result.example.test_idx} metrics={result.metrics} "
            f"retrieved={result.retrieved_lora.identifier if result.retrieved_lora else 'none'} "
            f"updated={result.updated_lora.identifier if result.updated_lora else 'none'}"
        )


if __name__ == "__main__":
    main()
