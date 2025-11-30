import os
import shutil
import time
from typing import List, Optional, Union, Dict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
# Optional bitsandbytes config for quantized loading
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
# Keys we will scan to find a reference solution in the dataset record.
TARGET_CODE_KEYS: List[str] = [
    "reference_solution",
    "reference_code",
    "reference",
    "solution",
    "gold_code",
    "golden_answer",
    "answer",
    "target",
    "code",
]


class LoRAPerTagManager:
    """
    Minimal helper that keeps one LoRA adapter per tag (problem class).
    - The inference model is kept frozen with adapters toggled on/off.
    - A fresh training copy is spun up for each fine-tune to avoid
      destabilizing inference state.
    """

    def __init__(
        self,
        model_path: str,
        adapter_root: str,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-5,
        device: Optional[str] = None,
        train_device: Optional[str] = None,
        torch_dtype_infer: Optional[torch.dtype] = None,
        train_max_length: int = 2048,
        train_lr: Optional[float] = None,
        device_map: Optional[Union[str, Dict]] = None,
        target_modules: Optional[Union[str, List[str]]] = "all-linear",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_compute_dtype: str = "bfloat16",
    ):
        os.makedirs(adapter_root, exist_ok=True)
        self.model_path = model_path
        self.adapter_root = adapter_root
        self.learning_rate = train_lr if train_lr is not None else learning_rate
        self.train_max_length = train_max_length
        self.device_map = device_map

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_device = torch.device(train_device) if (train_device and not device_map) else self.device
        self.torch_dtype_infer = torch_dtype_infer

        self.base_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load frozen base inference model and wrap with an empty adapter set.
        if self.torch_dtype_infer is not None:
            dtype = self.torch_dtype_infer
        else:
            if self.device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        load_kwargs = {
            "dtype": dtype,
            "device_map": self.device_map,
        }
        if load_in_8bit or load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes not available; cannot load in quantized mode")
            compute_dtype = torch.bfloat16 if bnb_compute_dtype == "bfloat16" else torch.float16
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=compute_dtype if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else None,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                bnb_8bit_compute_dtype=compute_dtype if load_in_8bit else None,
            )
            load_kwargs.pop("dtype", None)
            load_kwargs["quantization_config"] = quant_cfg
            if self.device_map is None:
                load_kwargs["device_map"] = "auto"
        self.infer_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )
        # If we are not sharding, move to a single device
        if self.device_map is None and not getattr(self.infer_model, "is_loaded_in_8bit", False):
            self.infer_model = self.infer_model.to(self.device)
        self.infer_model = get_peft_model(self.infer_model, self._clone_lora_config())
        # Enable checkpointing to reduce activation memory during LoRA training
        if hasattr(self.infer_model, "gradient_checkpointing_enable"):
            self.infer_model.gradient_checkpointing_enable()
        if hasattr(self.infer_model, "config"):
            self.infer_model.config.use_cache = False
        if hasattr(self.infer_model, "enable_input_require_grads"):
            self.infer_model.enable_input_require_grads()
        self.infer_model.eval()
        self.infer_model.disable_adapter()
        for p in self.infer_model.parameters():
            p.requires_grad = False

        # Pick a primary device to place inputs on (first in map or explicit device)
        if hasattr(self.infer_model, "hf_device_map") and self.infer_model.hf_device_map:
            first_device = list(self.infer_model.hf_device_map.values())[0]
            self.primary_device = torch.device(first_device)
        else:
            self.primary_device = self.device

    def _clone_lora_config(self) -> LoraConfig:
        return LoraConfig(**self.base_lora_config.to_dict())

    def _toggle_adapter_gradients(self, adapter_name: str, requires_grad: bool) -> None:
        for name, param in self.infer_model.named_parameters():
            if f".{adapter_name}" in name and "lora_" in name:
                param.requires_grad = requires_grad

    def _adapter_dir(self, tag: str) -> str:
        safe_tag = tag.replace("/", "_")
        return os.path.join(self.adapter_root, safe_tag)

    def _adapter_weights_finite(self, adapter_name: str) -> bool:
        for name, param in self.infer_model.named_parameters():
            if adapter_name in name and "lora_" in name:
                if not torch.all(torch.isfinite(param)):
                    return False
        return True

    def format_prompt(self, messages: List[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted = []
        for msg in messages:
            formatted.append(f"{msg['role']}: {msg['content']}")
        formatted.append("assistant:")
        return "\n".join(formatted)

    def ensure_adapter_loaded(self, tag: str, trainable: bool = False) -> str:
        adapter_name = tag or "default"
        adapter_dir = self._adapter_dir(adapter_name)
        adapter_exists = adapter_name in self.infer_model.peft_config
        adapter_files_ready = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

        if not adapter_exists:
            if adapter_files_ready:
                self.infer_model.load_adapter(adapter_dir, adapter_name=adapter_name, is_trainable=trainable)
            else:
                self.infer_model.add_adapter(adapter_name, self._clone_lora_config())

        if trainable:
            self.infer_model.set_adapter(adapter_name)
            self._toggle_adapter_gradients(adapter_name, True)
            self.infer_model.train()
        else:
            self._toggle_adapter_gradients(adapter_name, False)
            self.infer_model.set_adapter(adapter_name)
            self.infer_model.eval()
        return adapter_dir

    def generate(self, messages: List[dict], tag: Optional[str], max_new_tokens: int = 512) -> str:
        # If tag is None, run pure base model (no adapters)
        if tag is None:
            self.infer_model.disable_adapter()
            adapter_dir = "none"
        else:
            tag = tag or "default"
            self.ensure_adapter_loaded(tag, trainable=False)
            adapter_dir = self._adapter_dir(tag)
            # If adapter weights are non-finite, disable and fall back to base.
            if not self._adapter_weights_finite(tag):
                print(f"[LoRA] Adapter '{tag}' has non-finite weights; disabling for generation.")
                self.infer_model.disable_adapter()
                tag = None
        prompt = self.format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.primary_device)
        # Enable cache for fast generation
        if hasattr(self.infer_model, "config"):
            self.infer_model.config.use_cache = True
        try:
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = self.infer_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            gen_time = time.perf_counter() - start
        except Exception as e:
            # If the adapter is corrupted, fall back to base model
            print(f"[LoRA] Generation failed with adapter '{tag}', disabling adapter and retrying. Error: {e}")
            self.infer_model.disable_adapter()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = self.infer_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            gen_time = time.perf_counter() - start
        print(f"[LoRA] Using adapter '{tag or 'base'}' (dir: {adapter_dir}) for generation")
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        if generated_ids.numel() == 0:
            generated_ids = output_ids[0]
        gen_tokens = generated_ids.numel()
        total_tokens = prompt_len + gen_tokens
        if gen_time > 0:
            tps = total_tokens / gen_time
            print(f"[LoRA] gen_tokens={gen_tokens} prompt_tokens={prompt_len} total_tokens={total_tokens} time={gen_time:.2f}s tokens_per_sec={tps:.2f}")
        # Disable cache before training steps resume
        if hasattr(self.infer_model, "config"):
            self.infer_model.config.use_cache = False
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def finetune(self, tag: str, prompt_text: str, target_text: Optional[str]) -> Optional[str]:
        """
        Fine-tune the active adapter on GPU (preferred). Reuses the inference
        model to avoid loading a second full copy of the base weights.
        """
        if not target_text:
            return None
        adapter_name = tag or "default"
        adapter_dir = self._adapter_dir(adapter_name)
        os.makedirs(adapter_dir, exist_ok=True)

        model = self.infer_model
        self.ensure_adapter_loaded(adapter_name, trainable=True)
        # Explicitly set requires_grad on LoRA params of the active adapter
        active_adapter = adapter_name
        for name, p in model.named_parameters():
            if "lora_" in name and f".{active_adapter}" in name:
                p.requires_grad = True
            else:
                if "lora_" in name:
                    p.requires_grad = False
        model.train()

        # Prepare supervised pair: prompt -> target code.
        if not target_text.endswith(self.tokenizer.eos_token or ""):
            target_text = target_text + self.tokenizer.eos_token
        full_text = prompt_text + target_text
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.train_max_length,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.train_max_length,
        )
        inputs = {k: v.to(self.primary_device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        prompt_len = prompt_ids["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        seq_len = inputs["input_ids"].shape[1]
        if self.primary_device.type == "cuda":
            try:
                mem_alloc = torch.cuda.memory_allocated(self.primary_device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(self.primary_device) / 1e9
                print(f"[LoRA] tag={adapter_name} seq_len={seq_len} mem_alloc={mem_alloc:.2f}GB mem_reserved={mem_reserved:.2f}GB")
            except Exception:
                print(f"[LoRA] tag={adapter_name} seq_len={seq_len}")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        try:
            if self.primary_device.type == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast("cuda", dtype=dtype):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    return adapter_dir
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                # Check grad finiteness before stepping
                grads_finite = all(
                    (p.grad is None) or torch.all(torch.isfinite(p.grad))
                    for p in trainable_params
                )
                if not grads_finite:
                    optimizer.zero_grad(set_to_none=True)
                    return adapter_dir
                optimizer.step()
            else:
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    return adapter_dir
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                grads_finite = all(
                    (p.grad is None) or torch.all(torch.isfinite(p.grad))
                    for p in trainable_params
                )
                if not grads_finite:
                    optimizer.zero_grad(set_to_none=True)
                    return adapter_dir
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            print(f"[LoRA] CUDA OOM during train step; skipping update for tag='{adapter_name}'")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return adapter_dir
        optimizer.zero_grad(set_to_none=True)

        if not self._adapter_weights_finite(adapter_name):
            print(f"[LoRA] Non-finite weights detected after update for tag='{adapter_name}', discarding adapter.")
            if os.path.exists(adapter_dir):
                shutil.rmtree(adapter_dir, ignore_errors=True)
            self.infer_model.disable_adapter()
            return None

        model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

        model.eval()
        self._toggle_adapter_gradients(adapter_name, False)
        model.set_adapter(adapter_name)
        if self.primary_device.type == "cuda":
            torch.cuda.empty_cache()
        return adapter_dir


def extract_target_code(subproblem: dict, problem: Optional[dict] = None) -> Optional[str]:
    """
    Try to pull a reference solution from common fields.
    Priority:
    - solutions: list of dicts, prefer type == "code", else first content string
    - direct string fields in TARGET_CODE_KEYS
    """
    solutions = subproblem.get("solutions")
    if solutions is None and problem is not None:
        solutions = problem.get("solutions")
    if isinstance(solutions, list):
        # prefer code-type solutions
        for sol in solutions:
            if isinstance(sol, dict) and sol.get("type") == "code":
                content = sol.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        # fallback: first string content
        for sol in solutions:
            if isinstance(sol, dict):
                content = sol.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    for key in TARGET_CODE_KEYS:
        value = subproblem.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None
