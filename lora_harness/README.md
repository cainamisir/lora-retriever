# LoRA Harness

Benchmark-agnostic loop for running LoRA-based experiments one example at a
time. The harness only expects an `ExampleContext` stream that knows how to
evaluate predictions. Everything else (retrieval, inference, fine-tuning) stays
in user-supplied callables so you can plug in any serving stack, including a
vanilla Hugging Face `transformers` pipeline.

```python
from functools import partial
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_harness import LoRAHandle, LoRAHarness
from lora_harness.adapters import MemoryBenchExampleSource

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def retrieve_fn(example, harness):
    return harness.store.get_last(example.chat_id)

def inference_fn(example, lora_handle, harness):
    model = base_model
    if lora_handle and lora_handle.path:
        model = PeftModel.from_pretrained(base_model, lora_handle.path)
    prompt = tokenizer.apply_chat_template(
        example.messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def finetune_fn(example, lora_handle, response, metrics, harness):
    if metrics.get("accuracy", 1.0) < 1.0:
        new_path = train_or_update_adapter(example, response)
        return LoRAHandle(identifier=new_path, path=new_path)
    return None

example_source = partial(
    MemoryBenchExampleSource,
    dataset_type="single",
    name="DialSim-friends",
    limit_per_dataset=10,
)

harness = LoRAHarness(
    example_source=example_source,
    retrieve_lora_fn=retrieve_fn,
    inference_fn=inference_fn,
    finetune_fn=finetune_fn,
)

for result in harness.run():
    print(result.example.test_idx, result.metrics)
```

To support another benchmark, create a small adapter that yields
`ExampleContext` objects (see `lora_harness/adapters/memorybench.py`) and wire
it into the same harness. The rest of your loop stays identical.***
