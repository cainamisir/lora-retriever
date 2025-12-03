# lora_adapters.py
from peft import LoraConfig, get_peft_model, TaskType


def create_model_with_adapters(base_model, task_types, lora_params):
    """
    Wrap base_model with PEFT and add one LoRA adapter per task type.
    Returns a PeftModel.
    """
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
    )
    # This creates a PeftModel with a default adapter; we then add named adapters.
    model = get_peft_model(base_model, peft_config)

    # Add named adapters for each task type
    for name in task_types:
        if name not in model.peft_config:
            model.add_adapter(name, peft_config)

    return model


def set_active_adapter(model, adapter_name: str):
    """
    Activate a specific LoRA adapter by name.
    """
    model.set_adapter(adapter_name)
