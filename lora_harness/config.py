# config.py

TASK_TYPES = [
    "Django",
    "Astropy",
    "Sympy",
    "Pytest",
    "Matplotlib",
    "Scikit",
    "Sphinx",
    "Other"
]

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_LEN = 2048

LORA_PARAMS = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}
