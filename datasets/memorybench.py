# This implements the dataloader for memorybench
import datasets
import ast
import json

def convert_str_to_obj(example):
    for col in example.keys():
        if col.startswith("dialog") or col.startswith("implicit_feedback") or col in ["input_chat_messages", "info"]:
            try:
                example[col] = ast.literal_eval(example[col])
            except (ValueError, SyntaxError):
                example[col] = json.loads(example[col])
    if "Locomo" in example["dataset_name"]:
        if example["info"]["category"] == 5:
            example["info"]["golden_answer"] = json.dumps(example["info"]["golden_answer"])
        else:
            example["info"]["golden_answer"] = str(example["info"]["golden_answer"])
    return example

dataset = datasets.load_dataset("THUIR/MemoryBench", "NFCats")
dataset = dataset.map(convert_str_to_obj)

print(json.dumps(dataset["train"][0], indent=2))
