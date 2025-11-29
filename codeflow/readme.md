<h1 align="center"> CodeFlowBench: A Multi-turn, Iterative

Benchmark for Complex Code Generation </h1>



<p align="center">
  <a href="https://huggingface.co/datasets/WaterWang-001/CodeFlowBench-2505">
    <img alt="Hugging Face Dataset" src="https://img.shields.io/badge/HuggingFace-CodeFlowBench-blue?logo=huggingface">
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2504.21751">  
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2504.21751-b31b1b?logo=arxiv">
  </a>
</p>

## 🔧 Installation

```
cd codeflowbench

conda create -n codeflowbench python=3.10
conda activate codeflowbench

pip install -r requirements.txt
```

## 📋 Preparation
For Models:
Please place your model inside the `models` folder, for example: `models/Llama-3.1-8B-Instruct`.

For Datasets:
You can use the following command to download our sampling dataset from Hugging Face.
```
wget -O data/codeflowbench_sample.json "https://huggingface.co/datasets/WaterWang-001/CodeFlowBench-2505/resolve/main/codeflowbench_sample.json"
```

## 🏃 Quick Start
You can directly use our provided Bash scripts to test the model, with only minor modifications needed (all scripts use `Llama-3.1-8B-Instruct` as an example).

* For the multi-turn coding test, use `test_multi_turn.sh`;
* For the single-turn coding test, use `test_single_turn.sh`.

### 🧩 Quick Start (LoRA per tag)
We now support training a small LoRA adapter per problem tag (e.g., `dp`, `greedy`) after each generated output. Use the new inference scripts:

```bash
# multi-turn with per-tag LoRA
python run/multi_turn/inference_lora.py \
  --model_path models/Llama-3.1-8B-Instruct \
  --input_file data/codeflowbench_sample.json \
  --output_dir output/Llama-3.1-8B-Instruct_multi_turn_lora \
  --adapter_root output/adapters \
  --device cuda:0 \
  --train_device cuda:0

# single-turn with per-tag LoRA
python run/single_turn/inference_lora.py \
  --model_path models/Llama-3.1-8B-Instruct \
  --input_file data/codeflowbench_sample.json \
  --output_dir output/Llama-3.1-8B-Instruct_single_turn_lora \
  --adapter_root output/adapters \
  --device cuda:0 \
  --train_device cuda:0
```

What it does:
- Chooses one adapter per primary tag on each problem (`tags[0]` or `default`).
- Generates with the tag adapter active; after each output, fine-tunes that adapter on the reference solution if available in the dataset record.
- Saves adapters under `output/adapters/<tag>/` and keeps the original JSON structure so the existing harness/stat scripts still work.

You can point `--input_file` to a JSON array or a JSONL file (e.g., `codeflowbench_full.jsonl` from the HF dataset).

## 📊 Output Statistics
The final test results will be saved in the `result` folder, with the filename format `{model_name}_{multi_turn/single_turn}.json`.


