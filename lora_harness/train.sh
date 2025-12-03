#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH -t 0-08:00
#SBATCH -p seas_gpu
#SBATCH --constraint=a100|h100
#SBATCH --mem-per-cpu=7500
#SBATCH -o lora_swe_bench_%j.out
#SBATCH -e lora_swe_bench_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=aaditsaluja@college.harvard.edu

export PATH="$HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="/n/netscratch/idreos_lab/Everyone/share/aadit/lora"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

ENV_NAME="rag-retrieval"

if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo ">>> Creating micromamba environment '${ENV_NAME}'..."
  micromamba create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
else
  echo ">>> Micromamba environment '${ENV_NAME}' already exists. Skipping creation."
fi

eval "$(micromamba shell hook -s bash)"
micromamba activate "${ENV_NAME}"

# You may already have these installed; keep or comment out as needed.
# pip install --upgrade pip
# pip install datasets transformers peft accelerate bitsandbytes
# pip install swebench

echo ">>> Using Python: $(python --version)"
echo ">>> Env: ${ENV_NAME}"

# Paths
RUN_ID="${SLURM_JOB_ID}"
DATA_ROOT="/n/netscratch/idreos_lab/Everyone/share/aadit/lora/test_${RUN_ID}"
LOG_DIR="$DATA_ROOT/logs"
SAVE_DIR="$DATA_ROOT/adapters"
EVAL_RESULTS="$DATA_ROOT/eval_results.json"
PRED_BASE="$DATA_ROOT/predictions_base.jsonl"
PRED_LORA="$DATA_ROOT/predictions_lora.jsonl"
SWE_LOG_DIR="$DATA_ROOT/swebench_logs"

mkdir -p "$LOG_DIR" "$SAVE_DIR" "$SWE_LOG_DIR"

echo ">>> Running training + prediction generation..."
python core3.py \
  --data_root "${DATA_ROOT}" \
  --num_samples 200 \
  --max_new_tokens 1024

echo ">>> core3.py finished. eval_results.json:"
cat "$EVAL_RESULTS" || echo "No eval_results.json found."

echo ">>> Running SWE-bench harness on BASE predictions..."
python -m swebench.harness.run_evaluation \
  -d "princeton-nlp/SWE-bench_Lite" \
  -s "test[:100]" \
  -p "${PRED_BASE}" \
  --max_workers 4 \
  --cache_level "env" \
  --clean False \
  -id "qwen2_5_coder_lora"


echo ">>> Running SWE-bench harness on LoRA predictions..."
python -m swebench.harness.run_evaluation \
  -d "princeton-nlp/SWE-bench_Lite" \
  -s "test[:100]" \
  -p "${PRED_LORA}" \
  --max_workers 4 \
  --cache_level "env" \
  --clean False \
  -id "qwen2_5_coder_lora"

echo ">>> All done."
