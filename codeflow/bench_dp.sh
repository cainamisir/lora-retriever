#!/usr/bin/env bash
set -euo pipefail

# Benchmark: compare base vs LoRA-updated model on a DP subset (50 problems).
# Uses existing inference/harness pipelines. Requires the dataset JSONL.
#
# Example:
#   bash bench_dp.sh Qwen/Qwen2.5-Coder-7B-Instruct data/codeflowbench_full.jsonl cuda:0 8 5e-6
#
# Args:
#   $1 model_path             (HF id or local path)
#   $2 input_file             (default: data/codeflowbench_full.jsonl)
#   $3 device/train_device    (default: cuda:0)
#   $4 update_interval_lora   (default: 8)
#   $5 train_lr_lora          (default: 5e-6)

MODEL_PATH="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
INPUT_FILE="${2:-data/codeflowbench_full.jsonl}"
DEVICE="${3:-cuda:0}"
UPDATE_INTERVAL="${4:-8}"
TRAIN_LR="${5:-5e-6}"
LORA_R="${6:-8}"
LORA_ALPHA="${7:-32}"
TRAIN_DEVICE="$DEVICE"

TAG_FILTER="dp"
MAX_PROBLEMS=50
OUT_ROOT="output/bench_dp"
ADAPTER_ROOT="$OUT_ROOT/adapters"
mkdir -p "$OUT_ROOT" "$ADAPTER_ROOT"
rm -rf "$ADAPTER_ROOT"/*

SUBSET_FILE="$OUT_ROOT/dp_subset.jsonl"

echo "==> Building DP subset (first $MAX_PROBLEMS matching tag '$TAG_FILTER') from $INPUT_FILE"
python - <<'PY' "$INPUT_FILE" "$SUBSET_FILE" "$TAG_FILTER" "$MAX_PROBLEMS"
import json, sys
src, dst, tag, limit = sys.argv[1], sys.argv[2], sys.argv[3].lower(), int(sys.argv[4])
out = []
with open(src, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        tags = [t.lower() for t in obj.get("tags", []) if isinstance(t, str)]
        if any(tag in t for t in tags):
            out.append(obj)
        if len(out) >= limit:
            break
with open(dst, "w", encoding="utf-8") as w:
    for obj in out:
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"Wrote {len(out)} problems to {dst}")
PY

BASE_NAME="$(basename "$MODEL_PATH")-base"
LORA_NAME="$(basename "$MODEL_PATH")-lora"

# Clean old outputs for this run
rm -rf "$OUT_ROOT/lora_temp" "$OUT_ROOT/${LORA_NAME}_harness_temp"
rm -f "$OUT_ROOT/${LORA_NAME}_stats.json" "output/inference/${LORA_NAME}_multi_turn.json" "output/harness/${LORA_NAME}_multi_turn.json"

run_pipeline () {
  local MODEL_NAME="$1"
  local OUT_TEMP="$2"
  local UPDATE_INT="$3"
  local TRAIN_LR_ARG="$4"

  echo "==> Inference ($MODEL_NAME), update_interval=$UPDATE_INT train_lr=$TRAIN_LR_ARG"
  python run/multi_turn/inference_lora.py \
    --model_path "$MODEL_PATH" \
    --input_file "$SUBSET_FILE" \
    --output_dir "$OUT_TEMP" \
    --adapter_root "$ADAPTER_ROOT" \
    --device "$DEVICE" \
    --train_device "$TRAIN_DEVICE" \
    --update_interval "$UPDATE_INT" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    ${TRAIN_LR_ARG:+--train_lr "$TRAIN_LR_ARG"} \
    --train_max_length 4096

  echo "==> Combine inference for $MODEL_NAME"
  python run/multi_turn/combined.py \
    --model_name "$MODEL_NAME" \
    --combined_dir "$OUT_TEMP"

  echo "==> Harness for $MODEL_NAME"
  python run/multi_turn/harness.py \
    --model_name "$MODEL_NAME" \
    --input_path "output/inference/${MODEL_NAME}_multi_turn.json" \
    --output_dir "$OUT_ROOT/${MODEL_NAME}_harness_temp"

  echo "==> Combine harness for $MODEL_NAME"
  python run/multi_turn/combined.py \
    --model_name "$MODEL_NAME" \
    --combined_dir "$OUT_ROOT/${MODEL_NAME}_harness_temp" \
    --harness

  echo "==> Stats for $MODEL_NAME"
  python run/multi_turn/stats_report.py \
    --input "output/harness/${MODEL_NAME}_multi_turn.json" \
    --output "$OUT_ROOT/${MODEL_NAME}_stats.json"
}

run_pipeline "$LORA_NAME" "$OUT_ROOT/lora_temp" "$UPDATE_INTERVAL" "$TRAIN_LR"
