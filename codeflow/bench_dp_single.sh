#!/usr/bin/env bash
set -euo pipefail

# Benchmark: single-turn pipeline on a DP subset (50 problems) with optional LoRA updates.
# Uses run/single_turn/inference_lora.py + harness + stats.
#
# Args:
#   $1 model_path             (HF id or local path)
#   $2 input_file             (default: data/codeflowbench_full.jsonl)
#   $3 device/train_device    (default: cuda:0)
#   $4 update_interval_lora   (default: 8)
#   $5 train_lr_lora          (default: 5e-6)
#   $6 lora_r                 (default: 8)
#   $7 lora_alpha             (default: 32)
#   $8 num_samples            (default: 1, set to 5 for pass@5)

MODEL_PATH="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
INPUT_FILE="${2:-data/codeflowbench_full.jsonl}"
DEVICE="${3:-cuda:0}"
UPDATE_INTERVAL="${4:-8}"
TRAIN_LR="${5:-5e-6}"
LORA_R="${6:-8}"
LORA_ALPHA="${7:-32}"
NUM_SAMPLES="${8:-1}"
TRAIN_DEVICE="$DEVICE"

MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_NAME="${MODEL_BASENAME}-lora"

MAX_PROBLEMS=50
OUT_ROOT="output/bench_dp_single"
ADAPTER_ROOT="$OUT_ROOT/adapters"
mkdir -p "$OUT_ROOT" "$ADAPTER_ROOT"
rm -rf "$ADAPTER_ROOT"/* "$OUT_ROOT/lora_temp" "$OUT_ROOT/${MODEL_NAME}_harness_temp"
rm -f "$OUT_ROOT/${MODEL_NAME}_stats.json" \
      "output/inference/${MODEL_NAME}_single_turn.json" \
      "output/harness/${MODEL_NAME}_single_turn.json"

SUBSET_FILE="$OUT_ROOT/dp_subset.jsonl"

echo "==> Building subset (lowest $MAX_PROBLEMS by rating/elo across all problems) from $INPUT_FILE"
python - <<'PY' "$INPUT_FILE" "$SUBSET_FILE" "$MAX_PROBLEMS"
import json, sys
src, dst, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
tagged = []
with open(src, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        rating = obj.get("rating")
        rating_val = float(rating) if rating is not None else float("inf")
        tagged.append((rating_val, obj))

# Sort by rating/ELO ascending (lowest first) before selecting the subset
tagged.sort(key=lambda pair: pair[0])
subset = [obj for _, obj in tagged[:limit]]

with open(dst, "w", encoding="utf-8") as w:
    for obj in subset:
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {len(subset)} problems to {dst}")
print("Selected problems (rating / problem-id / title):")
for obj in subset:
    rating = obj.get("rating", "N/A")
    pid = obj.get("problem-id") or obj.get("problem_id") or ""
    title = obj.get("title") or ""
    print(f"{rating}\t{pid}\t{title}")
PY

OUT_TEMP="$OUT_ROOT/lora_temp"

echo "==> Single-turn inference (LoRA), update_interval=$UPDATE_INTERVAL train_lr=$TRAIN_LR r=$LORA_R alpha=$LORA_ALPHA"
python run/single_turn/inference_lora.py \
  --model_path "$MODEL_PATH" \
  --input_file "$SUBSET_FILE" \
  --output_dir "$OUT_TEMP" \
  --adapter_root "$ADAPTER_ROOT" \
  --device "$DEVICE" \
  --train_device "$TRAIN_DEVICE" \
  --update_interval "$UPDATE_INTERVAL" \
  --train_lr "$TRAIN_LR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --train_max_length 4096 \
  --num_samples "$NUM_SAMPLES"

echo "==> Combine inference"
python run/single_turn/combined.py \
  --model_name "$MODEL_NAME" \
  --combined_dir "$OUT_TEMP"

echo "==> Harness"
python run/single_turn/harness.py \
  --model_name "$MODEL_NAME" \
  --input_path "output/inference/${MODEL_NAME}_single_turn.json" \
  --output_dir "$OUT_ROOT/${MODEL_NAME}_harness_temp"

echo "==> Combine harness"
python run/single_turn/combined.py \
  --model_name "$MODEL_NAME" \
  --combined_dir "$OUT_ROOT/${MODEL_NAME}_harness_temp" \
  --harness

echo "==> Stats"
python run/single_turn/stats_report.py \
  --input "output/harness/${MODEL_NAME}_single_turn.json" \
  --output "$OUT_ROOT/${MODEL_NAME}_stats.json"

echo "==> Done. Stats at $OUT_ROOT/${MODEL_NAME}_stats.json"
python - <<'PY' "$OUT_ROOT/${MODEL_NAME}_stats.json"
import json, sys
stats_path = sys.argv[1]
data = json.load(open(stats_path))
overall = data["summary"]["overall"]
print(f"Overall all_pass1={overall.get('all_pass1')} over {overall.get('all_number')}")
for k,v in data["summary"].items():
    if k == "overall": continue
    print(f"{k}: {v}")
PY
