#!/usr/bin/env bash
set -euo pipefail

# Benchmark: single-turn pipeline with the base model (no LoRA) on a lowest-ELO subset.
# Mirrors bench_dp_single.sh but skips all LoRA updates/adapters.
#
# Args:
#   $1 model_path           (HF id or local path; default: Qwen/Qwen2.5-Coder-7B-Instruct)
#   $2 input_file           (default: data/codeflowbench_full.jsonl)
#   $3 device               (default: cuda:0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
INPUT_FILE="${2:-data/codeflowbench_full.jsonl}"
DEVICE="${3:-cuda:0}"

MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_NAME="${MODEL_BASENAME}-base"

MAX_PROBLEMS=20
OUT_ROOT="output/bench_base_single"
mkdir -p "$OUT_ROOT"

SUBSET_FILE="$OUT_ROOT/subset.jsonl"

# Clean old outputs for this run
rm -rf "$OUT_ROOT/base_temp" "$OUT_ROOT/${MODEL_NAME}_harness_temp"
rm -f "$OUT_ROOT/${MODEL_NAME}_stats.json" \
      "output/inference/${MODEL_NAME}_single_turn.json" \
      "output/harness/${MODEL_NAME}_single_turn.json"

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

OUT_TEMP="$OUT_ROOT/base_temp"

echo "==> Single-turn inference (base model, no LoRA)"
python run/single_turn/inference_local.py \
  --model_path "$MODEL_PATH" \
  --input_file "$SUBSET_FILE" \
  --output_dir "$OUT_TEMP"

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
python codeflow/run/single_turn/stats_report.py \
  --input "output/harness/${MODEL_NAME}_single_turn.json" \
  --output "$OUT_ROOT/${MODEL_NAME}_stats.json"

echo "==> Done. Stats at $OUT_ROOT/${MODEL_NAME}_stats.json"
python - <<'PY' "$OUT_ROOT/${MODEL_NAME}_stats.json"
import json, sys
stats_path = sys.argv[1]
data = json.load(open(stats_path))
overall = data["summary"]["overall"]
print(f"Overall all_pass1={overall.get('all_pass1')} over {overall.get('all_number')}")
for k, v in data["summary"].items():
    if k == "overall":
        continue
    print(f"{k}: {v}")
PY
