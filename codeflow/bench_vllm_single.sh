#!/usr/bin/env bash
set -euo pipefail

# Benchmark: single-turn pipeline using vLLM (no LoRA) on a lowest-ELO subset.
# Reuses the same combine/harness/stats steps as bench_dp_single.sh but runs
# inference via codeflow/run/single_turn/inference_local.py (vLLM).
#
print_help() {
  cat <<'EOF'
Usage: bash bench_vllm_single.sh [options]
  --model_path PATH          HF id or local path (default: Qwen/Qwen2.5-Coder-7B-Instruct)
  --input_file PATH          Input JSONL (default: data/codeflowbench_full.jsonl)
  --tensor_parallel_size N   vLLM tensor parallel size (default: 1)
  --max_model_len N          vLLM max model len (default: 4096)
  --max_new_tokens N         Max new tokens to generate (default: 512)
  --temperature T            Sampling temperature (default: 0.6)
  --top_p P                  Top-p nucleus sampling (default: 0.9)
  --dtype DTYPE              vLLM dtype (auto, float16, bfloat16, float32; default: auto)
  --gpu_memory_utilization F Fraction of GPU memory to use (default: 0.8)
  --max_problems N           Subset size (default: 50)
  -h, --help                 Show this help
EOF
}

MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
INPUT_FILE="data/codeflowbench_full.jsonl"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
MAX_NEW_TOKENS=512
TEMPERATURE=0.6
TOP_P=0.9
DTYPE="auto"
GPU_MEM_UTIL=0.8
MAX_PROBLEMS=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --input_file) INPUT_FILE="$2"; shift 2 ;;
    --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --gpu_memory_utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
    --max_problems) MAX_PROBLEMS="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_NAME="${MODEL_BASENAME}-vllm-base"

OUT_ROOT="output/bench_vllm_single"
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
        tags = obj.get("tags") or []
        # Skip "special" tagged problems
        if any(isinstance(t, str) and t.startswith("*special") for t in tags):
            continue
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

echo "==> Single-turn inference (vLLM base), tp_size=$TENSOR_PARALLEL_SIZE dtype=$DTYPE"
python run/single_turn/inference_local.py \
  --model_path "$MODEL_PATH" \
  --input_file "$SUBSET_FILE" \
  --output_dir "$OUT_TEMP" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --max_model_len "$MAX_MODEL_LEN" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --dtype "$DTYPE" \
  --gpu_memory_utilization "$GPU_MEM_UTIL"

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
for k, v in data["summary"].items():
    if k == "overall":
        continue
    print(f"{k}: {v}")
PY
