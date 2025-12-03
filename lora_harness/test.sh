#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-04:00
#SBATCH -p test
#SBATCH --mem-per-cpu=7500
#SBATCH -o lora_eval_podman_raw_%j.out
#SBATCH -e lora_eval_podman_raw_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=aaditsaluja@college.harvard.edu

# --- 1. FIX ENVIRONMENT SETUP ---
# Ensure we have access to user binaries (where micromamba usually lives)
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

# If you use a specific module for python/conda, load it here:
# module load python/3.10.9-fasrc01  <-- Uncomment if needed

# Explicitly source bashrc to get aliases and PATHs if standard setup failed
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# Debug check
if ! command -v micromamba &> /dev/null; then
    echo "ERROR: micromamba not found in PATH. Current PATH: $PATH"
    # Fallback: Try to find it in common locations
    if [ -f "$HOME/micromamba/bin/micromamba" ]; then
        export PATH="$HOME/micromamba/bin:$PATH"
    elif [ -f "$HOME/bin/micromamba" ]; then
        export PATH="$HOME/bin:$PATH"
    else
        echo "CRITICAL: Could not locate micromamba binary. Exiting."
        exit 1
    fi
fi

# Point Podman to the auth file created on the login node
export REGISTRY_AUTH_FILE="$HOME/.run/containers/auth.json"
# Fallback location if the above doesn't exist (check which one 'podman login' created)
if [ ! -f "$REGISTRY_AUTH_FILE" ]; then
    export REGISTRY_AUTH_FILE="$HOME/.config/containers/auth.json"
fi

# Activate Environment
ENV_NAME="rag-retrieval"
eval "$(micromamba shell hook -s bash)"
micromamba activate "${ENV_NAME}"

# Verify Python has the module
python -c "import swebench; print('SWE-bench found')" || { echo "Failed to import swebench"; exit 1; }


# --- 2. FIX PODMAN SOCKET ---
# Use a writable temp directory instead of /run/user
export XDG_RUNTIME_DIR="/tmp/${USER}_podman_run"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"  # Secure the directory

# Start Podman Service
echo ">>> Starting Podman socket in $XDG_RUNTIME_DIR..."
# Kill any stale socket process from previous failed jobs
pkill -u "$USER" -f "podman system service" || true

# Start background service
podman system service -t 0 "unix://$XDG_RUNTIME_DIR/podman.sock" &
PODMAN_PID=$!
sleep 5

# Point Docker SDK to our custom socket
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/podman.sock"

# Verify connection
echo ">>> Testing Container Engine..."
if python -c "import docker; print(docker.from_env().version())"; then
    echo ">>> Connection successful."
else
    echo ">>> Connection failed. Check logs."
    kill $PODMAN_PID
    exit 1
fi


# --- 3. RUN EVALUATION ---
DATA_ROOT="/n/netscratch/idreos_lab/Everyone/share/aadit/lora/test_48465031/"
PRED_BASE="$DATA_ROOT/predictions_base.jsonl"
PRED_LORA="$DATA_ROOT/predictions_lora.jsonl"

echo ">>> Env: ${ENV_NAME}"
echo ">>> Data Root: ${DATA_ROOT}"

if [ -f "$PRED_BASE" ]; then
    echo ">>> Running SWE-bench harness on BASE predictions..."
    python -m swebench.harness.run_evaluation \
      -d "princeton-nlp/SWE-bench_Lite" \
      -s "test[:200]" \
      -p "${PRED_BASE}" \
      --max_workers 4 \
      --cache_level "instance" \
      --clean True \
      -id "qwen2_5_coder_base"
else
    echo ">>> Base predictions file not found: $PRED_BASE"
fi

if [ -f "$PRED_LORA" ]; then
    echo ">>> Running SWE-bench harness on LoRA predictions..."
    python -m swebench.harness.run_evaluation \
      -d "princeton-nlp/SWE-bench_Lite" \
      -s "test[:200]" \
      -p "${PRED_LORA}" \
      --max_workers 4 \
      --cache_level "instance" \
      --clean True \
      -id "qwen2_5_coder_lora_new"
else
    echo ">>> LoRA predictions file not found: $PRED_LORA"
fi

# cleanup
kill $PODMAN_PID
rm -rf "$XDG_RUNTIME_DIR"
echo ">>> All done."