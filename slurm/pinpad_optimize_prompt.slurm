#!/bin/bash
#SBATCH --job-name=pinpad_opro
#SBATCH --output=logs/pinpad_optimize_prompt_%j.out
#SBATCH --error=logs/pinpad_optimize_prompt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# ==========================================
# PinPadImage: OPRO prompt optimization
# ==========================================
# Requires: grid_embeddings from pinpad_heatmap. Run heatmap first.
# Requires: OPENAI_API_KEY for GPT calls.

echo "=========================================="
echo "PinPadImage Prompt Optimization (OPRO)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_DIR" || { echo "ERROR: could not cd to $PROJECT_DIR"; exit 1; }

mkdir -p logs
mkdir -p results/pinpad_opro

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

EMBEDDINGS="results/pinpad_heatmap_siglip2/grid_embeddings_siglip2.pt"
GOAL_IDX="${GOAL_IDX:-0}"

if [ ! -f "$EMBEDDINGS" ]; then
    echo "ERROR: Embeddings not found: $EMBEDDINGS"
    echo "Run: sbatch slurm/pinpad_heatmap_siglip2.slurm"
    exit 1
fi

python scripts/visualization/pinpad_optimize_prompt.py \
    --embeddings "$EMBEDDINGS" \
    --goal-idx "$GOAL_IDX" \
    --output-dir results/pinpad_opro \
    --steps 10 \
    --batch-size 5

EXIT_CODE=$?
echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
