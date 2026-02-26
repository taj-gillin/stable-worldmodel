#!/bin/bash
#SBATCH --job-name=pinpad_siglip2
#SBATCH --output=logs/pinpad_heatmap_siglip2_%j.out
#SBATCH --error=logs/pinpad_heatmap_siglip2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --partition=gpu

# ==========================================
# PinPadImage SigLIP 2 Heatmap Visualization
# ==========================================

echo "=========================================="
echo "PinPadImage Heatmap (SigLIP 2)"
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
mkdir -p results/pinpad_heatmap_siglip2

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

CONFIG="scripts/visualization/configs/pinpad_heatmap_siglip2.yaml"

echo "Config: $CONFIG"
echo ""

python scripts/visualization/pinpad_heatmap.py \
    --config "$CONFIG" \
    --output results/pinpad_heatmap_siglip2

EXIT_CODE=$?
echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
