#!/bin/bash
#SBATCH --job-name=pinpad_heatmap
#SBATCH --output=logs/pinpad_heatmap_%j.out
#SBATCH --error=logs/pinpad_heatmap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# ==========================================
# PinPadImage CLIP/SigLIP Heatmap Visualization
# ==========================================
# PinPadImage: cat_background.jpg with red dot agent.
# For each (x,y) position, generates state, computes embedding,
# then L2 distance to goal text prompts. Produces heatmaps.

echo "=========================================="
echo "PinPadImage Heatmap Visualization"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Project root: use directory from which sbatch was run (SLURM copies script to spool,
# so BASH_SOURCE would point to spool and give wrong path)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_DIR" || { echo "ERROR: could not cd to $PROJECT_DIR"; exit 1; }

mkdir -p logs
mkdir -p results/pinpad_heatmap

# Activate environment if venv exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Config (override via env vars)
# Use CONFIG to switch models: pinpad_heatmap.yaml (CLIP) or pinpad_heatmap_siglip2.yaml (SigLIP 2)
CONFIG="${CONFIG:-scripts/visualization/configs/pinpad_heatmap.yaml}"
# Output dir is read from config; override with OUTPUT_DIR if needed
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "Config:"
echo "  Project: $PROJECT_DIR"
echo "  Config: $CONFIG"
echo ""

CMD="python scripts/visualization/pinpad_heatmap.py --config $CONFIG"
[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output $OUTPUT_DIR"
eval $CMD

EXIT_CODE=$?
echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
