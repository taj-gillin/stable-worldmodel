#!/bin/bash
#SBATCH --job-name=pinpad_corr
#SBATCH --output=logs/pinpad_correlation_%j.out
#SBATCH --error=logs/pinpad_correlation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# ==========================================
# PinPadImage: Correlation analysis (ground truth vs L2)
# ==========================================
# Requires heatmap results. Run pinpad_heatmap_siglip2.slurm first.
# Uses grid_embeddings if present; otherwise uses heatmap JSON.

echo "=========================================="
echo "PinPadImage Correlation Analysis"
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
mkdir -p results/pinpad_correlation

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

HEATMAP_DIR="results/pinpad_heatmap_siglip2"
EMBEDDINGS="${HEATMAP_DIR}/grid_embeddings_siglip2.pt"

# Prefer embeddings if available (enables full analysis + optimization)
if [ -f "$EMBEDDINGS" ]; then
    echo "Using embeddings: $EMBEDDINGS"
    python scripts/visualization/pinpad_correlation_analysis.py \
        --embeddings "$EMBEDDINGS" \
        --output results/pinpad_correlation \
        --model google/siglip2-so400m-patch14-384
else
    echo "Embeddings not found. Using heatmap JSON only."
    python scripts/visualization/pinpad_correlation_analysis.py \
        --heatmap-dir "$HEATMAP_DIR" \
        --output results/pinpad_correlation
fi

EXIT_CODE=$?
echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
