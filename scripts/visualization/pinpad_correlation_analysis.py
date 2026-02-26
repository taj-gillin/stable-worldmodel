"""
PinPadImage: Correlation analysis between ground-truth distance and SigLIP/CLIP L2 distance.

For each goal (target point), we compute:
- Ground truth: Euclidean distance from each grid position to the goal point
- L2 distance: embedding distance between rendered image and goal text prompt

We graph the correlation and report Pearson/Spearman.

Usage:
    # Using pre-computed heatmap results (requires grid_embeddings from heatmap run):
    python scripts/visualization/pinpad_correlation_analysis.py \
        --embeddings results/pinpad_heatmap_siglip2/grid_embeddings_siglip2.pt \
        --output results/pinpad_correlation

    # Or using heatmap JSON only (for visualization of existing results):
    python scripts/visualization/pinpad_correlation_analysis.py \
        --heatmap-dir results/pinpad_heatmap_siglip2 \
        --output results/pinpad_correlation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Allow imports from same directory when run from project root
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import stable_worldmodel.envs  # noqa: F401 - registers envs

from pinpad_distance_utils import (
    compute_ground_truth_distances,
    compute_ground_truth_distances_to_point,
    get_goal_bounds_for_task,
)
from pinpad_heatmap import load_embedding_model, encode_text


# Map color keyword in prompt to pad char (legacy, for colored squares)
PROMPT_COLOR_TO_PAD = {
    "red": "1", "green": "2", "blue": "3", "yellow": "4",
    "magenta": "5", "cyan": "6", "purple": "7",
}

# Map food keyword in prompt to pad char (for image-based env: 1=apple, 2=hamburger, 3=lemon, ...)
PROMPT_FOOD_TO_PAD = {
    "apple": "1", "hamburger": "2", "lemon": "3", "pear": "4",
    "pizza": "5", "taco": "6",
}


def goal_prompt_to_pad(prompt: str, bounds: dict) -> Optional[str]:
    """Infer pad from prompt (food or color keyword); fallback to first pad if no match."""
    prompt_lower = prompt.lower()
    # Prefer food names (image-based env)
    for food, pad in PROMPT_FOOD_TO_PAD.items():
        if food in prompt_lower and pad in bounds:
            return pad
    # Fallback to colors (legacy)
    for color, pad in PROMPT_COLOR_TO_PAD.items():
        if color in prompt_lower and pad in bounds:
            return pad
    return list(bounds.keys())[0] if bounds else None


def load_data_from_embeddings(embeddings_path: str, task: str = "three"):
    """Load positions and embeddings from grid_embeddings.pt."""
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    positions = data["valid_positions"]
    embeddings = data["embeddings"]
    env_type = data.get("env_type", "pinpad")
    goal_positions = data.get("goal_positions", [])
    return positions, embeddings, data.get("model", ""), env_type, goal_positions


def load_data_from_heatmap_json(heatmap_dir: str, goal_idx: int, task: str = "three"):
    """
    Load positions and L2 distances from heatmap JSON.
    Only provides L2 for the specific prompt - cannot recompute for new prompts.
    Returns (positions, l2_dists, model, env_type, goal_position).
    """
    json_path = Path(heatmap_dir) / f"heatmap_goal_{goal_idx}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Heatmap JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    surf = data["surface_data"]
    x_grid = np.array(surf["x"])
    y_grid = np.array(surf["y"])
    z_grid = np.array(surf["z"])

    # scale_factor used in heatmap: Z_grid = Z * 0.5 for valid, 0 for NaN
    scale_factor = 0.5
    raw_z = z_grid / scale_factor

    # Valid positions: z was non-NaN (saved as > 0 after scaling)
    valid_mask = raw_z > 0.01
    positions = []
    l2_dists = []
    for i in range(raw_z.shape[0]):
        for j in range(raw_z.shape[1]):
            if valid_mask[i, j]:
                positions.append((float(x_grid[i, j]), float(y_grid[i, j])))
                l2_dists.append(raw_z[i, j])

    env_type = meta.get("environment_name", "pinpad")
    goal_position = meta.get("goal_position")
    return positions, np.array(l2_dists), meta.get("model", ""), env_type, goal_position


def plot_correlation(
    gt_dist: np.ndarray,
    l2_dist: np.ndarray,
    output_path: Path,
    prompt: str,
    goal_name: str,
    pearson: float,
    spearman: float,
):
    """Create scatter plot and optional additional plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt

    # Scatter: ground truth vs L2
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(gt_dist, l2_dist, alpha=0.4, s=15)

    # Linear fit
    mask = np.isfinite(gt_dist) & np.isfinite(l2_dist)
    if mask.sum() > 2:
        slope, intercept, r_val, _, _ = stats.linregress(gt_dist[mask], l2_dist[mask])
        x_line = np.linspace(gt_dist.min(), gt_dist.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"Fit (r={r_val:.3f})")

    ax.set_xlabel("Ground Truth Distance (to closest point on square)")
    ax.set_ylabel("L2 Embedding Distance")
    ax.set_title(f"{goal_name}\nPearson={pearson:.4f}, Spearman={spearman:.4f}\n\"{prompt_short}\"")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def run_correlation_from_embeddings(
    embeddings_path: str,
    goal_prompts: list[str],
    task: str,
    model_name: str,
    output_dir: str,
    device: str,
):
    """Run full correlation analysis using embeddings (can optimize prompts later)."""
    positions, grid_embeddings, _, env_type, goal_positions = load_data_from_embeddings(
        embeddings_path, task
    )
    positions = [(float(x), float(y)) for x, y in positions]
    grid_embeddings = grid_embeddings.to(device)

    is_clip = "clip" in model_name.lower()
    model, processor = load_embedding_model(model_name, device)

    results = []
    for goal_idx, prompt in enumerate(goal_prompts):
        if env_type == "pinpad_image" and goal_positions and goal_idx < len(goal_positions):
            goal_point = tuple(goal_positions[goal_idx])
            gt_dist = compute_ground_truth_distances_to_point(positions, goal_point)
            goal_name = f"Goal {goal_idx} (point {goal_point})"
        else:
            bounds = get_goal_bounds_for_task(task)
            pad = goal_prompt_to_pad(prompt, bounds)
            if pad is None:
                continue
            goal_bounds = bounds[pad]
            gt_dist = compute_ground_truth_distances(positions, goal_bounds)
            goal_name = f"Goal {goal_idx} ({pad})"

        text_emb = encode_text(model, processor, prompt, device, is_clip)
        text_emb = text_emb.unsqueeze(0).to(device)
        l2_dist = torch.norm(grid_embeddings - text_emb, dim=1).cpu().numpy()

        pearson, _ = stats.pearsonr(gt_dist, l2_dist)
        spearman, _ = stats.spearmanr(gt_dist, l2_dist)

        print(f"{goal_name}: Pearson={pearson:.4f}, Spearman={spearman:.4f}  \"{prompt[:40]}...\"")

        out_path = Path(output_dir) / f"correlation_goal_{goal_idx}.png"
        plot_correlation(gt_dist, l2_dist, out_path, prompt, goal_name, pearson, spearman)

        results.append({
            "goal_idx": goal_idx,
            "prompt": prompt,
            "pearson": float(pearson),
            "spearman": float(spearman),
        })

    return results


def run_correlation_from_heatmap(
    heatmap_dir: str,
    goal_prompts: list[str],
    task: str,
    output_dir: str,
):
    """Run correlation using pre-computed heatmap JSON (no embeddings, no optimization)."""
    results = []
    for goal_idx, prompt in enumerate(goal_prompts):
        try:
            positions, l2_dist, _, env_type, goal_position = load_data_from_heatmap_json(
                heatmap_dir, goal_idx, task
            )
        except FileNotFoundError:
            print(f"  Skipping goal {goal_idx}: heatmap JSON not found")
            continue

        if env_type == "pinpad_image" and goal_position is not None:
            goal_point = tuple(goal_position)
            gt_dist = compute_ground_truth_distances_to_point(positions, goal_point)
            goal_name = f"Goal {goal_idx} (point {goal_point})"
        else:
            bounds = get_goal_bounds_for_task(task)
            pad = goal_prompt_to_pad(prompt, bounds)
            if pad is None:
                continue
            goal_bounds = bounds[pad]
            gt_dist = compute_ground_truth_distances(positions, goal_bounds)
            goal_name = f"Goal {goal_idx} ({pad})"

        pearson, _ = stats.pearsonr(gt_dist, l2_dist)
        spearman, _ = stats.spearmanr(gt_dist, l2_dist)

        print(f"{goal_name}: Pearson={pearson:.4f}, Spearman={spearman:.4f}  \"{prompt[:40]}...\"")

        out_path = Path(output_dir) / f"correlation_goal_{goal_idx}.png"
        plot_correlation(gt_dist, l2_dist, out_path, prompt, goal_name, pearson, spearman)

        results.append({
            "goal_idx": goal_idx,
            "prompt": prompt,
            "pearson": float(pearson),
            "spearman": float(spearman),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="PinPad correlation: ground truth distance vs L2")
    parser.add_argument("--embeddings", type=str, help="Path to grid_embeddings_*.pt")
    parser.add_argument("--heatmap-dir", type=str, help="Path to heatmap results dir (if no embeddings)")
    parser.add_argument("--task", type=str, default="three")
    parser.add_argument(
        "--goals",
        type=str,
        nargs="+",
        default=[
            "laser pointer on the cat's eye",
            "laser pointer on the cat's ear",
            "laser pointer on the cat's paws",
            "laser pointer on the cat's nose",
        ],
    )
    parser.add_argument("--model", type=str, default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--output", type=str, default="results/pinpad_correlation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.embeddings and os.path.exists(args.embeddings):
        print("Using embeddings for correlation analysis")
        results = run_correlation_from_embeddings(
            args.embeddings,
            args.goals,
            args.task,
            args.model,
            args.output,
            args.device,
        )
    elif args.heatmap_dir and os.path.exists(args.heatmap_dir):
        print("Using heatmap JSON for correlation analysis (embeddings not found)")
        results = run_correlation_from_heatmap(args.heatmap_dir, args.goals, args.task, args.output)
    else:
        print("ERROR: Provide --embeddings (path to grid_embeddings_*.pt) or --heatmap-dir")
        print("  Embeddings are created when you run pinpad_heatmap.py")
        return 1

    with open(Path(args.output) / "correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
